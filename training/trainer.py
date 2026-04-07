from __future__ import annotations
import os
from datetime import datetime
from pathlib import Path
from typing import Optional
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    TrainingArguments,
    Trainer,
)
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from core.logging import get_logger
from core.settings import load_config
from pipeline.dataset_builder import get_latest_dataset, load_hf_dataset
from pipeline.tokenizer_utils import load_tokenizer, tokenize_dataset
logger = get_logger("trainer")
def build_lora_model(model, config: dict):
    peft_cfg = config.get("peft", {})
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=peft_cfg.get("r", 16),
        lora_alpha=peft_cfg.get("lora_alpha", 32),
        lora_dropout=peft_cfg.get("lora_dropout", 0.05),
        bias=peft_cfg.get("bias", "none"),
        target_modules=peft_cfg.get("target_modules", ["q_proj", "v_proj"]),
    )
    return get_peft_model(model, lora_config)
def load_base_model(model_name: str, config: dict):
    model_cfg = config.get("model", {})
    quant_cfg = config.get("quantization", {})
    load_kwargs = dict(
        trust_remote_code=model_cfg.get("trust_remote_code", True),
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    )
    if quant_cfg.get("enabled") and torch.cuda.is_available():
        from transformers import BitsAndBytesConfig
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=quant_cfg.get("load_in_4bit", True),
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type=quant_cfg.get("bnb_4bit_quant_type", "nf4"),
            bnb_4bit_use_double_quant=quant_cfg.get("bnb_4bit_use_double_quant", True),
        )
        load_kwargs["quantization_config"] = bnb_config
    model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)
    if quant_cfg.get("enabled"):
        model = prepare_model_for_kbit_training(model)
    return model
def train(
    config: Optional[dict] = None,
    dataset_path: Optional[str] = None,
    run_name: Optional[str] = None,
    resume_from: Optional[str] = None,
) -> dict:
    from core.database import get_session, TrainingRun
    cfg = config or load_config()
    model_cfg = cfg.get("model", {})
    train_cfg = cfg.get("training", {})
    peft_cfg = cfg.get("peft", {})
    base_model = model_cfg.get("base_model", "microsoft/phi-2")
    max_length = model_cfg.get("max_length", 2048)
    output_dir = train_cfg.get("output_dir", "outputs/model")
    run_name = run_name or f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    session = get_session()
    db_run = TrainingRun(
        name=run_name,
        base_model=base_model,
        status="loading",
        config=cfg,
        total_samples=0,
    )
    session.add(db_run)
    session.commit()
    run_id = db_run.id
    try:
        dataset_path = dataset_path or get_latest_dataset()
        if not dataset_path:
            raise ValueError("No dataset found. Run dataset building first.")
        logger.info(f"Loading dataset from {dataset_path}")
        raw_dataset = load_hf_dataset(dataset_path)
        logger.info(f"Loading tokenizer for {base_model}")
        tokenizer = load_tokenizer(base_model, cfg)
        logger.info(f"Tokenizing dataset")
        tokenized = {
            split: tokenize_dataset(raw_dataset[split], tokenizer, max_length, num_proc=4)
            for split in raw_dataset
        }
        db_run.total_samples = len(tokenized.get("train", []))
        db_run.status = "training"
        db_run.started_at = datetime.now()
        session.commit()
        logger.info(f"Loading model {base_model}")
        model = load_base_model(base_model, cfg)
        if peft_cfg.get("enabled", True):
            logger.info("Applying LoRA adapter")
            model = build_lora_model(model, cfg)
            model.print_trainable_parameters()
        data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
        training_args = TrainingArguments(
            output_dir=output_dir,
            run_name=run_name,
            num_train_epochs=train_cfg.get("num_train_epochs", 3),
            per_device_train_batch_size=train_cfg.get("per_device_train_batch_size", 4),
            per_device_eval_batch_size=train_cfg.get("per_device_eval_batch_size", 4),
            gradient_accumulation_steps=train_cfg.get("gradient_accumulation_steps", 8),
            learning_rate=train_cfg.get("learning_rate", 2e-4),
            weight_decay=train_cfg.get("weight_decay", 0.01),
            warmup_ratio=train_cfg.get("warmup_ratio", 0.05),
            lr_scheduler_type=train_cfg.get("lr_scheduler_type", "cosine"),
            logging_steps=train_cfg.get("logging_steps", 10),
            eval_steps=train_cfg.get("eval_steps", 100),
            save_steps=train_cfg.get("save_steps", 200),
            save_total_limit=train_cfg.get("save_total_limit", 3),
            evaluation_strategy=train_cfg.get("evaluation_strategy", "steps"),
            load_best_model_at_end=train_cfg.get("load_best_model_at_end", True),
            fp16=train_cfg.get("fp16", False) and torch.cuda.is_available(),
            bf16=train_cfg.get("bf16", True) and torch.cuda.is_available(),
            dataloader_num_workers=train_cfg.get("dataloader_num_workers", 4),
            remove_unused_columns=train_cfg.get("remove_unused_columns", False),
            report_to=train_cfg.get("report_to", "none"),
            ddp_find_unused_parameters=train_cfg.get("ddp_find_unused_parameters", False),
        )
        eval_dataset = tokenized.get("validation") or tokenized.get("test")
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized["train"],
            eval_dataset=eval_dataset,
            data_collator=data_collator,
        )
        if resume_from:
            trainer.train(resume_from_checkpoint=resume_from)
        else:
            trainer.train()
        logger.info(f"Saving model to {output_dir}")
        trainer.save_model(output_dir)
        tokenizer.save_pretrained(output_dir)
        metrics = trainer.evaluate()
        db_run.status = "completed"
        db_run.output_path = output_dir
        db_run.metrics = metrics
        db_run.epochs_completed = int(train_cfg.get("num_train_epochs", 3))
        db_run.completed_at = datetime.now()
        session.commit()
        logger.info(f"Training complete. Model saved to {output_dir}")
        return {"run_id": run_id, "output_path": output_dir, "metrics": metrics}
    except Exception as e:
        db_run.status = "failed"
        db_run.error_message = str(e)
        session.commit()
        logger.error(f"Training failed: {e}")
        raise
    finally:
        session.close()