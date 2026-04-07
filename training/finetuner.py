from __future__ import annotations
import json
from datetime import datetime
from pathlib import Path
from typing import List, Optional
import torch
from datasets import Dataset
from peft import PeftModel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    TrainingArguments,
    Trainer,
)
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
from core.logging import get_logger
from core.settings import load_config
logger = get_logger("finetuner")
INSTRUCTION_TEMPLATE = "### Language: {language}\n### Task: Complete the following code\n### Code:\n"
RESPONSE_TEMPLATE = "\n### End"
def prepare_instruction_dataset(samples: List[dict]) -> Dataset:
    formatted = []
    for s in samples:
        lang = s.get("language", "code")
        code = s.get("text", s.get("code", ""))
        text = f"{INSTRUCTION_TEMPLATE.format(language=lang)}{code}{RESPONSE_TEMPLATE}"
        formatted.append({"text": text, "language": lang})
    return Dataset.from_list(formatted)
def finetune(
    model_path: str,
    samples: Optional[List[dict]] = None,
    dataset_path: Optional[str] = None,
    output_dir: Optional[str] = None,
    config: Optional[dict] = None,
    merge_adapters: bool = False,
) -> dict:
    from core.database import get_session, TrainingRun
    cfg = config or load_config()
    train_cfg = cfg.get("training", {})
    output_dir = output_dir or f"{model_path}_finetuned"
    run_name = f"finetune_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    session = get_session()
    db_run = TrainingRun(
        name=run_name,
        base_model=model_path,
        status="loading",
        config=cfg,
    )
    session.add(db_run)
    session.commit()
    run_id = db_run.id
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        logger.info(f"Loading model for fine-tuning from {model_path}")
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            trust_remote_code=True,
        )
        if samples:
            dataset = prepare_instruction_dataset(samples)
        elif dataset_path:
            from pipeline.dataset_builder import load_hf_dataset
            raw = load_hf_dataset(dataset_path)
            all_samples = [{"text": ex["text"], "language": ex.get("language", "code")}
                           for ex in raw["train"]]
            dataset = prepare_instruction_dataset(all_samples)
        else:
            raise ValueError("Provide either samples or dataset_path")
        db_run.total_samples = len(dataset)
        db_run.status = "training"
        db_run.started_at = datetime.now()
        session.commit()
        from peft import LoraConfig, TaskType, get_peft_model
        peft_cfg = cfg.get("peft", {})
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=peft_cfg.get("r", 16),
            lora_alpha=peft_cfg.get("lora_alpha", 32),
            lora_dropout=peft_cfg.get("lora_dropout", 0.05),
            bias=peft_cfg.get("bias", "none"),
            target_modules=peft_cfg.get("target_modules", ["q_proj", "v_proj"]),
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
        training_args = TrainingArguments(
            output_dir=output_dir,
            run_name=run_name,
            num_train_epochs=train_cfg.get("num_train_epochs", 3),
            per_device_train_batch_size=train_cfg.get("per_device_train_batch_size", 2),
            gradient_accumulation_steps=train_cfg.get("gradient_accumulation_steps", 8),
            learning_rate=train_cfg.get("learning_rate", 2e-4),
            weight_decay=train_cfg.get("weight_decay", 0.01),
            warmup_ratio=train_cfg.get("warmup_ratio", 0.05),
            lr_scheduler_type="cosine",
            logging_steps=train_cfg.get("logging_steps", 10),
            save_steps=train_cfg.get("save_steps", 200),
            save_total_limit=3,
            fp16=False,
            bf16=torch.cuda.is_available(),
            remove_unused_columns=False,
            report_to="none",
        )
        data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=dataset,
            data_collator=data_collator,
        )
        trainer.train()
        if merge_adapters:
            logger.info("Merging LoRA adapters into base model")
            merged = model.merge_and_unload()
            merged.save_pretrained(output_dir)
        else:
            trainer.save_model(output_dir)
        tokenizer.save_pretrained(output_dir)
        db_run.status = "completed"
        db_run.output_path = output_dir
        db_run.epochs_completed = int(train_cfg.get("num_train_epochs", 3))
        db_run.completed_at = datetime.now()
        session.commit()
        logger.info(f"Fine-tuning complete → {output_dir}")
        return {"run_id": run_id, "output_path": output_dir}
    except Exception as e:
        db_run.status = "failed"
        db_run.error_message = str(e)
        session.commit()
        logger.error(f"Fine-tuning failed: {e}")
        raise
    finally:
        session.close()