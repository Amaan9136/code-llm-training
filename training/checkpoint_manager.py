from __future__ import annotations
import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import List, Optional
from core.logging import get_logger
logger = get_logger("checkpoint_manager")
def list_checkpoints(output_dir: str) -> List[dict]:
    output_path = Path(output_dir)
    checkpoints = []
    for d in sorted(output_path.glob("checkpoint-*")):
        if d.is_dir():
            trainer_state = d / "trainer_state.json"
            step = int(d.name.split("-")[-1])
            loss = None
            if trainer_state.exists():
                try:
                    state = json.loads(trainer_state.read_text())
                    log_history = state.get("log_history", [])
                    losses = [e.get("loss") for e in log_history if "loss" in e]
                    if losses:
                        loss = losses[-1]
                except Exception:
                    pass
            checkpoints.append({"path": str(d), "step": step, "loss": loss})
    return sorted(checkpoints, key=lambda x: x["step"])
def get_best_checkpoint(output_dir: str) -> Optional[str]:
    output_path = Path(output_dir)
    trainer_state = output_path / "trainer_state.json"
    if trainer_state.exists():
        try:
            state = json.loads(trainer_state.read_text())
            best = state.get("best_model_checkpoint")
            if best and Path(best).exists():
                return best
        except Exception:
            pass
    checkpoints = list_checkpoints(output_dir)
    if not checkpoints:
        return None
    valid = [c for c in checkpoints if c["loss"] is not None]
    if valid:
        return min(valid, key=lambda x: x["loss"])["path"]
    return checkpoints[-1]["path"]
def export_model(model_path: str, export_dir: str, merge_peft: bool = True) -> str:
    export_path = Path(export_dir)
    export_path.mkdir(parents=True, exist_ok=True)
    adapter_config = Path(model_path) / "adapter_config.json"
    if merge_peft and adapter_config.exists():
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer
        logger.info("Merging PEFT adapters before export")
        try:
            config_data = json.loads(adapter_config.read_text())
            base_model_name = config_data.get("base_model_name_or_path", model_path)
            tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
            base_model = AutoModelForCausalLM.from_pretrained(
                base_model_name,
                torch_dtype=torch.float16,
                trust_remote_code=True,
            )
            model = PeftModel.from_pretrained(base_model, model_path)
            merged = model.merge_and_unload()
            merged.save_pretrained(str(export_path))
            tokenizer.save_pretrained(str(export_path))
            logger.info(f"Exported merged model to {export_path}")
            return str(export_path)
        except Exception as e:
            logger.warning(f"Merge failed, copying directory: {e}")
    shutil.copytree(model_path, str(export_path), dirs_exist_ok=True)
    logger.info(f"Exported model to {export_path}")
    return str(export_path)
def cleanup_old_checkpoints(output_dir: str, keep_last: int = 3):
    checkpoints = list_checkpoints(output_dir)
    to_delete = checkpoints[:-keep_last]
    for ckpt in to_delete:
        shutil.rmtree(ckpt["path"], ignore_errors=True)
        logger.info(f"Removed old checkpoint {ckpt['path']}")