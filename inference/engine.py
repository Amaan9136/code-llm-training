from __future__ import annotations
import time
from pathlib import Path
from typing import Dict, Generator, List, Optional
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from threading import Thread
from core.logging import get_logger
from core.settings import load_config
logger = get_logger("inference")
class CodeInferenceEngine:
    def __init__(self, model_path: str, config: Optional[dict] = None, device: Optional[str] = None):
        self.model_path = model_path
        self.config = config or load_config()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.tokenizer = None
        self._loaded = False
    def load(self):
        if self._loaded:
            return
        logger.info(f"Loading model from {self.model_path} on {self.device}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path, trust_remote_code=True, use_fast=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        load_kwargs = dict(
            trust_remote_code=True,
            torch_dtype=torch.bfloat16 if self.device != "cpu" else torch.float32,
        )
        if self.device == "cpu":
            load_kwargs["torch_dtype"] = torch.float32
        adapter_config = Path(self.model_path) / "adapter_config.json"
        if adapter_config.exists():
            import json
            from peft import PeftModel
            config_data = json.loads(adapter_config.read_text())
            base_model = config_data.get("base_model_name_or_path", self.model_path)
            base = AutoModelForCausalLM.from_pretrained(base_model, **load_kwargs)
            self.model = PeftModel.from_pretrained(base, self.model_path)
        else:
            self.model = AutoModelForCausalLM.from_pretrained(self.model_path, **load_kwargs)
        if self.device != "cpu":
            self.model = self.model.to(self.device)
        self.model.eval()
        self._loaded = True
        logger.info("Model loaded successfully")
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.95,
        top_k: int = 50,
        repetition_penalty: float = 1.1,
        do_sample: bool = True,
        stop_sequences: Optional[List[str]] = None,
    ) -> dict:
        if not self._loaded:
            self.load()
        infer_cfg = self.config.get("inference", {})
        max_new_tokens = max_new_tokens or infer_cfg.get("max_new_tokens", 512)
        temperature = temperature or infer_cfg.get("temperature", 0.7)
        t0 = time.time()
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1800)
        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                temperature=temperature if do_sample else 1.0,
                top_p=top_p if do_sample else 1.0,
                top_k=top_k if do_sample else 50,
                repetition_penalty=repetition_penalty,
                do_sample=do_sample,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        generated_ids = outputs[0][input_ids.shape[1]:]
        generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        if stop_sequences:
            for stop in stop_sequences:
                if stop in generated_text:
                    generated_text = generated_text[:generated_text.index(stop)]
        elapsed = time.time() - t0
        tokens_generated = len(generated_ids)
        return {
            "generated_text": generated_text,
            "prompt": prompt,
            "tokens_generated": tokens_generated,
            "time_seconds": round(elapsed, 3),
            "tokens_per_second": round(tokens_generated / elapsed, 2),
        }
    def stream(
        self,
        prompt: str,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.95,
    ) -> Generator[str, None, None]:
        if not self._loaded:
            self.load()
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1800)
        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)
        streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)
        gen_kwargs = dict(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            streamer=streamer,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        thread = Thread(target=self.model.generate, kwargs=gen_kwargs)
        thread.start()
        for token in streamer:
            yield token
        thread.join()
    def complete_code(
        self,
        code_prefix: str,
        language: str = "python",
        max_new_tokens: int = 256,
    ) -> dict:
        prompt = f"### Language: {language}\n### Task: Complete the following code\n### Code:\n{code_prefix}"
        result = self.generate(prompt, max_new_tokens=max_new_tokens, stop_sequences=["### End"])
        result["language"] = language
        result["code_prefix"] = code_prefix
        return result
    def explain_code(self, code: str, language: str = "python") -> dict:
        prompt = f"### Code:\n{code}\n### Explanation:\nThis code"
        result = self.generate(prompt, max_new_tokens=300, temperature=0.3)
        result["language"] = language
        result["generated_text"] = "This code" + result["generated_text"]
        return result
    def unload(self):
        if self._loaded:
            del self.model
            del self.tokenizer
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            self._loaded = False
            logger.info("Model unloaded")
_engine_instance: Optional[CodeInferenceEngine] = None
def get_engine(model_path: str = None, config: dict = None) -> CodeInferenceEngine:
    global _engine_instance
    if _engine_instance is None or (model_path and _engine_instance.model_path != model_path):
        if _engine_instance:
            _engine_instance.unload()
        cfg = config or load_config()
        path = model_path or cfg.get("training", {}).get("output_dir", "outputs/model")
        _engine_instance = CodeInferenceEngine(path, cfg)
        _engine_instance.load()
    return _engine_instance