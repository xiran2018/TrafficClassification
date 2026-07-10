from __future__ import annotations

from typing import List, Optional

import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
try:
    from peft import PeftModel
except Exception:
    PeftModel = None


class QwenPacketEmbedder:
    """Qwen-LoRA packet embedding extractor using scheme B: last-token pooling."""

    def __init__(self, base_model: str, lora_path: str = "", dtype: str = "float16"):
        torch_dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[dtype]
        self.tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(base_model, torch_dtype=torch_dtype, device_map="auto", trust_remote_code=True)
        if lora_path:
            if PeftModel is None:
                raise RuntimeError("peft is required to load LoRA adapters")
            self.model = PeftModel.from_pretrained(self.model, lora_path)
        self.model.eval()

    @torch.no_grad()
    def encode(self, texts: List[str], max_length: int = 1024) -> np.ndarray:
        inputs = self.tokenizer(texts, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        out = self.model(**inputs, output_hidden_states=True, return_dict=True)
        h = out.hidden_states[-1]
        last_idx = inputs["attention_mask"].sum(dim=1) - 1
        emb = h[torch.arange(h.size(0), device=h.device), last_idx]
        emb = torch.nn.functional.normalize(emb.float(), p=2, dim=-1)
        return emb.cpu().numpy().astype("float32")
