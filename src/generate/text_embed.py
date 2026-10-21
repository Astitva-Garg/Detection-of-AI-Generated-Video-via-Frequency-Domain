"""Encode prompts with a large T5-family text encoder once, cache to disk, free memory.

Lets 8 GB VRAM / 15 GB RAM run LTX-Video (T5-XXL) and Wan2.1 (UMT5-XXL): the encoder is
loaded alone, split between GPU and CPU by accelerate, and never co-resides with the video model.
"""
import gc
import os
from pathlib import Path

import torch


def _load(repo: str, cls_name: str):
    import transformers
    from transformers import AutoTokenizer
    cls = getattr(transformers, cls_name)
    tok = AutoTokenizer.from_pretrained(repo, subfolder="tokenizer")
    te = cls.from_pretrained(repo, subfolder="text_encoder", torch_dtype=torch.bfloat16,
                             device_map="auto",
                             max_memory={0: os.environ.get("IE643_TE_GPU_MEM", "6GiB"), "cpu": "9GiB"},
                             low_cpu_mem_usage=True)
    te.eval()
    return tok, te


@torch.no_grad()
def cache_embeddings(repo: str, cls_name: str, prompts: list[str], out_path: Path,
                     max_len: int, style: str) -> dict:
    """style='ltx' -> (embeds[max_len], mask); style='wan' -> embeds trimmed to true length then zero-padded."""
    out_path = Path(out_path)
    cache = torch.load(out_path) if out_path.exists() else {}
    todo = [p for p in dict.fromkeys(prompts) if p not in cache]
    if not todo:
        return cache
    tok, te = _load(repo, cls_name)
    dev = next(te.parameters()).device
    for p in todo:
        t = tok([p], padding="max_length", max_length=max_len, truncation=True,
                add_special_tokens=True, return_attention_mask=True, return_tensors="pt")
        emb = te(t.input_ids.to(dev), attention_mask=t.attention_mask.to(dev))[0].float().cpu()
        mask = t.attention_mask.bool()
        if style == "wan":
            n = int(mask.sum())
            emb[:, n:] = 0.0
        cache[p] = (emb.to(torch.bfloat16), mask)
    del te
    gc.collect()
    torch.cuda.empty_cache()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(cache, out_path)
    return cache
