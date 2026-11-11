"""The four systems under test: pinned revisions, loading, and one generation function.

Same treatment for every model (pre-registration, section 3): FP16, no quantisation, greedy
decoding, the model's own chat template, thinking off. Heavy imports (torch, transformers) happen
inside the functions that need them, so tests and CI never load them.
"""

from __future__ import annotations

import gc
import re
import shutil
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ModelSpec:
    key: str
    hf_id: str
    revision: str  # pinned in docs/PREREGISTRATION.md
    loaders: tuple[str, ...]  # classes to try, in order
    use_processor: bool = False  # multimodal models template through their processor
    template_kwargs: Mapping[str, Any] = field(default_factory=lambda: {"enable_thinking": False})


MODELS: dict[str, ModelSpec] = {
    "amalia": ModelSpec(
        "amalia",
        "amalia-llm/AMALIA-9B-0626-DPO",
        "c4614b9b5c7b4fe303fa20902e692fc7dface640",
        ("causal_lm",),
    ),
    "eurollm": ModelSpec(
        "eurollm",
        "utter-project/EuroLLM-9B-Instruct-2512",
        "def82454026b0353f12e48a0de081a9ed380d91e",
        ("causal_lm",),
    ),
    "gemma": ModelSpec(
        "gemma",
        "google/gemma-4-12B-it",
        "43c9d014b231474fa7212dbb48a762566c2b255d",
        ("multimodal_lm", "image_text_to_text"),
        use_processor=True,
    ),
    "qwen": ModelSpec(
        "qwen",
        "Qwen/Qwen3.5-9B",
        "c202236235762e1c871ad0ccb60c8ee5ba337b9a",
        ("multimodal_lm", "image_text_to_text", "causal_lm"),
        use_processor=True,
    ),
}


@dataclass(frozen=True)
class GenerationConfig:
    max_new_tokens: int = 400  # pre-registration, section 3; greedy, so no sampling settings


@dataclass(frozen=True)
class GenerationResult:
    text: str  # decoded without special tokens, otherwise untouched
    n_input_tokens: int
    n_new_tokens: int
    finish: str  # "stop", or "length" when the token limit cut the answer short


def clean_output(text: str) -> str:
    """The answer proper: thinking blocks removed, whitespace trimmed. The raw text is kept too."""
    if "</think>" in text:
        text = text.rsplit("</think>", 1)[1]
    elif "<think>" in text:
        text = text.split("<think>", 1)[0]  # an unclosed block is reasoning, not an answer
    return re.sub(r"^\s+|\s+$", "", text)


# --- loading and generating (needs torch and transformers; not exercised by the offline tests) ---


@dataclass
class LoadedModel:
    spec: ModelSpec
    model: Any
    tokenizer: Any
    templater: Any  # the tokenizer, or the processor for multimodal models
    loader_used: str
    load_seconds: float
    load_errors: list[str]


def load_model(spec: ModelSpec, max_memory_gib: int = 13) -> LoadedModel:
    import torch
    import transformers

    classes = {
        "causal_lm": getattr(transformers, "AutoModelForCausalLM", None),
        "multimodal_lm": getattr(transformers, "AutoModelForMultimodalLM", None),
        "image_text_to_text": getattr(transformers, "AutoModelForImageTextToText", None),
    }
    start = time.perf_counter()
    tokenizer = transformers.AutoTokenizer.from_pretrained(spec.hf_id, revision=spec.revision)
    templater = (
        transformers.AutoProcessor.from_pretrained(spec.hf_id, revision=spec.revision)
        if spec.use_processor
        else tokenizer
    )
    gpus = torch.cuda.device_count()
    kwargs: dict[str, Any] = {
        "dtype": torch.float16,
        "device_map": "auto",
        "revision": spec.revision,
    }
    if gpus:
        kwargs["max_memory"] = {i: f"{max_memory_gib}GiB" for i in range(gpus)}
    errors: list[str] = []
    for name in spec.loaders:
        cls = classes.get(name)
        if cls is None:
            errors.append(f"{name}: not available in this transformers version")
            continue
        try:
            model = cls.from_pretrained(spec.hf_id, **kwargs)
        except Exception as e:  # try the next class, keep the reason
            errors.append(f"{name}: {type(e).__name__}: {str(e)[:300]}")
            gc.collect()
            continue
        model.eval()
        return LoadedModel(
            spec, model, tokenizer, templater, name, time.perf_counter() - start, errors
        )
    raise RuntimeError(f"could not load {spec.hf_id}: {errors}")


def render_prompt(loaded: LoadedModel, messages: list[dict[str, str]]) -> str:
    """The exact text the model is given, as its own chat template renders it."""
    return str(
        loaded.templater.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            **loaded.spec.template_kwargs,
        )
    )


def make_generate(
    loaded: LoadedModel, cfg: GenerationConfig | None = None
) -> Callable[[list[dict[str, str]]], GenerationResult]:
    import torch

    cfg = cfg or GenerationConfig()

    model, spec = loaded.model, loaded.spec
    eos = model.generation_config.eos_token_id or model.config.eos_token_id
    eos_ids = set(eos if isinstance(eos, list) else [eos]) if eos is not None else set()

    def generate(messages: list[dict[str, str]]) -> GenerationResult:
        inputs = loaded.templater.apply_chat_template(
            messages,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            add_generation_prompt=True,
            **spec.template_kwargs,
        ).to(model.device)
        n_in = int(inputs["input_ids"].shape[-1])
        with torch.inference_mode():
            out = model.generate(
                **inputs,
                max_new_tokens=cfg.max_new_tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
                top_k=None,
            )
        new = out[0][n_in:]
        stopped = len(new) < cfg.max_new_tokens or (len(new) > 0 and int(new[-1]) in eos_ids)
        text = loaded.tokenizer.decode(new, skip_special_tokens=True)
        return GenerationResult(text, n_in, int(len(new)), "stop" if stopped else "length")

    return generate


def gpu_peaks_gib() -> dict[str, float]:
    import torch

    return {
        f"cuda:{i}": round(torch.cuda.max_memory_reserved(i) / 2**30, 2)
        for i in range(torch.cuda.device_count())
    }


def reset_gpu_peaks() -> None:
    import torch

    for i in range(torch.cuda.device_count()):
        torch.cuda.reset_peak_memory_stats(i)


def unload(loaded: LoadedModel) -> None:
    """Free GPU memory and delete the model's files from the Hugging Face cache (disk is scarce)."""
    import torch

    spec = loaded.spec
    del loaded.model
    gc.collect()
    torch.cuda.empty_cache()
    from huggingface_hub.constants import HF_HUB_CACHE

    shutil.rmtree(
        Path(HF_HUB_CACHE) / f"models--{spec.hf_id.replace('/', '--')}", ignore_errors=True
    )
