"""The four systems under test: pinned revisions, loading, and one generation function.

Same treatment for every model: BF16, no quantisation, greedy decoding, the model's own chat
template, thinking off. The pre-registration said FP16; AMALIA overflows FP16, so every model runs
in BF16 (docs/DEVIATIONS.md, section 3). Heavy imports (torch, transformers) happen inside the
functions that need them, so tests and CI never load them.
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


# The single precision every model runs in. FP16 (pre-registered) overflows in AMALIA.
DTYPE = "bfloat16"


@dataclass(frozen=True)
class GenerationConfig:
    max_new_tokens: int = 400  # pre-registration, section 3; greedy, so no sampling settings


@dataclass(frozen=True)
class GenerationResult:
    text: str  # decoded without special tokens, otherwise untouched
    n_input_tokens: int
    n_new_tokens: int
    finish: str  # "stop", or "length" when the token limit cut the answer short
    first_token_ids: tuple[int, ...] = ()  # the first generated ids, to diagnose empty output


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


def load_model(spec: ModelSpec, max_memory_gib: int = 13, dtype: str = DTYPE) -> LoadedModel:
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
        "dtype": getattr(torch, dtype),
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
            torch.cuda.empty_cache()
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
    spec = loaded.spec

    def generate(messages: list[dict[str, str]]) -> GenerationResult:
        model = (
            loaded.model
        )  # looked up per call, so nothing here keeps the model alive after unload
        if model is None:
            raise RuntimeError("the model has been unloaded")
        eos = model.generation_config.eos_token_id or model.config.eos_token_id
        eos_ids = set(eos if isinstance(eos, list) else [eos]) if eos is not None else set()
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
        head = tuple(int(i) for i in new[:16].tolist())
        return GenerationResult(text, n_in, int(len(new)), "stop" if stopped else "length", head)

    return generate


def _init_cuda(torch: Any) -> None:
    """Real PyTorch raises "Invalid device argument" on memory-stat calls until CUDA is initialised
    (it was only initialised by accident in the smoke run, which read GPU names first)."""
    if torch.cuda.is_available():
        torch.cuda.init()


def gpu_peaks_gib() -> dict[str, float]:
    import torch

    _init_cuda(torch)
    return {
        f"cuda:{i}": round(torch.cuda.max_memory_reserved(i) / 2**30, 2)
        for i in range(torch.cuda.device_count())
    }


def reset_gpu_peaks() -> None:
    import torch

    _init_cuda(torch)
    for i in range(torch.cuda.device_count()):
        torch.cuda.reset_peak_memory_stats(i)


def sanity_check(loaded: LoadedModel) -> dict[str, Any]:
    """One forward pass: are the logits finite, and what does the model want to say first?

    FP16 can overflow in models trained in bfloat16; non-finite logits show up as empty output.
    """
    import torch

    try:
        inputs = loaded.templater.apply_chat_template(
            [{"role": "user", "content": "Olá"}],
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            add_generation_prompt=True,
            **loaded.spec.template_kwargs,
        ).to(loaded.model.device)
        with torch.inference_mode():
            logits = loaded.model(**inputs).logits[0, -1].float()
        finite = bool(torch.isfinite(logits).all())
        top = torch.topk(logits, 5).indices.tolist() if finite else []
        return {
            "finite_logits": finite,
            "top5": [loaded.tokenizer.decode([i]) for i in top],
            "dtype": str(loaded.model.dtype),
        }
    except Exception as e:
        return {"error": f"{type(e).__name__}: {str(e)[:200]}"}


def unload(loaded: LoadedModel) -> dict[str, float]:
    """Free GPU memory, delete the model's files from the Hugging Face cache (disk is scarce), and
    report the GPU memory still allocated, which should be about zero."""
    import torch

    spec = loaded.spec
    loaded.model = None
    _init_cuda(torch)
    for _ in range(2):
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        torch.cuda.empty_cache()
    from huggingface_hub.constants import HF_HUB_CACHE

    cache = Path(HF_HUB_CACHE) / f"models--{spec.hf_id.replace('/', '--')}"
    shutil.rmtree(cache, ignore_errors=True)
    return {
        f"cuda:{i}": round(torch.cuda.memory_allocated(i) / 2**30, 2)
        for i in range(torch.cuda.device_count())
    }


def layer_report(loaded: LoadedModel) -> dict[str, Any]:
    """Where do activations get large, and where do they first stop being finite?

    One forward pass with hooks on the embeddings, every decoder layer and its attention and MLP
    blocks. The first non-finite module in execution order is where the overflow starts; the largest
    magnitudes show how close the other layers come to the FP16 limit (65504).
    """
    import re

    import torch

    stats: list[dict[str, Any]] = []
    hooks = []
    wanted = re.compile(r"(\.layers\.\d+(\.mlp|\.self_attn)?|embed_tokens)$")

    def make_hook(name: str) -> Callable[..., None]:
        def hook(module: Any, args: Any, output: Any) -> None:
            out = output[0] if isinstance(output, (tuple, list)) else output
            if torch.is_tensor(out):
                f = out.float()
                finite = bool(torch.isfinite(f).all())
                top = float(f.abs().nan_to_num(posinf=3.0e38).max())
                stats.append({"module": name, "max_abs": round(top, 1), "finite": finite})

        return hook

    try:
        inputs = loaded.templater.apply_chat_template(
            [{"role": "user", "content": "Olá"}],
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            add_generation_prompt=True,
            **loaded.spec.template_kwargs,
        ).to(loaded.model.device)
        for name, mod in loaded.model.named_modules():
            if wanted.search(name) or name.endswith(".norm"):
                hooks.append(mod.register_forward_hook(make_hook(name)))
        with torch.inference_mode():
            loaded.model(**inputs)
    except Exception as e:
        return {"error": f"{type(e).__name__}: {str(e)[:200]}", "partial": stats[:5]}
    finally:
        for h in hooks:
            h.remove()
    bad = next((s["module"] for s in stats if not s["finite"]), None)
    largest = sorted(stats, key=lambda s: -s["max_abs"])[:6]
    return {"first_non_finite": bad, "largest_activations": largest, "modules_seen": len(stats)}


def probe_dtype(spec: ModelSpec, dtype: str) -> dict[str, Any]:
    """Load `spec` in `dtype`, check it is numerically sane, and time a short generation."""
    from ptlegal_eval.prompts import build_messages

    reset_gpu_peaks()
    loaded = load_model(spec, dtype=dtype)
    res: dict[str, Any] = {}
    try:
        res = {
            "dtype": dtype,
            "loader_used": loaded.loader_used,
            "load_seconds": round(loaded.load_seconds, 1),
            "sanity": sanity_check(loaded),
            "layer_report": layer_report(loaded),
        }
        question = "Quantos dias úteis de férias tem, no mínimo, um trabalhador?"
        messages = build_messages(question, None)
        try:
            make_generate(loaded, GenerationConfig(max_new_tokens=1))(messages)  # warm-up
            start = time.perf_counter()
            r = make_generate(loaded, GenerationConfig(max_new_tokens=60))(messages)
            elapsed = time.perf_counter() - start
            res |= {
                "tokens_per_s": round(r.n_new_tokens / elapsed, 2),
                "n_new_tokens": r.n_new_tokens,
                "first_token_ids": list(r.first_token_ids[:8]),
                "sample": r.text[:200],
            }
        except Exception as e:
            res["generation_error"] = f"{type(e).__name__}: {str(e)[:200]}"
        res["peak_gib"] = gpu_peaks_gib()
    finally:
        res["allocated_after_unload_gib"] = unload(loaded)
    return res
