"""
FastAPI Serving Endpoint for EdgeTune compressed LLM deployments.
"""

import time
from contextlib import asynccontextmanager
from typing import Any

import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from edgetune.benchmark import get_peak_memory_mb
from edgetune.config import load_base_config
from edgetune.model_loader import (
    get_model_size_mb,
    load_model_and_tokenizer,
)
from edgetune.schemas import GenerationRequest, GenerationResponse

# Global model state
MODEL_STATE: dict[str, Any] = {
    "model": None,
    "tokenizer": None,
    "device": None,
    "variant_name": "Qwen2.5-0.5B-Combined-Compressed",
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("[API] Initializing EdgeTune model server...")
    try:
        base_cfg = load_base_config("configs/base_model.yaml")
        model, tokenizer, device = load_model_and_tokenizer(base_cfg["model"])
        MODEL_STATE["model"] = model
        MODEL_STATE["tokenizer"] = tokenizer
        MODEL_STATE["device"] = device
        print(f"[API] Server ready on device: {device.type}")
    except (RuntimeError, ValueError, OSError, AttributeError, KeyError) as e:
        print(f"[API] Initialization warning (model not loaded): {e}")
    yield
    print("[API] Shutting down EdgeTune model server...")
    MODEL_STATE["model"] = None
    MODEL_STATE["tokenizer"] = None
    MODEL_STATE["device"] = None


app = FastAPI(
    title="EdgeTune Model Serving API",
    description="Production-grade endpoint serving PEFT fine-tuned, pruned, and quantized SLMs",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health_check():
    return {"status": "ok", "service": "edgetune-api"}


@app.get("/model-info")
def get_model_info():
    model = MODEL_STATE["model"]
    device = MODEL_STATE["device"]
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet.")

    size_mb = get_model_size_mb(model)
    peak_mem_mb = get_peak_memory_mb(device)

    return {
        "variant_name": MODEL_STATE["variant_name"],
        "base_model": getattr(model.config, "_name_or_path", "unknown"),
        "device": str(device.type),
        "model_size_mb": size_mb,
        "peak_memory_mb": peak_mem_mb,
        "vocab_size": getattr(model.config, "vocab_size", 0),
    }


@app.post("/generate", response_model=GenerationResponse)
def generate(req: GenerationRequest):
    model = MODEL_STATE["model"]
    tokenizer = MODEL_STATE["tokenizer"]
    device = MODEL_STATE["device"]

    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="Model server unavailable.")

    if not req.prompt or not req.prompt.strip():
        raise HTTPException(status_code=400, detail="Prompt must not be empty.")

    try:
        inputs = tokenizer(req.prompt, return_tensors="pt").to(device)
        input_len = inputs["input_ids"].shape[1]

        # Measure TTFT (Time-To-First-Token)
        t_start = time.perf_counter()
        with torch.no_grad():
            _ = model.generate(**inputs, max_new_tokens=1, do_sample=False)
        if device.type == "cuda":
            torch.cuda.synchronize()
        elif device.type == "mps":
            torch.mps.synchronize()
        ttft_ms = round((time.perf_counter() - t_start) * 1000.0, 2)

        # Measure Full Generation
        t_gen_start = time.perf_counter()
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=req.max_new_tokens,
                temperature=req.temperature,
                top_p=req.top_p,
                do_sample=(req.temperature > 0),
            )
        if device.type == "cuda":
            torch.cuda.synchronize()
        elif device.type == "mps":
            torch.mps.synchronize()
        t_gen_end = time.perf_counter()

        gen_tokens = outputs.shape[1] - input_len
        gen_duration = max(t_gen_end - t_gen_start, 1e-5)
        tps = round(gen_tokens / gen_duration, 2)

        generated_text = tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True)
        peak_mem = get_peak_memory_mb(device)

        return GenerationResponse(
            generated_text=generated_text.strip(),
            time_to_first_token_ms=ttft_ms,
            tokens_per_second=tps,
            total_tokens=gen_tokens,
            peak_memory_mb=peak_mem,
            model_variant=MODEL_STATE["variant_name"],
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Generation failed: {e}") from e
