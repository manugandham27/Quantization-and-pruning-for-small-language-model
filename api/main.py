"""
FastAPI Serving Endpoint for EdgeTune compressed LLM deployments.
"""

import time
import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import Dict, Any

from edgetune.schemas import GenerationRequest, GenerationResponse
from edgetune.config import load_base_config
from edgetune.model_loader import load_model_and_tokenizer, get_model_size_mb, get_optimal_device
from edgetune.benchmark import get_peak_memory_mb

app = FastAPI(
    title="EdgeTune Model Serving API",
    description="Production-grade endpoint serving PEFT fine-tuned, pruned, and quantized SLMs",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model state
MODEL_STATE: Dict[str, Any] = {
    "model": None,
    "tokenizer": None,
    "device": None,
    "variant_name": "Qwen2.5-0.5B-Combined-Compressed",
}


@app.on_event("startup")
def startup_event():
    print("[API] Initializing EdgeTune model server...")
    base_cfg = load_base_config("configs/base_model.yaml")
    model, tokenizer, device = load_model_and_tokenizer(base_cfg["model"])
    MODEL_STATE["model"] = model
    MODEL_STATE["tokenizer"] = tokenizer
    MODEL_STATE["device"] = device
    print(f"[API] Server ready on device: {device.type}")


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
        "base_model": model.config._name_or_path,
        "device": str(device.type),
        "model_size_mb": size_mb,
        "peak_memory_mb": peak_mem_mb,
        "vocab_size": model.config.vocab_size,
    }


@app.post("/generate", response_model=GenerationResponse)
def generate(req: GenerationRequest):
    model = MODEL_STATE["model"]
    tokenizer = MODEL_STATE["tokenizer"]
    device = MODEL_STATE["device"]

    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="Model server unavailable.")

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


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
