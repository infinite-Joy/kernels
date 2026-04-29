#!/usr/bin/env python3
"""
=============================================================================
  Edge Deployment Readiness Evaluator for Nandi-Mini-150M
=============================================================================

Measures everything that matters when shipping a small LLM to edge devices
(phones, kiosks, Jetson, Raspberry Pi, browser, on-prem micro-servers).

Benchmarks:
  1. Memory Footprint      — peak RSS, parameter bytes, KV-cache estimate
  2. Quantization Sweep    — fp32 → fp16 → int8 → int4 size & quality impact
  3. Latency Profiling     — first-token, decode, end-to-end across devices
  4. Batch Scaling          — throughput vs batch size (edge batches are small)
  5. Context-Length Stress  — latency & memory at 128 / 256 / 512 / 1024 / 2048
  6. Tokenizer Efficiency   — Indic fertility (fewer tokens = less compute)
  7. Power / Compute Proxy  — FLOPs estimate per token
  8. ONNX Export Check       — can the model export to ONNX for edge runtimes?
  9. Edge Readiness Scorecard — single summary with pass/warn/fail verdicts

Requirements:
  pip install torch transformers accelerate sentencepiece protobuf \
              tqdm tabulate numpy psutil

Usage:
  python edge_deploy_eval.py                        # CPU eval (edge-realistic)
  python edge_deploy_eval.py --device cuda          # GPU eval
  python edge_deploy_eval.py --quick                # Fewer passes
  python edge_deploy_eval.py --skip-onnx            # Skip ONNX export test
  python edge_deploy_eval.py --output edge.json     # Save results
"""

import argparse
import gc
import json
import math
import os
import resource
import sys
import tempfile
import time
import warnings
from contextlib import contextmanager
from dataclasses import dataclass, field, asdict
from pathlib import Path

import numpy as np
import psutil
import torch
from tabulate import tabulate
from tqdm import tqdm

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

MODEL_ID = "FrontiersMind/Nandi-Mini-150M"
MODEL_NAME = "Nandi-Mini-150M"

EDGE_THRESHOLDS = {
    "model_size_mb_fp16": 500,
    "model_size_mb_int8": 300,
    "first_token_ms_cpu": 500,
    "decode_ms_per_tok_cpu": 100,
    "first_token_ms_gpu": 50,
    "decode_ms_per_tok_gpu": 15,
    "avg_indic_fertility": 3.0,
    "peak_rss_mb": 1500,
    "onnx_exportable": True,
}

INDIC_SENTENCES = {
    "English": "The quick brown fox jumps over the lazy dog near the river bank.",
    "Hindi": "भारत एक विविधताओं से भरा देश है जहाँ अनेक भाषाएँ बोली जाती हैं।",
    "Bengali": "বাংলাদেশ এবং ভারতের পশ্চিমবঙ্গে বাংলা ভাষায় কথা বলা হয়।",
    "Tamil": "தமிழ்நாட்டில் தமிழ் மொழி பேசப்படுகிறது, இது மிகவும் பழமையான மொழி.",
    "Telugu": "తెలుగు భాషలో అనేక సాహిత్య రచనలు ఉన్నాయి మరియు ఇది చాలా అందమైన భాషా.",
    "Marathi": "महाराष्ट्रात मराठी भाषा बोलली जाते आणि ती एक समृद्ध भाषा आहे.",
    "Gujarati": "ગુજરાતી ભાષા ગુજરાતમાં બોલાય છે અને તેનો સમૃદ્ધ ઇતિહાસ છે.",
    "Kannada": "ಕರ್ನಾಟಕದಲ್ಲಿ ಕನ್ನಡ ಭಾಷೆಯನ್ನು ಮಾತನಾಡಲಾಗುತ್ತದೆ ಮತ್ತು ಇದು ಪ್ರಾಚೀನ ಭಾಷೆ.",
    "Malayalam": "കേരളത്തിൽ മലയാളം ഭാഷ സംസാരിക്കുന്നു, ഇത് ദ്രാവിഡ ഭാഷാ കുടുംബത്തിൽ പെടുന്നു.",
    "Punjabi": "ਪੰਜਾਬੀ ਭਾਸ਼ਾ ਪੰਜਾਬ ਵਿੱਚ ਬੋਲੀ ਜਾਂਦੀ ਹੈ ਅਤੇ ਇਸਦਾ ਅਮੀਰ ਸਾਹਿਤ ਹੈ।",
    "Odia": "ଓଡ଼ିଆ ଭାଷା ଓଡ଼ିଶାରେ କୁହାଯାଏ ଏବଂ ଏହା ଏକ ପ୍ରାଚୀନ ଭାଷା।",
    "Assamese": "অসমীয়া ভাষা অসমত কোৱা হয় আৰু ই এটা অতি পুৰণি ভাষা।",
}

CONTEXT_LENGTHS = [128, 256, 512, 1024, 2048]
BATCH_SIZES = [1, 2, 4, 8]

QUALITY_TEXTS = {
    "English": (
        "India is the world's largest democracy and the seventh-largest country by area. "
        "It is a land of incredible diversity, with hundreds of languages spoken across "
        "its states and territories. The Indian economy has grown rapidly in recent decades."
    ),
    "Hindi": (
        "भारत विश्व का सबसे बड़ा लोकतंत्र है और क्षेत्रफल के हिसाब से सातवां सबसे बड़ा देश है। "
        "यह अविश्वसनीय विविधता की भूमि है, जहाँ इसके राज्यों और केंद्र शासित प्रदेशों में "
        "सैकड़ों भाषाएँ बोली जाती हैं।"
    ),
    "Tamil": (
        "இந்தியா உலகின் மிகப்பெரிய ஜனநாயக நாடு ஆகும். இது பல்வேறு மொழிகள் "
        "மற்றும் கலாச்சாரங்களைக் கொண்ட நாடு."
    ),
}


# ─────────────────────────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────────────────────────

@contextmanager
def timer():
    start = time.perf_counter()
    elapsed = lambda: (time.perf_counter() - start) * 1000
    yield elapsed


def get_peak_rss_mb():
    return psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)


def param_bytes(model, dtype_name="fp16"):
    multiplier = {"fp32": 4, "fp16": 2, "int8": 1, "int4": 0.5}
    n_params = sum(p.numel() for p in model.parameters())
    return n_params * multiplier.get(dtype_name, 2)


def param_size_mb(model, dtype_name="fp16"):
    return param_bytes(model, dtype_name) / (1024 * 1024)


def estimate_kv_cache_mb(config, seq_len, batch=1, dtype_bytes=2):
    n_layers = getattr(config, "num_hidden_layers", getattr(config, "n_layer", 0))
    n_kv = getattr(config, "num_key_value_heads", getattr(config, "num_attention_heads", 0))
    head_dim = getattr(config, "hidden_size", 0) // max(
        getattr(config, "num_attention_heads", 1), 1
    )
    # 2 tensors (K, V) per layer
    kv_bytes = 2 * n_layers * n_kv * head_dim * seq_len * batch * dtype_bytes
    return kv_bytes / (1024 * 1024)


def estimate_flops_per_token(config):
    H = getattr(config, "hidden_size", 0)
    L = getattr(config, "num_hidden_layers", 0)
    V = getattr(config, "vocab_size", 0)
    I = getattr(config, "intermediate_size", H * 4)
    # rough: 2*H*H per attention + 2*H*I per FFN, per layer, plus embedding
    flops = L * (2 * H * H + 2 * H * I) + 2 * H * V
    return flops


def print_section(title):
    w = 78
    print(f"\n{'='*w}")
    print(f"  {title}")
    print(f"{'='*w}\n")


def verdict(value, threshold, lower_better=True):
    if lower_better:
        if value <= threshold:
            return "PASS"
        elif value <= threshold * 1.5:
            return "WARN"
        return "FAIL"
    else:
        if value >= threshold:
            return "PASS"
        elif value >= threshold * 0.5:
            return "WARN"
        return "FAIL"


# ─────────────────────────────────────────────────────────────────────────────
# 1. Memory Footprint
# ─────────────────────────────────────────────────────────────────────────────

def benchmark_memory(model, config):
    print_section("1. MEMORY FOOTPRINT")

    rows = []
    for dtype_name in ["fp32", "fp16", "int8", "int4"]:
        size = param_size_mb(model, dtype_name)
        rows.append([dtype_name.upper(), f"{size:.1f} MB"])

    kv_rows = []
    for seq in CONTEXT_LENGTHS:
        kv = estimate_kv_cache_mb(config, seq)
        kv_rows.append([seq, f"{kv:.2f} MB"])

    peak_rss = get_peak_rss_mb()

    print("  Model weights (estimated):")
    print(tabulate(rows, headers=["Precision", "Size"], tablefmt="rounded_grid"))
    print(f"\n  KV-cache per request (fp16, batch=1):")
    print(tabulate(kv_rows, headers=["Context Length", "KV Cache"], tablefmt="rounded_grid"))
    print(f"\n  Current peak RSS: {peak_rss:.0f} MB")

    return {
        "weight_sizes": {r[0]: r[1] for r in rows},
        "kv_cache": {str(r[0]): r[1] for r in kv_rows},
        "peak_rss_mb": round(peak_rss, 1),
    }


# ─────────────────────────────────────────────────────────────────────────────
# 2. Quantization Quality Sweep
# ─────────────────────────────────────────────────────────────────────────────

def benchmark_quantization(model, tokenizer, device):
    print_section("2. QUANTIZATION SWEEP (perplexity impact)")

    text = QUALITY_TEXTS["English"] + " " + QUALITY_TEXTS["Hindi"]
    encodings = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    input_ids = encodings["input_ids"].to(model.device)

    results = []

    # fp16 / fp32 baseline (whatever model is loaded as)
    with torch.no_grad():
        out = model(input_ids, labels=input_ids)
        base_loss = out.loss.item()
    base_ppl = min(math.exp(base_loss), 1e7)
    loaded_dtype = next(model.parameters()).dtype
    results.append([str(loaded_dtype).replace("torch.", ""), f"{base_ppl:.2f}", "baseline"])

    # Simulate int8 via round-trip quantization of weights
    for bits, label in [(8, "int8-simulated"), (4, "int4-simulated")]:
        try:
            model_copy = type(model)(model.config).to(model.device)
            model_copy.load_state_dict(model.state_dict())
            with torch.no_grad():
                for p in model_copy.parameters():
                    if p.dtype in (torch.float16, torch.float32, torch.bfloat16):
                        scale = p.abs().max() / (2 ** (bits - 1) - 1)
                        if scale > 0:
                            p.data = (torch.round(p.data / scale) * scale)
                out = model_copy(input_ids, labels=input_ids)
                loss = out.loss.item()
            ppl = min(math.exp(loss), 1e7)
            delta = ((ppl - base_ppl) / base_ppl) * 100
            results.append([label, f"{ppl:.2f}", f"+{delta:.1f}%"])
            del model_copy
            gc.collect()
            if "cuda" in str(device):
                torch.cuda.empty_cache()
        except Exception as e:
            results.append([label, "error", str(e)[:40]])

    print(tabulate(results, headers=["Precision", "Perplexity", "Δ vs baseline"], tablefmt="rounded_grid"))
    return results


# ─────────────────────────────────────────────────────────────────────────────
# 3. Latency Profiling
# ─────────────────────────────────────────────────────────────────────────────

def benchmark_latency(model, tokenizer, device, warmup=2, runs=5, gen_tokens=32):
    print_section("3. LATENCY PROFILING")

    prompt = "India is a diverse country with many languages and"
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to(model.device)
    attn_mask = inputs.get("attention_mask")
    if attn_mask is not None:
        attn_mask = attn_mask.to(model.device)

    prompt_len = input_ids.shape[1]

    # Warmup
    for _ in range(warmup):
        with torch.no_grad():
            model.generate(input_ids, attention_mask=attn_mask, max_new_tokens=5, do_sample=False)

    # First-token latency (prefill)
    prefill_times = []
    for _ in range(runs):
        if "cuda" in str(model.device):
            torch.cuda.synchronize()
        with timer() as elapsed:
            with torch.no_grad():
                model(input_ids)
        if "cuda" in str(model.device):
            torch.cuda.synchronize()
        prefill_times.append(elapsed())

    # Full generation
    gen_times = []
    for _ in range(runs):
        if "cuda" in str(model.device):
            torch.cuda.synchronize()
        with timer() as elapsed:
            with torch.no_grad():
                out = model.generate(
                    input_ids, attention_mask=attn_mask,
                    max_new_tokens=gen_tokens, do_sample=False,
                )
        if "cuda" in str(model.device):
            torch.cuda.synchronize()
        actual_gen = out.shape[1] - prompt_len
        gen_times.append((elapsed(), actual_gen))

    avg_prefill = np.mean(prefill_times)
    avg_total = np.mean([t[0] for t in gen_times])
    avg_gen_tok = np.mean([t[1] for t in gen_times])
    decode_per_tok = (avg_total - avg_prefill) / max(avg_gen_tok - 1, 1)
    tps = (avg_gen_tok / avg_total) * 1000 if avg_total > 0 else 0

    rows = [
        ["First token (prefill)", f"{avg_prefill:.1f} ms"],
        ["Decode per token", f"{decode_per_tok:.1f} ms"],
        ["End-to-end ({:.0f} tokens)".format(avg_gen_tok), f"{avg_total:.0f} ms"],
        ["Throughput", f"{tps:.1f} tok/s"],
    ]
    print(tabulate(rows, headers=["Metric", "Value"], tablefmt="rounded_grid"))

    return {
        "first_token_ms": round(avg_prefill, 1),
        "decode_ms_per_tok": round(decode_per_tok, 1),
        "total_ms": round(avg_total, 0),
        "tokens_per_sec": round(tps, 1),
        "gen_tokens": int(avg_gen_tok),
        "device": device,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 4. Batch Scaling
# ─────────────────────────────────────────────────────────────────────────────

def benchmark_batch_scaling(model, tokenizer, device, gen_tokens=16):
    print_section("4. BATCH SCALING (throughput vs batch size)")

    prompt = "India is a country known for its"
    single = tokenizer(prompt, return_tensors="pt", padding=True)

    rows = []
    for bs in BATCH_SIZES:
        try:
            batch_ids = single["input_ids"].repeat(bs, 1).to(model.device)
            batch_mask = single["attention_mask"].repeat(bs, 1).to(model.device)

            # Warmup
            with torch.no_grad():
                model.generate(batch_ids, attention_mask=batch_mask, max_new_tokens=4, do_sample=False)

            if "cuda" in str(model.device):
                torch.cuda.synchronize()
            with timer() as elapsed:
                with torch.no_grad():
                    model.generate(batch_ids, attention_mask=batch_mask,
                                   max_new_tokens=gen_tokens, do_sample=False)
            if "cuda" in str(model.device):
                torch.cuda.synchronize()
            total = elapsed()
            tps = (bs * gen_tokens / total) * 1000

            mem = ""
            if "cuda" in str(model.device):
                mem = f"{torch.cuda.max_memory_allocated() / 1e6:.0f} MB"

            rows.append([bs, f"{total:.0f} ms", f"{tps:.1f} tok/s", mem or "—"])
        except RuntimeError as e:
            rows.append([bs, "OOM", "—", "—"])
            break

    print(tabulate(rows, headers=["Batch", "Total", "Throughput", "GPU Mem"], tablefmt="rounded_grid"))
    return rows


# ─────────────────────────────────────────────────────────────────────────────
# 5. Context-Length Stress
# ─────────────────────────────────────────────────────────────────────────────

def benchmark_context_stress(model, tokenizer, device):
    print_section("5. CONTEXT-LENGTH STRESS TEST")

    max_pos = getattr(model.config, "max_position_embeddings",
                      getattr(model.config, "n_positions", 2048))

    filler = "India is a diverse country. " * 200
    rows = []

    for ctx in CONTEXT_LENGTHS:
        if ctx > max_pos:
            rows.append([ctx, "exceeds max_pos", "—", "—"])
            continue

        tokens = tokenizer(filler, return_tensors="pt", truncation=True, max_length=ctx)
        input_ids = tokens["input_ids"].to(model.device)
        actual_len = input_ids.shape[1]

        try:
            if "cuda" in str(model.device):
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.synchronize()

            with timer() as elapsed:
                with torch.no_grad():
                    model(input_ids)
            if "cuda" in str(model.device):
                torch.cuda.synchronize()

            lat = elapsed()
            gpu_mem = ""
            if "cuda" in str(model.device):
                gpu_mem = f"{torch.cuda.max_memory_allocated() / 1e6:.0f} MB"

            rows.append([actual_len, f"{lat:.0f} ms", f"{lat/actual_len:.2f} ms/tok", gpu_mem or "—"])
        except RuntimeError:
            rows.append([actual_len, "OOM", "—", "—"])
            break

    print(tabulate(rows, headers=["Tokens", "Prefill", "Per-Token", "GPU Mem"], tablefmt="rounded_grid"))
    return rows


# ─────────────────────────────────────────────────────────────────────────────
# 6. Tokenizer Efficiency
# ─────────────────────────────────────────────────────────────────────────────

def benchmark_tokenizer(tokenizer):
    print_section("6. TOKENIZER EFFICIENCY (Indic fertility)")

    rows = []
    fertilities = []
    for lang, text in INDIC_SENTENCES.items():
        toks = tokenizer.encode(text, add_special_tokens=False)
        words = text.split()
        fert = len(toks) / max(len(words), 1)
        fertilities.append(fert)
        rows.append([lang, len(words), len(toks), f"{fert:.2f}"])

    avg = np.mean(fertilities)
    rows.append(["AVERAGE", "—", "—", f"{avg:.2f}"])

    print(tabulate(rows, headers=["Language", "Words", "Tokens", "Fertility"], tablefmt="rounded_grid"))
    print(f"  Vocab size: {tokenizer.vocab_size:,}")

    return {"per_language": {r[0]: float(r[3]) for r in rows}, "avg": round(avg, 2)}


# ─────────────────────────────────────────────────────────────────────────────
# 7. FLOPs / Compute Proxy
# ─────────────────────────────────────────────────────────────────────────────

def benchmark_flops(config):
    print_section("7. COMPUTE COST (FLOPs per token)")

    flops = estimate_flops_per_token(config)
    gflops = flops / 1e9

    rows = [
        ["FLOPs / token", f"{flops:,.0f}"],
        ["GFLOPs / token", f"{gflops:.3f}"],
        ["Hidden size", getattr(config, "hidden_size", "?")],
        ["Layers", getattr(config, "num_hidden_layers", "?")],
        ["Intermediate size", getattr(config, "intermediate_size", "?")],
    ]
    print(tabulate(rows, headers=["Metric", "Value"], tablefmt="rounded_grid"))
    return {"flops_per_token": flops, "gflops_per_token": round(gflops, 3)}


# ─────────────────────────────────────────────────────────────────────────────
# 8. ONNX Export Check
# ─────────────────────────────────────────────────────────────────────────────

def benchmark_onnx_export(model, tokenizer, device):
    print_section("8. ONNX EXPORT CHECK")

    try:
        import onnx  # noqa: F401
    except ImportError:
        print("  onnx not installed — install with: pip install onnx")
        print("  Skipping ONNX export test.\n")
        return {"exportable": None, "reason": "onnx not installed"}

    dummy = tokenizer("Hello", return_tensors="pt")
    input_ids = dummy["input_ids"].to(model.device)

    with tempfile.TemporaryDirectory() as tmpdir:
        onnx_path = os.path.join(tmpdir, "model.onnx")
        try:
            cpu_model = model.cpu()
            cpu_ids = input_ids.cpu()

            torch.onnx.export(
                cpu_model,
                (cpu_ids,),
                onnx_path,
                input_names=["input_ids"],
                output_names=["logits"],
                dynamic_axes={"input_ids": {0: "batch", 1: "seq"}, "logits": {0: "batch", 1: "seq"}},
                opset_version=17,
            )
            size_mb = os.path.getsize(onnx_path) / (1024 * 1024)
            print(f"  ONNX export: SUCCESS")
            print(f"  ONNX file size: {size_mb:.1f} MB")

            model.to(device)
            return {"exportable": True, "size_mb": round(size_mb, 1)}
        except Exception as e:
            model.to(device)
            reason = str(e)[:200]
            print(f"  ONNX export: FAILED")
            print(f"  Reason: {reason}")
            return {"exportable": False, "reason": reason}


# ─────────────────────────────────────────────────────────────────────────────
# 9. Edge Readiness Scorecard
# ─────────────────────────────────────────────────────────────────────────────

def display_scorecard(memory, latency, fertility, flops, onnx_result, device):
    print_section("9. EDGE READINESS SCORECARD")

    is_gpu = "cuda" in device
    suffix = "gpu" if is_gpu else "cpu"

    checks = []

    # Model size
    fp16_mb = float(memory["weight_sizes"]["FP16"].replace(" MB", ""))
    v = verdict(fp16_mb, EDGE_THRESHOLDS["model_size_mb_fp16"])
    checks.append(["Model size (fp16)", f"{fp16_mb:.0f} MB", f"≤ {EDGE_THRESHOLDS['model_size_mb_fp16']} MB", v])

    int8_mb = float(memory["weight_sizes"]["INT8"].replace(" MB", ""))
    v = verdict(int8_mb, EDGE_THRESHOLDS["model_size_mb_int8"])
    checks.append(["Model size (int8)", f"{int8_mb:.0f} MB", f"≤ {EDGE_THRESHOLDS['model_size_mb_int8']} MB", v])

    # Latency
    ft_thresh = EDGE_THRESHOLDS[f"first_token_ms_{suffix}"]
    v = verdict(latency["first_token_ms"], ft_thresh)
    checks.append(["First-token latency", f"{latency['first_token_ms']:.0f} ms", f"≤ {ft_thresh} ms", v])

    dec_thresh = EDGE_THRESHOLDS[f"decode_ms_per_tok_{suffix}"]
    v = verdict(latency["decode_ms_per_tok"], dec_thresh)
    checks.append(["Decode latency", f"{latency['decode_ms_per_tok']:.1f} ms/tok", f"≤ {dec_thresh} ms", v])

    # Fertility
    v = verdict(fertility["avg"], EDGE_THRESHOLDS["avg_indic_fertility"])
    checks.append(["Indic fertility (avg)", f"{fertility['avg']:.2f}", f"≤ {EDGE_THRESHOLDS['avg_indic_fertility']}", v])

    # Peak RSS
    v = verdict(memory["peak_rss_mb"], EDGE_THRESHOLDS["peak_rss_mb"])
    checks.append(["Peak RSS", f"{memory['peak_rss_mb']:.0f} MB", f"≤ {EDGE_THRESHOLDS['peak_rss_mb']} MB", v])

    # ONNX
    if onnx_result["exportable"] is True:
        checks.append(["ONNX exportable", "Yes", "Yes", "PASS"])
    elif onnx_result["exportable"] is False:
        checks.append(["ONNX exportable", "No", "Yes", "FAIL"])
    else:
        checks.append(["ONNX exportable", "untested", "Yes", "WARN"])

    # FLOPs
    gf = flops["gflops_per_token"]
    checks.append(["GFLOPs/token", f"{gf:.3f}", "info", "INFO"])

    print(tabulate(checks, headers=["Check", "Measured", "Threshold", "Verdict"], tablefmt="rounded_grid"))

    n_pass = sum(1 for c in checks if c[3] == "PASS")
    n_warn = sum(1 for c in checks if c[3] == "WARN")
    n_fail = sum(1 for c in checks if c[3] == "FAIL")
    n_scored = n_pass + n_warn + n_fail

    print(f"\n  Results: {n_pass} PASS / {n_warn} WARN / {n_fail} FAIL  (out of {n_scored} scored checks)")

    if n_fail == 0 and n_warn == 0:
        print("  Verdict: EDGE-READY — model fits typical edge constraints.")
    elif n_fail == 0:
        print("  Verdict: EDGE-POSSIBLE — passes core checks with some warnings.")
    else:
        print("  Verdict: NOT EDGE-READY — one or more hard failures.")
    print()

    return checks


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Edge Deployment Readiness Evaluator for Nandi-Mini-150M")
    parser.add_argument("--device", default="cpu", help="cpu, cuda, cuda:0, etc.")
    parser.add_argument("--quick", action="store_true", help="Fewer benchmark passes")
    parser.add_argument("--skip-onnx", action="store_true", help="Skip ONNX export test")
    parser.add_argument("--skip-quantization", action="store_true", help="Skip quantization sweep")
    parser.add_argument("--output", type=str, default=None, help="Save results JSON")
    args = parser.parse_args()

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        device = "cpu"

    dtype = torch.float16 if "cuda" in device else torch.float32

    print(f"\n{'='*78}")
    print(f"  Edge Deployment Readiness — {MODEL_NAME}")
    print(f"{'='*78}")
    print(f"  Device: {device} | Dtype: {dtype}")
    if "cuda" in device:
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print()

    # Load model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"  Loading {MODEL_NAME} ({MODEL_ID})...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, trust_remote_code=True, torch_dtype=dtype, low_cpu_mem_usage=True,
    )
    if device != "auto":
        model = model.to(device)
    model.eval()

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Loaded: {n_params:,} params\n")

    config = model.config
    all_results = {}

    # 1. Memory
    all_results["memory"] = benchmark_memory(model, config)

    # 2. Quantization
    if not args.skip_quantization:
        all_results["quantization"] = benchmark_quantization(model, tokenizer, device)

    # 3. Latency
    warmup = 1 if args.quick else 2
    runs = 3 if args.quick else 5
    all_results["latency"] = benchmark_latency(model, tokenizer, device, warmup=warmup, runs=runs)

    # 4. Batch scaling
    all_results["batch_scaling"] = benchmark_batch_scaling(model, tokenizer, device)

    # 5. Context stress
    all_results["context_stress"] = benchmark_context_stress(model, tokenizer, device)

    # 6. Tokenizer
    all_results["fertility"] = benchmark_tokenizer(tokenizer)

    # 7. FLOPs
    all_results["flops"] = benchmark_flops(config)

    # 8. ONNX
    if not args.skip_onnx:
        all_results["onnx"] = benchmark_onnx_export(model, tokenizer, device)
    else:
        all_results["onnx"] = {"exportable": None, "reason": "skipped"}

    # 9. Scorecard
    all_results["scorecard"] = display_scorecard(
        all_results["memory"], all_results["latency"],
        all_results["fertility"], all_results["flops"],
        all_results["onnx"], device,
    )

    # Save
    if args.output:
        serializable = {}
        for k, v in all_results.items():
            if isinstance(v, (dict, list)):
                serializable[k] = v
            else:
                serializable[k] = str(v)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(serializable, f, indent=2, ensure_ascii=False, default=str)
        print(f"  Results saved to: {args.output}")

    # Cleanup
    del model, tokenizer
    gc.collect()
    if "cuda" in device:
        torch.cuda.empty_cache()

    print("\nDone.\n")


if __name__ == "__main__":
    main()
