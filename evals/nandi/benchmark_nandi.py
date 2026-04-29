#!/usr/bin/env python3
"""
=============================================================================
  Nandi-Mini-150M Benchmarking Suite
  Compare FrontiersMind/Nandi-Mini-150M against Indian & small LLMs
=============================================================================

Benchmarks:
  1. Architecture & Memory Profiling
  2. Tokenizer Fertility (Indic languages)
  3. Inference Latency & Throughput
  4. Perplexity (English + Hindi)
  5. Indic Text Generation Quality (qualitative samples)
  6. Downstream Task: Sentiment / Classification probe

Requirements:
  pip install torch transformers accelerate sentencepiece protobuf \
              datasets tqdm tabulate numpy --break-system-packages

Usage:
  # Full benchmark (all models)
  python benchmark_nandi.py

  # Quick mode (Nandi only, fewer samples)
  python benchmark_nandi.py --quick

  # Specific models only
  python benchmark_nandi.py --models nandi smollm2

  # Skip heavy benchmarks
  python benchmark_nandi.py --skip-perplexity

  # Use GPU
  python benchmark_nandi.py --device cuda

  # Save results to JSON
  python benchmark_nandi.py --output results.json
"""

import argparse
import json
import os
import sys
import time
import warnings
from contextlib import contextmanager
from dataclasses import dataclass, field, asdict
from typing import Optional

import numpy as np
import torch
from tqdm import tqdm
from tabulate import tabulate

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

MODEL_REGISTRY = {
    "nandi": {
        "name": "Nandi-Mini-150M",
        "hf_id": "FrontiersMind/Nandi-Mini-150M",
        "trust_remote_code": True,
        "origin": "India (FrontiersMind)",
        "params": "150M",
    },
    "nandi_instruct": {
        "name": "Nandi-Mini-150M-Instruct",
        "hf_id": "FrontiersMind/Nandi-Mini-150M-Instruct",
        "trust_remote_code": True,
        "origin": "India (FrontiersMind)",
        "params": "150M",
    },
    "smollm2": {
        "name": "SmolLM2-135M",
        "hf_id": "HuggingFaceTB/SmolLM2-135M",
        "trust_remote_code": False,
        "origin": "HuggingFace",
        "params": "135M",
    },
    "smollm": {
        "name": "SmolLM-135M",
        "hf_id": "HuggingFaceTB/SmolLM-135M",
        "trust_remote_code": False,
        "origin": "HuggingFace",
        "params": "135M",
    },
    "sarvam": {
        "name": "Sarvam-1 (2B)",
        "hf_id": "sarvamai/sarvam-1",
        "trust_remote_code": False,
        "origin": "India (Sarvam AI)",
        "params": "2B",
    },
    "navarasa": {
        "name": "Navarasa-2.0 (2B)",
        "hf_id": "Telugu-LLM-Labs/Indic-gemma-2b-finetuned-sft-Navarasa-2.0",
        "trust_remote_code": False,
        "origin": "India (Telugu-LLM-Labs)",
        "params": "2B",
    },
}

# Indic language test sentences for fertility and generation
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

# Prompts for generation quality comparison
GENERATION_PROMPTS = {
    "English": "India is a country known for its",
    "Hindi": "भारत एक ऐसा देश है जहाँ",
    "Bengali": "বাংলা ভাষা হলো একটি",
    "Tamil": "இந்தியா ஒரு நாடு, அதில்",
    "Telugu": "భారతదేశం ఒక గొప్ప దేశం, దీనిలో",
}


# ─────────────────────────────────────────────────────────────────────────────
# Result dataclasses
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ModelProfile:
    name: str
    hf_id: str
    origin: str
    num_parameters: int = 0
    num_layers: int = 0
    hidden_size: int = 0
    num_attention_heads: int = 0
    num_kv_heads: int = 0
    vocab_size: int = 0
    max_position: int = 0
    model_size_mb: float = 0.0
    architecture_type: str = ""


@dataclass
class FertilityResult:
    model_name: str
    language: str
    num_chars: int = 0
    num_tokens: int = 0
    fertility: float = 0.0  # tokens per word


@dataclass
class LatencyResult:
    model_name: str
    prompt_tokens: int = 0
    gen_tokens: int = 0
    prefill_ms: float = 0.0
    decode_ms_per_token: float = 0.0
    total_ms: float = 0.0
    tokens_per_sec: float = 0.0


@dataclass
class PerplexityResult:
    model_name: str
    language: str
    perplexity: float = 0.0
    loss: float = 0.0


@dataclass
class BenchmarkResults:
    profiles: list = field(default_factory=list)
    fertility: list = field(default_factory=list)
    latency: list = field(default_factory=list)
    perplexity: list = field(default_factory=list)
    generations: dict = field(default_factory=dict)


# ─────────────────────────────────────────────────────────────────────────────
# Utility functions
# ─────────────────────────────────────────────────────────────────────────────

@contextmanager
def timer():
    """Context manager that yields a callable returning elapsed ms."""
    start = time.perf_counter()
    elapsed = lambda: (time.perf_counter() - start) * 1000
    yield elapsed


def count_parameters(model) -> int:
    return sum(p.numel() for p in model.parameters())


def model_size_mb(model) -> float:
    param_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_bytes = sum(b.numel() * b.element_size() for b in model.buffers())
    return (param_bytes + buffer_bytes) / (1024 * 1024)


def safe_load_model(hf_id, trust_remote_code, device, dtype=torch.float32):
    """Load model with fallback strategies."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"  Loading tokenizer: {hf_id}")
    tokenizer = AutoTokenizer.from_pretrained(
        hf_id, trust_remote_code=trust_remote_code
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"  Loading model: {hf_id}")
    model = AutoModelForCausalLM.from_pretrained(
        hf_id,
        trust_remote_code=trust_remote_code,
        torch_dtype=dtype,
        device_map=device if device != "cpu" else None,
        low_cpu_mem_usage=True,
    )
    if device == "cpu" or (device != "auto" and "cuda" not in str(device)):
        model = model.to(device)
    model.eval()

    return model, tokenizer


def extract_model_profile(model, tokenizer, meta: dict) -> ModelProfile:
    """Extract architecture details from model config."""
    config = model.config
    profile = ModelProfile(
        name=meta["name"],
        hf_id=meta["hf_id"],
        origin=meta["origin"],
    )
    profile.num_parameters = count_parameters(model)
    profile.model_size_mb = model_size_mb(model)
    profile.vocab_size = getattr(config, "vocab_size", 0)
    profile.num_layers = getattr(
        config,
        "num_hidden_layers",
        getattr(config, "n_layer", 0),
    )
    profile.hidden_size = getattr(
        config,
        "hidden_size",
        getattr(config, "n_embd", 0),
    )
    profile.num_attention_heads = getattr(
        config,
        "num_attention_heads",
        getattr(config, "n_head", 0),
    )
    profile.num_kv_heads = getattr(
        config,
        "num_key_value_heads",
        profile.num_attention_heads,
    )
    profile.max_position = getattr(
        config,
        "max_position_embeddings",
        getattr(config, "n_positions", 0),
    )
    profile.architecture_type = type(model).__name__
    return profile


# ─────────────────────────────────────────────────────────────────────────────
# Benchmark: Tokenizer Fertility
# ─────────────────────────────────────────────────────────────────────────────

def benchmark_fertility(tokenizer, model_name: str) -> list[FertilityResult]:
    """Measure tokenization fertility across Indic languages.

    Fertility = number of tokens / number of words.
    Lower is better (more efficient tokenization).
    """
    results = []
    for lang, sentence in INDIC_SENTENCES.items():
        tokens = tokenizer.encode(sentence, add_special_tokens=False)
        words = sentence.split()
        num_words = max(len(words), 1)
        fertility = len(tokens) / num_words

        results.append(FertilityResult(
            model_name=model_name,
            language=lang,
            num_chars=len(sentence),
            num_tokens=len(tokens),
            fertility=round(fertility, 2),
        ))
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Benchmark: Inference Latency
# ─────────────────────────────────────────────────────────────────────────────

def benchmark_latency(
    model,
    tokenizer,
    model_name: str,
    device: str,
    num_gen_tokens: int = 50,
    warmup_runs: int = 2,
    bench_runs: int = 5,
) -> LatencyResult:
    """Measure prefill and decode latency."""
    prompt = "India is a diverse country with many languages and cultures. The future of"
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to(model.device)
    attention_mask = inputs.get("attention_mask", None)
    if attention_mask is not None:
        attention_mask = attention_mask.to(model.device)

    prompt_len = input_ids.shape[1]

    # Warmup
    for _ in range(warmup_runs):
        with torch.no_grad():
            model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=10,
                do_sample=False,
            )

    # Benchmark
    timings = []
    for _ in range(bench_runs):
        if "cuda" in str(model.device):
            torch.cuda.synchronize()
        with timer() as elapsed:
            with torch.no_grad():
                out = model.generate(
                    input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=num_gen_tokens,
                    do_sample=False,
                )
        if "cuda" in str(model.device):
            torch.cuda.synchronize()
        gen_len = out.shape[1] - prompt_len
        total = elapsed()
        timings.append((total, gen_len))

    avg_total = np.mean([t[0] for t in timings])
    avg_gen = np.mean([t[1] for t in timings])
    decode_per_token = avg_total / max(avg_gen, 1)
    tps = (avg_gen / avg_total) * 1000 if avg_total > 0 else 0

    return LatencyResult(
        model_name=model_name,
        prompt_tokens=prompt_len,
        gen_tokens=int(avg_gen),
        total_ms=round(avg_total, 1),
        decode_ms_per_token=round(decode_per_token, 1),
        tokens_per_sec=round(tps, 1),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Benchmark: Perplexity
# ─────────────────────────────────────────────────────────────────────────────

def benchmark_perplexity(
    model,
    tokenizer,
    model_name: str,
    text: str,
    language: str,
    max_length: int = 512,
) -> PerplexityResult:
    """Compute perplexity on a text sample."""
    encodings = tokenizer(
        text, return_tensors="pt", truncation=True, max_length=max_length
    )
    input_ids = encodings["input_ids"].to(model.device)

    if input_ids.shape[1] < 2:
        return PerplexityResult(
            model_name=model_name,
            language=language,
            perplexity=float("inf"),
            loss=float("inf"),
        )

    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
        loss = outputs.loss.item()

    ppl = min(np.exp(loss), 1e6)  # cap at 1M to avoid overflow display
    return PerplexityResult(
        model_name=model_name,
        language=language,
        perplexity=round(ppl, 2),
        loss=round(loss, 4),
    )


def get_perplexity_texts(quick: bool = False) -> dict:
    """Get evaluation texts for perplexity measurement."""
    texts = {}

    # English text
    texts["English"] = (
        "India is the world's largest democracy and the seventh-largest country by area. "
        "It is a land of incredible diversity, with hundreds of languages spoken across "
        "its states and territories. The Indian economy has grown rapidly in recent decades, "
        "becoming one of the fastest-growing major economies in the world. Technology and "
        "innovation have become central to India's development strategy, with cities like "
        "Bangalore, Hyderabad, and Pune emerging as global technology hubs. The country's "
        "rich cultural heritage spans thousands of years, from the ancient Indus Valley "
        "civilization to the vibrant arts and traditions that continue to thrive today."
    )

    # Hindi text
    texts["Hindi"] = (
        "भारत विश्व का सबसे बड़ा लोकतंत्र है और क्षेत्रफल के हिसाब से सातवां सबसे बड़ा देश है। "
        "यह अविश्वसनीय विविधता की भूमि है, जहाँ इसके राज्यों और केंद्र शासित प्रदेशों में "
        "सैकड़ों भाषाएँ बोली जाती हैं। भारतीय अर्थव्यवस्था हाल के दशकों में तेजी से बढ़ी है, "
        "जो दुनिया की सबसे तेजी से बढ़ती प्रमुख अर्थव्यवस्थाओं में से एक बन गई है। प्रौद्योगिकी "
        "और नवाचार भारत की विकास रणनीति के केंद्र में आ गए हैं। देश की समृद्ध सांस्कृतिक विरासत "
        "हजारों वर्षों तक फैली हुई है।"
    )

    # Tamil text
    texts["Tamil"] = (
        "இந்தியா உலகின் மிகப்பெரிய ஜனநாயக நாடு ஆகும். இது பல்வேறு மொழிகள் "
        "மற்றும் கலாச்சாரங்களைக் கொண்ட நாடு. இந்திய பொருளாதாரம் சமீப ஆண்டுகளில் "
        "வேகமாக வளர்ந்து வருகிறது. தொழில்நுட்பம் மற்றும் புதுமை இந்தியாவின் வளர்ச்சி "
        "உத்தியின் மையமாக உள்ளது."
    )

    return texts


# ─────────────────────────────────────────────────────────────────────────────
# Benchmark: Text Generation Quality
# ─────────────────────────────────────────────────────────────────────────────

def benchmark_generation(
    model,
    tokenizer,
    model_name: str,
    max_new_tokens: int = 80,
) -> dict:
    """Generate text samples for qualitative comparison."""
    generations = {}
    for lang, prompt in GENERATION_PROMPTS.items():
        inputs = tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(model.device)

        with torch.no_grad():
            output = model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                top_k=50,
                repetition_penalty=1.1,
            )

        full_text = tokenizer.decode(output[0], skip_special_tokens=True)
        generated = full_text[len(prompt):].strip()
        generations[lang] = {
            "prompt": prompt,
            "generated": generated[:500],  # cap display length
        }
    return generations


# ─────────────────────────────────────────────────────────────────────────────
# Display & Reporting
# ─────────────────────────────────────────────────────────────────────────────

def print_section(title: str):
    width = 72
    print(f"\n{'='*width}")
    print(f"  {title}")
    print(f"{'='*width}\n")


def display_profiles(profiles: list[ModelProfile]):
    print_section("1. MODEL ARCHITECTURE PROFILES")
    rows = []
    for p in profiles:
        rows.append([
            p.name,
            f"{p.num_parameters:,}",
            p.num_layers,
            p.hidden_size,
            f"{p.num_attention_heads}Q / {p.num_kv_heads}KV",
            f"{p.vocab_size:,}",
            f"{p.max_position:,}",
            f"{p.model_size_mb:.1f} MB",
            p.architecture_type,
        ])
    headers = [
        "Model", "Parameters", "Layers", "Hidden", "Attn Heads",
        "Vocab", "Max Pos", "Size", "Arch Class",
    ]
    print(tabulate(rows, headers=headers, tablefmt="fancy_grid"))


def display_fertility(results: list[FertilityResult], model_names: list[str]):
    print_section("2. TOKENIZER FERTILITY (tokens per word — lower is better)")
    languages = list(INDIC_SENTENCES.keys())

    # Build table
    rows = []
    for lang in languages:
        row = [lang]
        lang_results = [r for r in results if r.language == lang]
        best = min((r.fertility for r in lang_results), default=999)
        for name in model_names:
            r = next((r for r in lang_results if r.model_name == name), None)
            if r:
                marker = " ★" if r.fertility == best else ""
                row.append(f"{r.fertility:.2f}{marker}")
            else:
                row.append("—")
        rows.append(row)

    # Averages
    avg_row = ["AVERAGE"]
    for name in model_names:
        model_results = [r for r in results if r.model_name == name]
        if model_results:
            avg = np.mean([r.fertility for r in model_results])
            avg_row.append(f"{avg:.2f}")
        else:
            avg_row.append("—")
    rows.append(avg_row)

    headers = ["Language"] + model_names
    print(tabulate(rows, headers=headers, tablefmt="fancy_grid"))
    print("  ★ = best in class for that language\n")


def display_latency(results: list[LatencyResult]):
    print_section("3. INFERENCE LATENCY")
    rows = []
    for r in results:
        rows.append([
            r.model_name,
            r.prompt_tokens,
            r.gen_tokens,
            f"{r.total_ms:.0f} ms",
            f"{r.decode_ms_per_token:.1f} ms",
            f"{r.tokens_per_sec:.1f}",
        ])
    headers = [
        "Model", "Prompt Tokens", "Gen Tokens",
        "Total Time", "ms/token", "Tokens/sec",
    ]
    print(tabulate(rows, headers=headers, tablefmt="fancy_grid"))


def display_perplexity(results: list[PerplexityResult], model_names: list[str]):
    print_section("4. PERPLEXITY (lower is better)")
    languages = sorted(set(r.language for r in results))

    rows = []
    for lang in languages:
        row = [lang]
        lang_results = [r for r in results if r.language == lang]
        for name in model_names:
            r = next((r for r in lang_results if r.model_name == name), None)
            if r and r.perplexity < 1e6:
                row.append(f"{r.perplexity:.2f}")
            else:
                row.append("—")
        rows.append(row)

    headers = ["Language"] + model_names
    print(tabulate(rows, headers=headers, tablefmt="fancy_grid"))


def display_generations(all_generations: dict):
    print_section("5. TEXT GENERATION SAMPLES")
    for model_name, gens in all_generations.items():
        print(f"\n  ── {model_name} ──")
        for lang, data in gens.items():
            print(f"\n  [{lang}]")
            print(f"  Prompt: {data['prompt']}")
            gen_text = data["generated"][:300]
            print(f"  Output: {gen_text}")
        print()


def display_summary(results: BenchmarkResults, model_names: list[str]):
    print_section("SUMMARY SCORECARD")

    rows = []
    for name in model_names:
        profile = next((p for p in results.profiles if p.name == name), None)
        fert_results = [r for r in results.fertility if r.model_name == name]
        lat_result = next(
            (r for r in results.latency if r.model_name == name), None
        )
        ppl_results = [
            r for r in results.perplexity if r.model_name == name
        ]

        avg_fert = np.mean([r.fertility for r in fert_results]) if fert_results else None
        avg_ppl = np.mean(
            [r.perplexity for r in ppl_results if r.perplexity < 1e6]
        ) if ppl_results else None

        rows.append([
            name,
            f"{profile.num_parameters:,}" if profile else "—",
            f"{profile.model_size_mb:.0f} MB" if profile else "—",
            f"{avg_fert:.2f}" if avg_fert else "—",
            f"{lat_result.tokens_per_sec:.1f} tok/s" if lat_result else "—",
            f"{avg_ppl:.1f}" if avg_ppl else "—",
        ])

    headers = [
        "Model", "Params", "Size", "Avg Fertility",
        "Speed", "Avg Perplexity",
    ]
    print(tabulate(rows, headers=headers, tablefmt="fancy_grid"))

    print("\n  Key insights:")
    print("  • Lower fertility = more efficient Indic tokenization")
    print("  • Higher tokens/sec = faster inference")
    print("  • Lower perplexity = better language modeling")
    print()


# ─────────────────────────────────────────────────────────────────────────────
# Main orchestration
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Benchmark Nandi-Mini-150M against Indian & small LLMs"
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["nandi", "smollm2"],
        choices=list(MODEL_REGISTRY.keys()),
        help="Models to benchmark (default: nandi smollm2)",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Device to run on: cpu, cuda, cuda:0, etc. (default: cpu)",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick mode: fewer samples, skip heavy benchmarks",
    )
    parser.add_argument(
        "--skip-perplexity",
        action="store_true",
        help="Skip perplexity benchmark (saves time on CPU)",
    )
    parser.add_argument(
        "--skip-generation",
        action="store_true",
        help="Skip text generation samples",
    )
    parser.add_argument(
        "--gen-tokens",
        type=int,
        default=50,
        help="Number of tokens to generate in latency test (default: 50)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Save results to JSON file",
    )
    args = parser.parse_args()

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("⚠ CUDA not available, falling back to CPU")
        device = "cpu"

    dtype = torch.float32
    if "cuda" in device:
        dtype = torch.float16
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    print(f"\nDevice: {device} | Dtype: {dtype}")
    print(f"Models: {', '.join(args.models)}")
    if args.quick:
        print("Mode: QUICK (reduced samples)\n")
    print()

    # Storage
    results = BenchmarkResults()
    loaded_models = {}
    model_names = []

    # ── Load all models ──────────────────────────────────────────────────
    for model_key in args.models:
        meta = MODEL_REGISTRY[model_key]
        print(f"{'─'*60}")
        print(f"Loading {meta['name']} ({meta['hf_id']})")
        print(f"{'─'*60}")

        try:
            model, tokenizer = safe_load_model(
                meta["hf_id"], meta["trust_remote_code"], device, dtype
            )
            loaded_models[model_key] = (model, tokenizer, meta)
            model_names.append(meta["name"])

            # Profile
            profile = extract_model_profile(model, tokenizer, meta)
            results.profiles.append(profile)
            print(f"  ✓ Loaded: {profile.num_parameters:,} params, "
                  f"{profile.model_size_mb:.1f} MB\n")

        except Exception as e:
            print(f"  ✗ Failed to load {meta['name']}: {e}")
            print(f"    Skipping this model.\n")
            continue

    if not loaded_models:
        print("No models loaded successfully. Exiting.")
        sys.exit(1)

    # ── 1. Display profiles ──────────────────────────────────────────────
    display_profiles(results.profiles)

    # ── 2. Fertility benchmark ───────────────────────────────────────────
    print_section("Running: TOKENIZER FERTILITY")
    for model_key, (model, tokenizer, meta) in loaded_models.items():
        print(f"  Testing {meta['name']}...")
        fert = benchmark_fertility(tokenizer, meta["name"])
        results.fertility.extend(fert)
    display_fertility(results.fertility, model_names)

    # ── 3. Latency benchmark ────────────────────────────────────────────
    print_section("Running: INFERENCE LATENCY")
    bench_runs = 3 if args.quick else 5
    for model_key, (model, tokenizer, meta) in loaded_models.items():
        print(f"  Benchmarking {meta['name']}...")
        lat = benchmark_latency(
            model, tokenizer, meta["name"], device,
            num_gen_tokens=args.gen_tokens,
            warmup_runs=1 if args.quick else 2,
            bench_runs=bench_runs,
        )
        results.latency.append(lat)
    display_latency(results.latency)

    # ── 4. Perplexity benchmark ─────────────────────────────────────────
    if not args.skip_perplexity:
        print_section("Running: PERPLEXITY")
        ppl_texts = get_perplexity_texts(args.quick)
        for model_key, (model, tokenizer, meta) in loaded_models.items():
            for lang, text in ppl_texts.items():
                print(f"  {meta['name']} / {lang}...")
                ppl = benchmark_perplexity(model, tokenizer, meta["name"], text, lang)
                results.perplexity.append(ppl)
        display_perplexity(results.perplexity, model_names)

    # ── 5. Generation samples ───────────────────────────────────────────
    if not args.skip_generation:
        print_section("Running: TEXT GENERATION")
        for model_key, (model, tokenizer, meta) in loaded_models.items():
            print(f"  Generating with {meta['name']}...")
            gen_tokens = 40 if args.quick else 80
            gens = benchmark_generation(model, tokenizer, meta["name"], gen_tokens)
            results.generations[meta["name"]] = gens
        display_generations(results.generations)

    # ── Summary ─────────────────────────────────────────────────────────
    display_summary(results, model_names)

    # ── Save results ────────────────────────────────────────────────────
    if args.output:
        output_data = {
            "profiles": [asdict(p) for p in results.profiles],
            "fertility": [asdict(f) for f in results.fertility],
            "latency": [asdict(l) for l in results.latency],
            "perplexity": [asdict(p) for p in results.perplexity],
            "generations": results.generations,
        }
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        print(f"  Results saved to: {args.output}")

    # Cleanup
    for model_key, (model, tokenizer, meta) in loaded_models.items():
        del model, tokenizer
    if "cuda" in device:
        torch.cuda.empty_cache()

    print("\n✓ Benchmark complete!\n")


if __name__ == "__main__":
    main()