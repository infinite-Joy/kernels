#!/usr/bin/env python3
"""
=============================================================================
  IndicBench-Micro — India-Specific Evaluation for Small LLMs
=============================================================================

Evaluates models on 100 India-specific test cases across 10 categories:
  1. Code-Mixing (Hinglish, Tanglish, Benglish, etc.)
  2. Cultural Knowledge (festivals, food, social systems)
  3. Indic NLU (sentiment, intent, sarcasm, formality)
  4. Translation (en↔Indic, cross-Indic)
  5. Number Systems (lakh/crore, Devanagari numerals)
  6. Agriculture (crop advisory, pest, mandi, schemes)
  7. Legal & Governance (FIR, RTI, Panchayat, land)
  8. Healthcare (Ayurveda, maternal, rural access)
  9. Script Generation (coherent text in 10 languages)
  10. Robustness (transliteration, mixed-script, abbreviations)

Metrics per task type:
  - perplexity tasks  → language modeling loss (lower = better understanding)
  - next_token_quality → perplexity on prompt + generation coherence score
  - tokenizer fertility → tokens/word across all test texts

Requirements:
  pip install torch transformers accelerate sentencepiece protobuf \
              tqdm tabulate numpy --break-system-packages

Usage:
  python eval_indicbench.py                           # Nandi vs SmolLM2
  python eval_indicbench.py --models nandi sarvam     # Nandi vs Sarvam
  python eval_indicbench.py --quick                   # Fewer samples
  python eval_indicbench.py --device cuda              # GPU mode
  python eval_indicbench.py --output results.json      # Save results
  python eval_indicbench.py --categories code_mixing agriculture  # Specific cats
"""

import argparse
import json
import math
import os
import sys
import time
import warnings
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from tqdm import tqdm
from tabulate import tabulate

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# Model Registry
# ─────────────────────────────────────────────────────────────────────────────

MODEL_REGISTRY = {
    "nandi": {
        "name": "Nandi-Mini-150M",
        "hf_id": "FrontiersMind/Nandi-Mini-150M",
        "trust_remote_code": True,
        "origin": "India",
        "params": "150M",
    },
    "nandi_instruct": {
        "name": "Nandi-Instruct-150M",
        "hf_id": "FrontiersMind/Nandi-Mini-150M-Instruct",
        "trust_remote_code": True,
        "origin": "India",
        "params": "150M",
    },
    "smollm2": {
        "name": "SmolLM2-135M",
        "hf_id": "HuggingFaceTB/SmolLM2-135M",
        "trust_remote_code": False,
        "origin": "Global",
        "params": "135M",
    },
    "smollm": {
        "name": "SmolLM-135M",
        "hf_id": "HuggingFaceTB/SmolLM-135M",
        "trust_remote_code": False,
        "origin": "Global",
        "params": "135M",
    },
    "sarvam": {
        "name": "Sarvam-1",
        "hf_id": "sarvamai/sarvam-1",
        "trust_remote_code": False,
        "origin": "India",
        "params": "2B",
    },
    "navarasa": {
        "name": "Navarasa-2.0",
        "hf_id": "Telugu-LLM-Labs/Indic-gemma-2b-finetuned-sft-Navarasa-2.0",
        "trust_remote_code": False,
        "origin": "India",
        "params": "2B",
    },
}

# ─────────────────────────────────────────────────────────────────────────────
# Evaluation Metrics
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TaskResult:
    task_id: str
    model_name: str
    category: str
    subcategory: str
    language: str
    task_type: str
    perplexity: float = 0.0
    loss: float = 0.0
    num_tokens: int = 0
    fertility: float = 0.0
    generation: str = ""
    script_consistency: float = 0.0
    generation_length: int = 0


@dataclass
class CategoryScore:
    category: str
    model_name: str
    avg_perplexity: float = 0.0
    avg_fertility: float = 0.0
    num_tasks: int = 0
    avg_script_consistency: float = 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Script Detection Utilities
# ─────────────────────────────────────────────────────────────────────────────

SCRIPT_RANGES = {
    "Devanagari": (0x0900, 0x097F),
    "Bengali":    (0x0980, 0x09FF),
    "Gurmukhi":   (0x0A00, 0x0A7F),
    "Gujarati":   (0x0A80, 0x0AFF),
    "Oriya":      (0x0B00, 0x0B7F),
    "Tamil":      (0x0B80, 0x0BFF),
    "Telugu":     (0x0C00, 0x0C7F),
    "Kannada":    (0x0C80, 0x0CFF),
    "Malayalam":  (0x0D00, 0x0D7F),
    "Latin":      (0x0041, 0x024F),
}

LANG_TO_SCRIPT = {
    "hi": "Devanagari", "mr": "Devanagari", "bn": "Bengali",
    "as": "Bengali", "ta": "Tamil", "te": "Telugu", "gu": "Gujarati",
    "kn": "Kannada", "ml": "Malayalam", "pa": "Gurmukhi",
    "or": "Oriya", "en": "Latin",
}


def detect_scripts(text: str) -> dict:
    """Count characters belonging to each script."""
    counts = defaultdict(int)
    for ch in text:
        cp = ord(ch)
        for script, (lo, hi) in SCRIPT_RANGES.items():
            if lo <= cp <= hi:
                counts[script] += 1
                break
    return dict(counts)


def script_consistency_score(text: str, expected_lang: str) -> float:
    """Measure how much of the generated text stays in the expected script.

    Returns 0.0–1.0 where 1.0 = all script chars match expected.
    For code-mixed languages (e.g. 'hi-en'), we accept both scripts.
    """
    if not text.strip():
        return 0.0

    accepted_scripts = set()
    lang_parts = expected_lang.replace("-roman", "").replace("-casual", "").replace("-mixed", "").split("-")
    for lp in lang_parts:
        lp = lp.strip()
        if lp in LANG_TO_SCRIPT:
            accepted_scripts.add(LANG_TO_SCRIPT[lp])

    # If no known script mapping, skip scoring
    if not accepted_scripts:
        return 1.0

    # For code-mixed, always accept Latin too
    if len(lang_parts) > 1 or "roman" in expected_lang:
        accepted_scripts.add("Latin")

    counts = detect_scripts(text)
    total_script_chars = sum(counts.values())
    if total_script_chars == 0:
        return 0.5  # neutral — all punctuation/numbers/spaces

    matched = sum(v for k, v in counts.items() if k in accepted_scripts)
    return matched / total_script_chars


# ─────────────────────────────────────────────────────────────────────────────
# Core Evaluation Functions
# ─────────────────────────────────────────────────────────────────────────────

def compute_perplexity(model, tokenizer, text: str, max_len: int = 512) -> tuple:
    """Compute perplexity on given text. Returns (ppl, loss, n_tokens)."""
    encodings = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_len)
    input_ids = encodings["input_ids"].to(model.device)
    n_tokens = input_ids.shape[1]

    if n_tokens < 2:
        return (float("inf"), float("inf"), n_tokens)

    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
        loss = outputs.loss.item()

    ppl = min(math.exp(loss), 1e7)
    return (ppl, loss, n_tokens)


def generate_text(model, tokenizer, prompt: str, max_new: int = 60) -> str:
    """Generate text continuation."""
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to(model.device)

    with torch.no_grad():
        output = model.generate(
            input_ids,
            max_new_tokens=max_new,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            top_k=50,
            repetition_penalty=1.15,
            pad_token_id=tokenizer.eos_token_id,
        )

    full = tokenizer.decode(output[0], skip_special_tokens=True)
    return full[len(prompt):].strip()


def compute_fertility(tokenizer, text: str) -> tuple:
    """Compute tokenizer fertility: tokens/word. Returns (fertility, n_tokens)."""
    tokens = tokenizer.encode(text, add_special_tokens=False)
    words = text.split()
    n_words = max(len(words), 1)
    return (len(tokens) / n_words, len(tokens))


def repetition_ratio(text: str, ngram: int = 3) -> float:
    """Fraction of repeated n-grams. High = degenerate output."""
    words = text.split()
    if len(words) < ngram + 1:
        return 0.0
    ngrams = [tuple(words[i:i+ngram]) for i in range(len(words) - ngram + 1)]
    if not ngrams:
        return 0.0
    unique = set(ngrams)
    return 1.0 - (len(unique) / len(ngrams))


# ─────────────────────────────────────────────────────────────────────────────
# Task Evaluation Router
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_task(task: dict, model, tokenizer, model_name: str, gen_tokens: int = 60) -> TaskResult:
    """Evaluate a single task and return structured result."""
    result = TaskResult(
        task_id=task["id"],
        model_name=model_name,
        category=task["category"],
        subcategory=task.get("subcategory", ""),
        language=task.get("language", ""),
        task_type=task["task"],
    )

    if task["task"] == "perplexity":
        text = task["text"]
        ppl, loss, n_tok = compute_perplexity(model, tokenizer, text)
        fert, _ = compute_fertility(tokenizer, text)
        result.perplexity = round(ppl, 2)
        result.loss = round(loss, 4)
        result.num_tokens = n_tok
        result.fertility = round(fert, 2)

    elif task["task"] == "next_token_quality":
        prompt = task["prompt"]

        # Perplexity on the prompt itself (how well model understands it)
        ppl, loss, n_tok = compute_perplexity(model, tokenizer, prompt)
        result.perplexity = round(ppl, 2)
        result.loss = round(loss, 4)
        result.num_tokens = n_tok

        # Fertility on prompt
        fert, _ = compute_fertility(tokenizer, prompt)
        result.fertility = round(fert, 2)

        # Generate continuation
        gen = generate_text(model, tokenizer, prompt, max_new=gen_tokens)
        result.generation = gen[:500]
        result.generation_length = len(gen.split())

        # Script consistency
        lang = task.get("language", "")
        result.script_consistency = round(script_consistency_score(gen, lang), 3)

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Aggregate Scoring
# ─────────────────────────────────────────────────────────────────────────────

def aggregate_results(results: list[TaskResult]) -> dict:
    """Aggregate results by category and model."""
    scores = defaultdict(lambda: defaultdict(list))

    for r in results:
        key = (r.category, r.model_name)
        scores[key]["ppl"].append(r.perplexity)
        scores[key]["loss"].append(r.loss)
        scores[key]["fertility"].append(r.fertility)
        if r.script_consistency > 0:
            scores[key]["script_con"].append(r.script_consistency)
        if r.generation_length > 0:
            scores[key]["gen_len"].append(r.generation_length)

    aggregated = {}
    for (cat, model), metrics in scores.items():
        valid_ppl = [p for p in metrics["ppl"] if p < 1e6]
        aggregated[(cat, model)] = CategoryScore(
            category=cat,
            model_name=model,
            avg_perplexity=round(np.mean(valid_ppl), 2) if valid_ppl else float("inf"),
            avg_fertility=round(np.mean(metrics["fertility"]), 2) if metrics["fertility"] else 0,
            num_tasks=len(metrics["ppl"]),
            avg_script_consistency=round(np.mean(metrics["script_con"]), 3) if metrics["script_con"] else 0,
        )
    return aggregated


# ─────────────────────────────────────────────────────────────────────────────
# Display Functions
# ─────────────────────────────────────────────────────────────────────────────

def print_header(title: str):
    w = 78
    print(f"\n{'━'*w}")
    print(f"  {title}")
    print(f"{'━'*w}\n")


def display_category_comparison(aggregated: dict, model_names: list[str], categories: list[str]):
    """Show per-category comparison table."""
    print_header("CATEGORY-WISE COMPARISON")

    for metric_name, metric_key, lower_better in [
        ("Avg Perplexity (↓ better)", "avg_perplexity", True),
        ("Avg Fertility (↓ better)", "avg_fertility", True),
        ("Avg Script Consistency (↑ better)", "avg_script_consistency", False),
    ]:
        print(f"\n  ▸ {metric_name}\n")
        rows = []
        for cat in categories:
            row = [cat.replace("_", " ").title()]
            values = []
            for name in model_names:
                score = aggregated.get((cat, name))
                val = getattr(score, metric_key, None) if score else None
                values.append(val)

            # Find best
            valid_vals = [(i, v) for i, v in enumerate(values) if v and v < 1e6 and v > 0]
            best_idx = None
            if valid_vals:
                if lower_better:
                    best_idx = min(valid_vals, key=lambda x: x[1])[0]
                else:
                    best_idx = max(valid_vals, key=lambda x: x[1])[0]

            for i, val in enumerate(values):
                if val is None or val >= 1e6 or val == 0:
                    row.append("—")
                else:
                    marker = " ★" if i == best_idx else ""
                    if metric_key == "avg_script_consistency":
                        row.append(f"{val:.3f}{marker}")
                    else:
                        row.append(f"{val:.2f}{marker}")
            rows.append(row)

        headers = ["Category"] + model_names
        print(tabulate(rows, headers=headers, tablefmt="rounded_grid"))
        print()

    print("  ★ = best in category\n")


def display_language_breakdown(results: list[TaskResult], model_names: list[str]):
    """Show per-language perplexity comparison."""
    print_header("LANGUAGE-WISE PERPLEXITY (↓ better)")

    # Collect per-language stats
    lang_stats = defaultdict(lambda: defaultdict(list))
    for r in results:
        base_lang = r.language.split("-")[0] if r.language else "unknown"
        if r.perplexity < 1e6:
            lang_stats[base_lang][r.model_name].append(r.perplexity)

    rows = []
    for lang in sorted(lang_stats.keys()):
        row = [lang]
        values = []
        for name in model_names:
            ppls = lang_stats[lang].get(name, [])
            avg = np.mean(ppls) if ppls else None
            values.append(avg)

        valid = [(i, v) for i, v in enumerate(values) if v is not None]
        best_idx = min(valid, key=lambda x: x[1])[0] if valid else None

        for i, v in enumerate(values):
            if v is None:
                row.append("—")
            else:
                marker = " ★" if i == best_idx else ""
                row.append(f"{v:.1f}{marker}")
        rows.append(row)

    headers = ["Language"] + model_names
    print(tabulate(rows, headers=headers, tablefmt="rounded_grid"))
    print()


def display_generation_samples(results: list[TaskResult], model_names: list[str], max_samples: int = 12):
    """Show selected generation samples for qualitative comparison."""
    print_header("GENERATION SAMPLES (qualitative)")

    gen_results = [r for r in results if r.task_type == "next_token_quality" and r.generation]

    # Group by task_id
    by_task = defaultdict(dict)
    for r in gen_results:
        by_task[r.task_id][r.model_name] = r

    shown = 0
    for task_id in sorted(by_task.keys()):
        if shown >= max_samples:
            break
        task_results = by_task[task_id]
        if not task_results:
            continue

        first = next(iter(task_results.values()))
        print(f"  ┌─ [{task_id}] {first.category}/{first.subcategory} ({first.language})")

        # Find prompt from any result
        for name in model_names:
            r = task_results.get(name)
            if r:
                gen_preview = r.generation[:200].replace("\n", " ")
                sc = f"  script:{r.script_consistency:.2f}" if r.script_consistency else ""
                print(f"  │  {name:25s} → {gen_preview}")
                print(f"  │  {'':25s}   [ppl={r.perplexity:.1f}  fert={r.fertility:.1f}{sc}]")

        print(f"  └{'─'*76}")
        print()
        shown += 1


def display_overall_scorecard(aggregated: dict, model_names: list[str], categories: list[str]):
    """Final summary scorecard."""
    print_header("OVERALL SCORECARD")

    rows = []
    for name in model_names:
        all_ppl = []
        all_fert = []
        all_sc = []
        for cat in categories:
            score = aggregated.get((cat, name))
            if score:
                if score.avg_perplexity < 1e6:
                    all_ppl.append(score.avg_perplexity)
                if score.avg_fertility > 0:
                    all_fert.append(score.avg_fertility)
                if score.avg_script_consistency > 0:
                    all_sc.append(score.avg_script_consistency)

        # Count category wins
        wins = 0
        for cat in categories:
            cat_scores = {
                m: aggregated.get((cat, m))
                for m in model_names
                if aggregated.get((cat, m)) and aggregated[(cat, m)].avg_perplexity < 1e6
            }
            if cat_scores:
                best = min(cat_scores.values(), key=lambda s: s.avg_perplexity)
                if best.model_name == name:
                    wins += 1

        rows.append([
            name,
            f"{np.mean(all_ppl):.1f}" if all_ppl else "—",
            f"{np.mean(all_fert):.2f}" if all_fert else "—",
            f"{np.mean(all_sc):.3f}" if all_sc else "—",
            f"{wins}/{len(categories)}",
        ])

    headers = [
        "Model", "Avg PPL ↓", "Avg Fertility ↓",
        "Avg Script Consistency ↑", "Category Wins",
    ]
    print(tabulate(rows, headers=headers, tablefmt="rounded_grid"))

    print("\n  Interpretation:")
    print("  • Perplexity: How well the model understands India-specific text (lower = better)")
    print("  • Fertility:  Tokenization efficiency for Indic scripts (lower = cheaper inference)")
    print("  • Script Consistency: Does generation stay in the correct script? (higher = better)")
    print("  • Category Wins: # of categories where model had lowest avg perplexity")
    print()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="IndicBench-Micro: India-specific LLM evaluation")
    parser.add_argument("--models", nargs="+", default=["nandi", "smollm2"],
                        choices=list(MODEL_REGISTRY.keys()))
    parser.add_argument("--dataset", default="indicbench_micro.json",
                        help="Path to evaluation dataset JSON")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--quick", action="store_true",
                        help="Run subset (3 tasks per category)")
    parser.add_argument("--categories", nargs="+", default=None,
                        help="Run only specific categories")
    parser.add_argument("--gen-tokens", type=int, default=60,
                        help="Max tokens to generate per prompt")
    parser.add_argument("--show-samples", type=int, default=12,
                        help="Number of generation samples to display")
    parser.add_argument("--output", type=str, default=None,
                        help="Save detailed results to JSON")
    args = parser.parse_args()

    # Device setup
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("⚠ CUDA unavailable, using CPU")
        device = "cpu"
    dtype = torch.float16 if "cuda" in device else torch.float32

    # ── Load dataset ─────────────────────────────────────────────────────
    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        # Try looking in the same directory as the script
        alt_path = Path(__file__).parent / args.dataset
        if alt_path.exists():
            dataset_path = alt_path
        else:
            print(f"✗ Dataset not found: {args.dataset}")
            print(f"  Download indicbench_micro.json or specify --dataset path")
            sys.exit(1)

    with open(dataset_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    tasks = dataset["tasks"]
    all_categories = sorted(set(t["category"] for t in tasks))

    # Filter categories
    if args.categories:
        tasks = [t for t in tasks if t["category"] in args.categories]
        all_categories = sorted(set(t["category"] for t in tasks))

    # Quick mode: 3 per category
    if args.quick:
        cat_counts = defaultdict(int)
        filtered = []
        for t in tasks:
            if cat_counts[t["category"]] < 3:
                filtered.append(t)
                cat_counts[t["category"]] += 1
        tasks = filtered

    print(f"\n{'='*78}")
    print(f"  IndicBench-Micro — India-Specific LLM Evaluation")
    print(f"{'='*78}")
    print(f"  Device: {device} | Tasks: {len(tasks)} | Categories: {len(all_categories)}")
    print(f"  Models: {', '.join(args.models)}")
    if args.quick:
        print(f"  Mode: QUICK (3 tasks per category)")
    print()

    # ── Load models ──────────────────────────────────────────────────────
    from transformers import AutoModelForCausalLM, AutoTokenizer

    loaded = {}
    model_names = []
    for key in args.models:
        meta = MODEL_REGISTRY[key]
        print(f"  Loading {meta['name']} ({meta['hf_id']})...")
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                meta["hf_id"], trust_remote_code=meta["trust_remote_code"]
            )
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            model = AutoModelForCausalLM.from_pretrained(
                meta["hf_id"],
                trust_remote_code=meta["trust_remote_code"],
                torch_dtype=dtype,
                low_cpu_mem_usage=True,
            )
            if device == "cpu":
                model = model.to(device)
            elif device != "auto":
                model = model.to(device)
            model.eval()

            loaded[key] = (model, tokenizer, meta)
            model_names.append(meta["name"])
            print(f"  ✓ {meta['name']} loaded\n")
        except Exception as e:
            print(f"  ✗ Failed: {e}\n")

    if not loaded:
        print("No models loaded. Exiting.")
        sys.exit(1)

    # ── Run evaluation ───────────────────────────────────────────────────
    all_results = []
    total = len(tasks) * len(loaded)

    print_header("RUNNING EVALUATION")
    pbar = tqdm(total=total, desc="Evaluating", unit="task")

    for task in tasks:
        for key, (model, tokenizer, meta) in loaded.items():
            try:
                result = evaluate_task(
                    task, model, tokenizer, meta["name"],
                    gen_tokens=args.gen_tokens,
                )
                all_results.append(result)
            except Exception as e:
                tqdm.write(f"  ⚠ {task['id']}/{meta['name']}: {e}")
            pbar.update(1)

    pbar.close()

    # ── Aggregate & display ──────────────────────────────────────────────
    aggregated = aggregate_results(all_results)

    display_category_comparison(aggregated, model_names, all_categories)
    display_language_breakdown(all_results, model_names)
    display_generation_samples(all_results, model_names, max_samples=args.show_samples)
    display_overall_scorecard(aggregated, model_names, all_categories)

    # ── Save results ─────────────────────────────────────────────────────
    if args.output:
        output_data = {
            "meta": {
                "models": args.models,
                "num_tasks": len(tasks),
                "categories": all_categories,
                "device": device,
            },
            "task_results": [asdict(r) for r in all_results],
            "category_scores": {
                f"{k[0]}|{k[1]}": asdict(v) for k, v in aggregated.items()
            },
        }
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        print(f"  Results saved to: {args.output}\n")

    # Cleanup
    for key, (model, tokenizer, meta) in loaded.items():
        del model, tokenizer
    if "cuda" in device:
        torch.cuda.empty_cache()

    print("✓ Evaluation complete!\n")


if __name__ == "__main__":
    main()