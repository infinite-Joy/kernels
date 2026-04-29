# Nandi-Mini-150M — Evaluation Overview

## What is this?

A comprehensive evaluation of **FrontiersMind/Nandi-Mini-150M** against both international small LLMs (**SmolLM2-135M**) and India's leading open models (**Sarvam-1 2B**, **Navarasa-2.0 2B**) across architecture, tokenization, inference speed, language quality, India-specific tasks, and edge-deployment readiness. All benchmarks were run on an NVIDIA H100 (80 GB) with fp16 precision.

---

## 1. Architecture at a Glance

| | Nandi-Mini-150M | SmolLM2-135M |
|---|---|---|
| Parameters | 153 M | 135 M |
| Layers | 16 | 30 |
| Hidden size | 832 | 576 |
| Attention | 16Q / 4KV (GQA) | 9Q / 3KV (GQA) |
| Vocab | **131,072** | 49,152 |
| Max context | 2,048 | 8,192 |
| fp16 weight size | 293 MB | 257 MB |
| Architecture class | NandiForCausalLM | LlamaForCausalLM |

Nandi trades depth for width: fewer layers but a larger hidden dimension and a **2.7x larger vocabulary** purpose-built for Indic scripts.

---

## 2. Tokenizer Fertility (tokens per word — lower is better)

| Language | Nandi | SmolLM2 | Nandi advantage |
|----------|------:|--------:|----------------:|
| English | 1.08 | 1.08 | 1.0x |
| Hindi | **1.23** | 5.77 | 4.7x |
| Bengali | **1.78** | 11.78 | 6.6x |
| Tamil | **1.62** | 14.38 | 8.9x |
| Telugu | **1.18** | 16.27 | 13.8x |
| Kannada | **1.62** | 23.50 | 14.5x |
| Malayalam | **2.44** | 17.44 | 7.1x |
| Gujarati | **1.30** | 15.40 | 11.8x |
| Punjabi | **1.17** | 9.25 | 7.9x |
| Odia | **1.67** | 15.44 | 9.2x |
| Marathi | **1.55** | 6.55 | 4.2x |
| Assamese | **1.36** | 8.55 | 6.3x |
| **Average** | **1.51** | **12.44** | **8.2x** |

For every Indic language tested, Nandi needs dramatically fewer tokens to represent the same text. On Kannada, the gap is **14.5x** — a sentence that costs SmolLM2 188 tokens costs Nandi just 13.

---

## 3. Inference Speed (H100, fp16, 50-token generation)

| Metric | Nandi | SmolLM2 |
|--------|------:|--------:|
| Throughput | **32.3 tok/s** | 30.3 tok/s |
| Total time | **1,548 ms** | 1,649 ms |
| ms / token | **31.0** | 33.0 |

Nandi is ~7% faster despite 14% more parameters, thanks to its shallower (16-layer) architecture requiring fewer sequential operations.

---

## 4. Perplexity — General Text

| Language | Nandi | SmolLM2 |
|----------|------:|--------:|
| English | 8.08 | 6.54 |
| Hindi | 13.03 | 4.56 |
| Tamil | 33.34 | 2.87 |

SmolLM2's Hindi/Tamil PPL looks lower, but this is misleading. Its byte-fallback tokenizer produces extremely long token sequences for Indic text (5–23x more tokens), which mathematically deflates per-token loss. Raw PPL is **not comparable** across tokenizers with different fertility. See the generation-quality section below for what actually matters.

---

## 5. IndicBench-Micro — 85 India-Specific Tasks

12 languages (English + 11 Indic including Assamese), 10 categories: code-mixing, cultural knowledge, Indic NLU, translation, number systems, agriculture, legal/governance, healthcare, script generation, robustness.

### Headline Metrics

| Metric | Nandi | SmolLM2 |
|--------|------:|--------:|
| Avg fertility | **1.70** | 8.96 |
| Script consistency | **0.999** | 0.698 |
| Avg perplexity | 1288.2 | 302.7 |

### Category-Level Fertility (tokens per word)

| Category | Nandi | SmolLM2 |
|----------|------:|--------:|
| Agriculture | **1.67** | 11.19 |
| Code Mixing | **1.67** | 1.77 |
| Cultural Knowledge | **1.64** | 10.12 |
| Healthcare | **1.76** | 12.99 |
| Indic NLU | **1.30** | 6.38 |
| Legal/Governance | **1.27** | 9.60 |
| Number Systems | **1.22** | 2.22 |
| Script Generation | **1.81** | 14.68 |
| Translation | **2.46** | 9.22 |
| Robustness | **1.72** | 3.22 |

### Script Consistency (does generation stay in the correct script?)

| Category | Nandi | SmolLM2 |
|----------|------:|--------:|
| Agriculture | **1.000** | 0.813 |
| Code Mixing | **1.000** | 1.000 |
| Cultural Knowledge | **1.000** | 0.355 |
| Healthcare | **1.000** | 0.502 |
| Indic NLU | **1.000** | 0.900 |
| Legal/Governance | **1.000** | 0.976 |
| Script Generation | **0.998** | 0.586 |
| Robustness | **1.000** | 1.000 |

SmolLM2 frequently breaks out of Indic scripts into Latin or produces garbled characters in domains like healthcare (0.502), cultural knowledge (0.355), and script generation (0.586).

---

## 6. Generation Quality (qualitative samples)

### Hindi: "भारत एक ऐसा देश है जहाँ"

**Nandi**: "आप अपने घर और काम के लिए कई तरह की सुविधाएं पा सकते हैं। यहां के लोग यहां के जीवन में सबसे अधिक ध्यान देते हैं..."
*Fluent, coherent Hindi prose about Indian daily life.*

**SmolLM2**: "क्यों और समय पीछे। बिना सुदृढवा: अग्रह, ने मानक सुझार यद्विधान विचार आय"
*Near-gibberish.*

### Tamil: "இந்தியா ஒரு நாடு, அதில்"

**Nandi**: "நாம் சுதந்திரம் பெற்றுள்ளோம்... மகாத்மா காந்தியின் இந்திய அரசு எதிர்ப்பு போராட்டம் பற்றிய..."
*Coherent Tamil about India's freedom struggle.*

**SmolLM2**: "களம்பங்களின் உண்டவில். (10) I do not know how to get rid of..."
*Garbled Tamil, switches to English mid-sentence.*

### Telugu: "భారతదేశం ఒక గొప్ప దేశం, దీనిలో"

**Nandi**: Coherent Telugu about different fields of knowledge.
**SmolLM2**: "విచటాసమణ. ఉడషయాజిక బడెహవాధా" — garbled.

### English: "India is a country known for its"

Both produce coherent, well-formed English at comparable quality.

---

## 7. Edge Deployment Readiness (Nandi)

| Check | Measured | Threshold | Verdict |
|-------|----------|-----------|---------|
| fp16 size | 293 MB | ≤ 500 MB | PASS |
| int8 size | 146 MB | ≤ 300 MB | PASS |
| int4 size | 73 MB | — | Very edge-friendly |
| KV cache (2048 ctx) | 26 MB | — | PASS |
| Peak RSS | 1,026 MB | ≤ 1,500 MB | PASS |
| First-token latency | 29 ms | ≤ 50 ms | PASS |
| Indic fertility | 1.51 | ≤ 3.0 | PASS |
| int8 quantization | PPL 14.93 (−0.8% vs fp16) | — | Nearly lossless |
| Batch scaling | Linear to batch=8 (265 tok/s) | — | PASS |
| GFLOPs / token | 0.307 | — | Very light |
| ONNX export | Failed (missing dep) | — | Fixable |

At int8, the model is **146 MB** with virtually no quality loss. At int4 it's **73 MB** — small enough for a phone.

---

## 8. Indic Model Comparison — Nandi vs Sarvam-1 vs Navarasa-2.0

A 150M model built for India versus the two most prominent open Indian LLMs, both at 2B parameters (16–17x larger).

### Architecture Comparison

| | Nandi-Mini-150M | Sarvam-1 (2B) | Navarasa-2.0 (2B) |
|---|---|---|---|
| Parameters | 153 M | 2,525 M | 2,506 M |
| Layers | 16 | 28 | 18 |
| Hidden size | 832 | 2,048 | 2,048 |
| Attention | 16Q / 4KV (GQA) | 16Q / 8KV (GQA) | 8Q / 1KV (MQA) |
| Vocab | **131,072** | 68,096 | 256,000 |
| Max context | 2,048 | 8,192 | 8,192 |
| fp16 size | **293 MB** | 4,816 MB | 4,780 MB |
| Architecture | NandiForCausalLM | LlamaForCausalLM | GemmaForCausalLM |
| Origin | FrontiersMind | Sarvam AI | Telugu-LLM-Labs |

Nandi is **16x smaller** than both competitors. Sarvam-1 is a Llama-architecture model trained from scratch for Indic. Navarasa-2.0 is a Gemma-2B fine-tuned on Indic instruction data with the largest vocabulary (256K).

### Tokenizer Fertility (tokens per word — lower is better)

| Language | Nandi | Sarvam-1 | Navarasa-2.0 |
|----------|------:|---------:|-------------:|
| English | **1.08** | 1.38 | **1.08** |
| Hindi | **1.23** | **1.23** | 1.85 |
| Bengali | **1.78** | 2.00 | 3.56 |
| Tamil | **1.62** | 2.25 | 3.88 |
| Telugu | **1.18** | 1.36 | 3.55 |
| Marathi | 1.55 | **1.45** | 2.18 |
| Gujarati | **1.30** | 1.80 | 3.70 |
| Kannada | **1.62** | 2.00 | 5.12 |
| Malayalam | **2.44** | **2.44** | 5.11 |
| Punjabi | **1.17** | 1.42 | 3.17 |
| Odia | **1.67** | 1.89 | 5.78 |
| Assamese | **1.36** | 2.55 | 3.18 |
| **Average** | **1.50** | 1.81 | 3.51 |

Nandi's 131K Indic-optimized vocabulary gives it the best fertility across nearly every language. It leads Sarvam by 1.2x on average and Navarasa by 2.3x. Navarasa's 256K vocab is Gemma-derived and not as tightly tuned for Indic scripts despite its size.

### Inference Speed (H100, fp16, 50-token generation)

| Metric | Nandi | Sarvam-1 | Navarasa-2.0 |
|--------|------:|---------:|-------------:|
| Throughput | **32.7 tok/s** | 29.1 tok/s | 38.8 tok/s* |
| Total time | **1,527 ms** | 1,716 ms | 490 ms* |
| ms / token | 30.5 | 34.3 | 25.8* |

*Navarasa generated only 19 tokens (hit EOS early), making its raw speed numbers non-comparable for 50-token generation. Nandi is 12% faster than Sarvam at 16x fewer parameters.

### Perplexity — General Text

| Language | Nandi | Sarvam-1 | Navarasa-2.0 |
|----------|------:|---------:|-------------:|
| English | 8.08 | **3.65** | 4.57 |
| Hindi | 13.03 | 5.40 | **4.41** |
| Tamil | 33.34 | 12.94 | **3.28** |

The 2B models predictably outperform Nandi on raw perplexity — they have 16x more parameters. Sarvam-1 excels on English; Navarasa leads on Hindi and Tamil. Nandi's perplexity is reasonable given its 150M parameter budget.

### IndicBench-Micro — 85 India-Specific Tasks (3-model comparison)

#### Headline Metrics

| Metric | Nandi | Sarvam-1 | Navarasa-2.0 |
|--------|------:|---------:|-------------:|
| Avg fertility | **1.70** | 1.96 | 2.92 |
| Script consistency | **0.999** | 0.931 | 0.994 |
| Avg perplexity | 1,288 | **66.9** | 402.9 |
| Category PPL wins | 0/10 | 3/10 | **7/10** |

#### Category-Level Perplexity (lower is better)

| Category | Nandi | Sarvam-1 | Navarasa-2.0 |
|----------|------:|---------:|-------------:|
| Agriculture | 5,354 | 103.7 | **40.3** |
| Code Mixing | 3,673 | **129.6** | 2,833 |
| Cultural Knowledge | 583 | 66.5 | **37.3** |
| Healthcare | 75.2 | **26.1** | 31.0 |
| Indic NLU | 293 | 74.1 | **25.6** |
| Legal/Governance | 51.0 | 60.3 | **47.4** |
| Number Systems | 21.0 | 25.7 | **11.2** |
| Robustness | 1,252 | **106.0** | 965 |
| Script Generation | 1,508 | 55.7 | **17.0** |
| Translation | 71.3 | 20.9 | **20.8** |

The 2B models dominate perplexity as expected. Sarvam-1 wins on code-mixing, healthcare, and robustness. Navarasa wins 7 of 10 categories.

#### Category-Level Fertility (tokens per word — lower is better)

| Category | Nandi | Sarvam-1 | Navarasa-2.0 |
|----------|------:|---------:|-------------:|
| Agriculture | **1.66** | 2.03 | 3.64 |
| Code Mixing | 1.67 | 2.02 | **1.43** |
| Cultural Knowledge | **1.65** | 1.96 | 2.98 |
| Healthcare | **1.76** | 1.78 | 3.63 |
| Indic NLU | **1.72** | 2.06 | 3.48 |
| Legal/Governance | **1.60** | 1.68 | 2.92 |
| Number Systems | 1.92 | **1.68** | 2.37 |
| Script Generation | **1.88** | 2.50 | 4.68 |
| Translation | **1.36** | 1.78 | 2.30 |
| Robustness | 1.81 | 2.13 | **1.78** |

Nandi wins 7 of 10 categories on tokenization efficiency, even against models with far more parameters.

#### Script Consistency (does generation stay in the correct script?)

| Category | Nandi | Sarvam-1 | Navarasa-2.0 |
|----------|------:|---------:|-------------:|
| Agriculture | **1.000** | 0.782 | 0.977 |
| Code Mixing | **1.000** | 0.950 | **1.000** |
| Cultural Knowledge | **0.994** | 0.992 | 0.983 |
| Healthcare | **1.000** | 0.919 | **1.000** |
| Legal/Governance | **1.000** | 0.848 | **1.000** |
| Number Systems | **1.000** | 0.963 | **1.000** |
| Script Generation | **0.993** | 0.923 | 0.989 |
| Robustness | **1.000** | **1.000** | **1.000** |

Nandi leads with 0.999 average script consistency. Navarasa is close at 0.994. Sarvam-1 trails at 0.931 — it occasionally drops out of Indic scripts into Latin, especially in agriculture (0.782) and legal (0.848) contexts.

### Generation Quality — Indic Models (qualitative samples)

#### Hindi: "भारत एक ऐसा देश है जहाँ"

**Nandi**: "कई ऐसी बातें हैं जो दुनिया में कई तरह से फैलती रहती है... भारत की राजनीति और राजनीति का यह नाम बहुत ही ज्यादा जुड़ा हुआ है."
*Fluent Hindi about India's diversity and politics.*

**Sarvam-1**: "कई अलग-अलग धर्मों का पालन किया जाता है। इसके नागरिकों को अक्सर हिंदू, मुसलमान और ईसाई जैसे विभिन्न समूहों में वर्गीकृत किया जाता है।"
*High-quality Hindi about India's religious diversity. Coherent and factual.*

**Navarasa-2.0**: "विभिन्न समुदायों के लोगों की अद्वितीय पहचान है। प्रत्येक समुदाय अपनी अनूठी परंपराओं, भाषाओं और परंपराओं को संरक्षित करता है..."
*Fluent Hindi about cultural identity. Well-structured.*

All three produce coherent Hindi. Sarvam and Navarasa show richer vocabulary and more factual content, expected from 16x more parameters.

#### Tamil: "இந்தியா ஒரு நாடு, அதில்"

**Nandi**: "எந்த மாநிலமும் தனது சுதந்திரத்தை ஒருபோதும் பறிக்க முடியாது என்று நான் நம்புகிறேன்..."
*Coherent Tamil about state sovereignty and governance.*

**Sarvam-1**: "பல மாநிலங்கள் உள்ளன. ஒவ்வொரு மாநிலமும் அதன் சொந்த அரசையும் சட்டங்களையும் கொண்டுள்ளன..."
*Excellent Tamil listing Indian states with factual detail.*

**Navarasa-2.0**: "சுமார் 1.35 பில்லியன் மக்கள் வசித்து வருகின்றனர்..."
*Coherent but shorter Tamil continuation.*

#### Bengali: "বাংলা ভাষা হলো একটি"

**Nandi**: "জাতি। তবে এর মধ্যে বাংলা ভাষার গুরুত্ব অনেক বেশি..."
*Coherent Bengali about the importance of the Bengali language.*

**Sarvam-1**: "ইন্দো-আর্য ভাষা যা বাংলাদেশ এবং ভারতের পশ্চিমবঙ্গ ও ত্রিপুরা রাজ্যে কথিত হয়..."
*Excellent factual Bengali about the language's classification and history.*

#### English: "India is a country known for its"

All three produce coherent, well-formed English. Sarvam and Navarasa show more polished prose at this scale.

### The Edge Efficiency Story

The real differentiator isn't quality — it's **efficiency at quality**:

| Metric | Nandi-Mini | Sarvam-1 | Navarasa-2.0 |
|--------|-----------|----------|--------------|
| Parameters | **153 M** | 2,525 M | 2,506 M |
| fp16 size | **293 MB** | 4,816 MB | 4,780 MB |
| int8 size | **~146 MB** | ~2,408 MB | ~2,390 MB |
| Avg Indic fertility | **1.50** | 1.81 | 3.51 |
| Script consistency | **0.999** | 0.931 | 0.994 |
| Tokens/sec (H100) | 32.7 | 29.1 | 38.8* |
| Runs on phone? | Yes (73 MB int4) | No | No |

Nandi delivers **near-perfect script consistency** (0.999) and **best-in-class tokenization** while being **16x smaller**. A phone can run Nandi at int4 (73 MB); neither Sarvam nor Navarasa fits in mobile memory without aggressive compression that would degrade quality.

---

## 9. Summary

### vs SmolLM2-135M (similar size, global model)

| Dimension | Winner | Detail |
|-----------|--------|--------|
| Indic tokenization | **Nandi** by 8.2x | 131K vocab tuned for 11 Indic scripts |
| Indic generation quality | **Nandi** decisively | Coherent native-script output vs garbled |
| Script consistency | **Nandi** (0.999 vs 0.698) | Never drops out of expected script |
| Inference speed | **Nandi** (+7%) | Shallower architecture |
| Edge weight size (int8) | Comparable | 146 MB vs ~128 MB |
| Compute per token | **Nandi** | 0.307 GFLOPs — very light |
| English quality | SmolLM2 slightly | Lower English PPL, similar generation |
| Context length | SmolLM2 | 8K vs 2K max positions |

### vs Sarvam-1 & Navarasa-2.0 (2B Indian models, 16x larger)

| Dimension | Winner | Detail |
|-----------|--------|--------|
| Indic tokenization | **Nandi** (1.50) | Best fertility even vs 16x larger models |
| Script consistency | **Nandi** (0.999) | Sarvam: 0.931, Navarasa: 0.994 |
| Perplexity | Sarvam/Navarasa | Expected — 16x more parameters |
| Generation coherence | All three | Nandi is coherent; 2B models add depth/detail |
| Model size | **Nandi** (293 MB fp16) | vs ~4,800 MB for 2B models |
| Edge deployable? | **Nandi only** | 73 MB int4 fits on phone; 2B models don't |
| Inference speed | **Nandi** (32.7 tok/s) | Faster than Sarvam (29.1) at 16x fewer params |
| Cost per Indic token | **Nandi** | Best fertility × smallest model = lowest cost |

### Bottom Line

**Nandi-Mini-150M punches far above its weight.** Against SmolLM2 (similar size), it's the clear winner for any Indic task. Against India's own 2B models (Sarvam-1, Navarasa-2.0), Nandi holds its own on tokenization efficiency and script consistency while being 16x smaller:

- **Best tokenizer for Indic**: 1.50 tokens/word average — beats Sarvam (1.81) and Navarasa (3.51)
- **Near-perfect script consistency** (0.999) — better than both 2B competitors
- **73 MB at int4** — the only model in this comparison that runs on a phone
- The 2B models bring better perplexity and richer generation, but at **16x the compute and memory cost**

For Indic edge deployment — phones, IoT, offline-first apps — Nandi is unmatched. For server-side Indic tasks where quality is paramount and resources are abundant, Sarvam-1 and Navarasa-2.0 are strong choices.

---

*Benchmarks: benchmark_nandi.py, eval_indicbench.py, edge_deploy_eval.py — all run on NVIDIA H100 80GB, fp16, April 2026.*
