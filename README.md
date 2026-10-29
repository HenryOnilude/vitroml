# Vitro — Client-Side Transformer Inference and Mid-Run Intervention Engine

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Live Demo](https://img.shields.io/badge/Live-vitroml.com-09090b)](https://vitroml.com)

**Live workbench:** [https://vitroml.com](https://vitroml.com)

<img width="1283" height="803" alt="Screenshot 2026-10-29 at 23 38 37" src="https://github.com/user-attachments/assets/73061ed7-1a8b-4a06-9529-6b37cc4af031" />


An interactive, in-browser visualizer and mechanistic interpretability environment written in TypeScript and Svelte. The tool executes autoregressive transformer inference locally on `Float32Array` buffers, allowing users to step through individual layers, inspect intermediate activation tensors, and apply mid-run interventions (such as attention head ablation, activation patching, and steering vectors) without a server-side runtime.

---

## 1. Project Overview

Existing educational and interpretability tools typically fall into two categories:

1. **Client-side, read-only visualizers** (e.g., *Transformer Explainer*, *LLM Visualization* by Brendan Bycroft): These run directly in the browser to visualize static activations and attention weights, but do not support dynamic state mutation during the forward pass.
2. **Intervention libraries** (e.g., *TransformerLens*, *Neuronpedia*, *Baukit*): These enable causal interventions (activation patching, head ablation, steering), but require Python runtimes, PyTorch environments, or remote API/inference servers.

This tool implements both stepped execution and mutable mid-run intervention directly in the browser using a hand-written TypeScript tensor engine. All computation, weight parsing, and state modifications occur client-side in memory.

> **Key finding:** Ablating all 36 attention heads across 10 prompts shows that layer 5, head 4 is the most consistently influential head in the model, it is the single most impactful head in 8 of 20 non-BOS samples, with a mean logit shift of ΔL2 285 versus 190 for the next-ranked head. Meanwhile the heads that send the most attention to the first token, the "attention sink" pattern — are the least impactful of all: removing them barely changes the output. See [Section 7](#7-ablation-sweep-which-heads-matter-and-when) for the methodology, full data, and limitations.

---

## 2. Architecture Summary

The codebase is split into three decoupled layers:

```
src/
├── engine/          # Framework-agnostic tensor math, hooks, loader, and forward pass
│   ├── tensor.ts             # Tensor wrapper (shape, strides, zero-copy slicing, cloning)
│   ├── model.ts              # ModelConfig and ModelWeights interfaces
│   ├── hooks.ts              # HookRegistry with chained interceptors
│   ├── loader.ts             # Parser for llama2.c .bin format (zero-copy Float32Array views)
│   ├── tokenizer.ts          # SentencePiece BPE tokenizer parser and decoder
│   ├── forward.ts            # Stepped forward-pass generator (forwardPass) and un-stepped forward()
│   ├── interventions.ts      # Intervention implementations (zeroAblateHead, patchActivation, etc.)
│   ├── analysis.ts           # Comparative ablation analysis utility (compareHeadAblation)
│   └── ops/                  # Pure numerical primitives (matmul, softmax, rmsnorm, rope, swiglu)
├── bridge/          # Framework-independent state orchestration
│   ├── observable.ts         # Minimal store implementation fulfilling the Svelte store contract
│   └── controller.ts         # ForwardController with requestAnimationFrame scheduling
└── ui/              # Reactive interface components (Svelte 4 + Tailwind CSS)
    └── components/
        ├── HeroSection.svelte          # Landing explainer with animated forward-pass motif
        ├── WorkbenchToolbar.svelte     # Prompt input, presets, transport controls, token tape
        ├── NarrativeBar.svelte         # Plain-language live narration of the current phase
        ├── ArchitecturePipeline.svelte # Stage-by-stage journey map with the traveling token
        ├── ResidualStreamView.svelte   # SVG-based residual stream activation inspector
        ├── AttentionMap.svelte         # Per-head attention weight rows (Attention Field)
        ├── InterventionPanel.svelte    # Ablation/steering bench wrapping HeadGrid and LogitWaterfall
        ├── HeadGrid.svelte             # 36-square attention head importance map (Circuit Map)
        └── LogitWaterfall.svelte       # Baseline vs. intervened top-token comparison
```

- **Engine (`src/engine/`)**: Has zero DOM or UI dependencies. Matrix operations write into pre-allocated scratch buffers following the memory layout of `llama2.c` (`run.c`). Ninety-three named hook points allow intermediate tensors to be read or replaced.
- **Bridge (`src/bridge/`)**: Exposes observable state conforming to the Svelte store contract (`subscribe`, `get`, `set`) without importing Svelte packages. A controller manages generator execution and yields control to the browser event loop using a `requestAnimationFrame` trampoline.
- **UI (`src/ui/`)**: Built in Svelte 4. Renders reactive components for tokenization, stepping state, interactive residual stream activations (via SVG), and side-by-side logit perturbation tables.

---

## 3. Model Details

The tool targets the **stories15M** checkpoint from Andrej Karpathy's [tinyllamas repository on Hugging Face](https://huggingface.co/karpathy/tinyllamas) (MIT License), trained on the synthetic TinyStories dataset.

- **Architecture**: Llama-2 decoder (RMSNorm, Rotary Positional Embeddings, SwiGLU FFN activation, shared token embedding and unembedding classifier)
- **Parameters**: 15.2M
- **Hyperparameters**:
  - Layers ($L$): 6
  - Attention query heads ($H$): 6
  - Key/Value heads ($KV$): 6
  - Model dimension ($d$): 288
  - Hidden FFN dimension ($d_{ffn}$): 768
  - Vocabulary size ($V$): 32,000 (SentencePiece BPE)
  - Max sequence length: 256 tokens
  - Checkpoint size: ~58.00 MB unquantized float32 binary format

---

## 4. Local Development and Testing

### Requirements
- Node.js >= 18.0.0
- npm >= 9.0.0

### Setup and Verification
```bash
# Install dependencies
npm install

# Run unit test suite (tensor math, ops, hooks, stepping)
npm test

# Type-check TypeScript sources
npm run typecheck

# Download model/tokenizer cache and verify baseline forward pass
npm run verify

# Verify head ablation perturbation against baseline
npm run verify:interventions
```

### Running the Web Visualizer
```bash
npm run dev
```

### Building for Production
```bash
npm run build
```

---

## 5. Correctness Verification

To confirm numerical alignment with the reference C implementation (`run.c`), a single forward pass on the beginning-of-sequence token `1` (`<s>`) at position `0` was evaluated against the unquantized `stories15M.bin` checkpoint:

```
Top 5 predictions:
  #1: token_id=9038 ("Once") logit=12.2900
  #2: token_id=3118 ("One")  logit=10.5333
  #3: token_id=365  ("L")    logit=8.4984
  #4: token_id=4335 ("Tom")  logit=8.0649
  #5: token_id=11230 ("Anna") logit=7.6199
```

This top-token distribution reflects the TinyStories training objective, in which narrative sequences characteristically open with `"Once upon a time"` or `"One day"`.

---

## 6. Supported Interventions

Interventions are implemented as modular `HookFn` callbacks registered against designated hook points:

- **Zero-Ablation (`zeroAblateHead`)**: Zeros the output slice corresponding to a targeted attention head at `layer.L.attn.head_out` (dimensions $[h \cdot d_{head}, (h+1) \cdot d_{head})$) before the linear projection $W_O$.
- **Mean-Ablation (`meanAblateHead`)**: Replaces a head's output slice with a precomputed mean vector or baseline scalar to avoid sending out-of-distribution zero vectors into downstream layers.
- **Activation Patching (`patchActivation`)**: Overwrites the activation tensor at an arbitrary hook point with an activation tensor cached from a different prompt run (useful for causal tracing of specific facts or tokens).
- **Steering Vector Injection (`addSteeringVector`)**: Adds an offset vector scaled by a hyperparameter $\alpha$ ($\vec{x} \leftarrow \vec{x} + \alpha \vec{v}$) into the residual stream at either `attn.resid` or `ffn.resid`.

---

## 7. Ablation Sweep: Which Heads Matter, and When

All 36 heads were zero-ablated across 10 story-style prompts at 3 positions each (BOS, mid-prompt, final prompt token), 30 conditions, 1,080 forward passes. Impact is measured as $\|\Delta\text{logits}\|_2$ between the baseline and ablated output at that position. Raw data: `scripts/ablation_sweep_results.json` (reproduce with `npx tsx scripts/ablation_sweep.ts`).

Top 12 heads by mean impact:

| Head | Mean $\Delta$ (all) | Mean $\Delta$ (non-BOS) | In top 5 (non-BOS) | Attn → BOS | Attn → prev | Attn → self |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **L5:H4** | **288.3** | **285.4** | **75%** | 0.57 | 0.13 | 0.45 |
| L0:H5 | 137.0 | 190.1 | 50% | 0.54 | 0.12 | 0.53 |
| L0:H3 | 122.7 | 162.0 | 35% | 0.42 | **0.51** | 0.45 |
| L0:H2 | 118.1 | 163.5 | 40% | 0.49 | 0.24 | 0.45 |
| L0:H4 | 117.1 | 156.4 | 35% | 0.45 | 0.37 | 0.51 |
| L1:H5 | 109.0 | 139.5 | 30% | 0.42 | 0.22 | 0.46 |
| L3:H1 | 105.4 | 125.0 | 35% | 0.70 | 0.09 | 0.41 |
| L4:H0 | 104.5 | 106.9 | 15% | 0.53 | 0.11 | 0.41 |
| L5:H3 | 86.3 | 119.5 | 20% | 0.38 | 0.33 | 0.53 |
| L0:H1 | 83.3 | 108.1 | 15% | 0.67 | 0.13 | 0.40 |

Bottom 3 (least impactful): **L5:H1** (21.8, attn→BOS 0.84), **L2:H3** (27.0), **L4:H2** (29.5, attn→BOS 0.87).

### Findings

**1. L5:H4 is a consistently high-impact head, the original result was real, not an outlier.** It is the top-impact head in all 10 BOS conditions and in 8 of 20 non-BOS conditions and ranks in the top 5 in 75% of non-BOS samples. Its mean non-BOS impact (285.4) is ~1.5× the second-ranked head (L0:H5, 190.1). However, the original single-point comparison overstated the gap: vs. L0:H0 the sweep shows ~2.7×, not ~25×, because impact varies substantially by context ($\sigma \approx 133$).

**2. Heads that dump attention on BOS are the least impactful consistent with "attention sink" heads.** The three heads with the highest BOS attention share (L4:H2 0.87, L5:H1 0.84, L0:H0 0.80) occupy the bottom of the impact table. This matches the attention-sink literature: heads that route attention to the start token are effectively holding a "no-op" register, so removing them changes little. **L5:H4 is the instructive exception** — it also weights BOS heavily (0.57) but is the *most* impactful head, and unlike the sink heads it puts 0.45 of its mass on the current position. Same surface attention pattern, different function: the sink heads treat BOS as a place to park probability mass; L5:H4 treats it as an anchor that carries real signal into the output projection.

**3. L0:H3 and L2:H0 behave like "previous-token heads."** They place 0.51 and 0.56 of their mean attention on the immediately preceding token, the largest such fractions in the model and carry real weight (non-BOS means 162.0 and 102.0). This resembles the previous-token component described in the induction-heads literature (the first half of the "previous token → copy" circuit), though we have not demonstrated the copying half here.

### Limitations

- **This is one 15M-parameter model trained only on TinyStories.** Every claim above is about `stories15M`, not transformers in general. Claiming more would require repeating the sweep on the other tinyllamas checkpoints (stories42M, stories110M) and on non-story text.
- **L2 norm measures magnitude, not meaning.** A large $\|\Delta\text{logits}\|_2$ says the head matters to the output distribution, not *what feature* it encodes. Naming what L5:H4 computes would need activation-level analysis (e.g., value-vector inspection or feature-level ablations), not just output shifts.
- **Zero-ablation is blunt.** It removes the head's output entirely rather than isolating what it writes. The high variance across positions ($\sigma/\mu \approx 0.5$) means most heads matter *in context*, not categorically.
- **Sample size**: 30 conditions over 10 prompts, all in the TinyStories register. The 10 BOS conditions are the identical computation (prompt-independent), so non-BOS stats are reported separately where they differ.

---

## 8. Limitations

- **Small Model Scale (15.2M Parameters)**: Unlike larger models (e.g., Llama-2 7B/70B), small models exhibit heavy polysemanticity. Attention heads frequently multiplex multiple syntactic and associative functions, making clean mechanistic isolation of isolated circuits challenging.
- **Dataset Specialization**: `stories15M` was trained exclusively on the synthetic TinyStories dataset. Its representations capture simple narrative English syntax and basic vocabulary, but do not generalize to open-domain knowledge, complex reasoning, or multi-turn dialogue.
- **Synchronous Computation**: While the controller yields to the browser's render loop via `requestAnimationFrame` between layer steps, the tensor operations within each step execute synchronously on the main JavaScript thread. For `stories15M` (matrix sizes up to $288 \times 768$), latency remains under 3ms per step; however, running larger models smoothly will require migrating heavy matmuls to Web Workers or WebGPU.

---

## 9. License

This project is released under the [MIT License](LICENSE).  
The underlying `stories15M.bin` model weights are authored by Andrej Karpathy and distributed under the [MIT License](https://huggingface.co/karpathy/tinyllamas).
