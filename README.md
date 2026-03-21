# evo_moe — Evolving MoE Routers with Evolution Strategies

> **Proof of concept.** This personal project explores a research idea at an early stage, and is also a mean of buliding intuition about recent LLMs advancements.
>  Experimentation at scale would be needed to draw any definitive
> conclusions.

## The Idea

Fine-tuning large language models is expensive. Even parameter-efficient methods
like LoRA still require backpropagation, gradient computation, and to keep optimizer
states.

Evolution strategies (ES) can optimize functions without computing any gradient, as they are black-box optimization techniques.
However, they usually struggle as dimensionality grows (they work well up
to tens or hundreds of thousands of parameters, but millions become impractical).

**Mixture-of-Experts** (MoE) architectures contain small components that may be worth evolving using evolutionary methods.
In an MoE transformer, each layer contains a set of expert MLPs and a *router*
that decides which experts process each token. The router is a small linear
projection far smaller than the experts it controls.

Consider [OLMoE-1B-7B](https://huggingface.co/allenai/OLMoE-1B-7B-0924-Instruct),
the model used here:

| Component | Parameters |
|---|---|
| Single expert MLP (1 of 64) | ~6.3M |
| All 64 experts in one layer | ~403M |
| **Router of one layer** | **131,072** (64 × 2048) |

The router is small enough for ES (~131K params), and may be enough to meaningfully change model behavior, since it controls
which experts are used for every token.

### In this approach, we propose to only consider the last layer MoE router. Reasons are:

1. **Dimensionality.** Evolving all 16 routers would mean ~2.1M parameters,
   too many for ES.
2. **Intuition.** The last layer operates on the most abstract representations,
   changing expert routing there has a direct impact on the output distribution.
3. **Computational shortcut.** With all prior layers frozen, we can pre-compute
   and cache intermediate activations shared across all solutions (routers in the ES population), dramatically
   speeding up evaluation.


## The ES Optimization Process

The evolutionary loop acts as a global optimization framework. In this work we use [OpenAI-ES](https://arxiv.org/abs/1703.03864), which maintains a population of perturbed router solutions, evaluates each one, and updates the centroid toward better solutions.

## Two Possibilities for Computing the Fitness 

### 1. Reward-Model Fitness

Similar to the evaluation step of RLHF. A pre-trained reward model ([GRM-Llama3.2-3B](https://huggingface.co/Ray2333/GRM-Llama3.2-3B-rewardmodel-ft)) provides the signal:

$$
fitness(\theta) = E_{prompt} [ RewardModel(prompt, Generate_{\theta}(prompt)) ]
$$

For each solution (router): set the router's weights in the LLM, generate a full response for every
evaluation prompt, score each (prompt, response) pair with the reward model,
and average.

**Prompt KV-cache reuse.** Since only the last layer's router changes, the KV
cache from layers 0..N−2 is identical for all routers and for all evaluation prompts. Therefore we compute it once and reuse it at evaluation time.

**Drawback:** autoregressive generation is still needed for every router ×
every prompt, making this mode slow, even with the use of prompt pre-computed KV cache and standard KV cache computed during the autoregressive generation.

### 2. DPO Fitness

Uses a [DPO](https://arxiv.org/abs/2305.18290)-inspired objective that avoids
generation entirely. 

A language model is fundamentally a machine that assigns probabilities to
sequences of tokens:

$$
P(x_1, x_2, ..., x_n) = \prod_{t=1}^{n} P(x_t \mid x_1, ..., x_{t-1})
$$

The DPO fitness exploits this directly. Given a preference dataset
([UltraFeedback](https://huggingface.co/datasets/HuggingFaceH4/ultrafeedback_binarized))
where each example contains a prompt, a *chosen* (preferred) response, and a
*rejected* response:

$$
fitness(\theta) = E_{(chosen, rejected)} [ \log P_{\theta}(chosen \mid prompt) - \log P_{\theta}(rejected \mid prompt) ]
$$

A good router should assign higher probability to preferred responses.

**Activation-cache speedup.** This mode does not require autoregressive
generation at all. We only need a forward pass to compute log-probabilities.
Since only the last layer's router changes, we can pre-compute the
hidden states entering the last layer for every
sequence in the evaluation set once at the beginning of the run. Each solution (router) evaluation then only runs the
cheap tail of the model, since it is the only changing part:

```
Cached activations → LayerNorm → MoE FFN (with solution router) → Residual → Final RMSNorm → LM Head
```


**KL Divergence penalty.** To prevent the evolved router from drifting too far from the
pre-trained one, a Kullback-Leibler-style penalty term is applied.

## Limitations

- **DPO vs Reward-Model signal.** The DPO fitness measures how well the model
  discriminates between existing preferred/dispreferred responses, not how
  well it generates good ones.
- **Overfitting.** Evaluating on a fixed set (relativeley small due to hardward limitations) across generations risks
  overfitting. This is mitigated by the very small number of modified
  parameters (~131K out of 7B).
- **Scale.** All experiments run on a single consumer GPU with a small
  evaluation set. Larger populations, more samples, and proper benchmarking
  would require more compute.

## Installation

**Requirements:** Python ≥ 3.10, CUDA (tested on RTX 2080 Ti 12 GB).

```bash
pip install -r requirements.txt
```

The OLMoE model downloads automatically from HuggingFace on first run.
To download manually:

```bash
python -c "from huggingface_hub import snapshot_download; snapshot_download('allenai/OLMoE-1B-7B-0924-Instruct', local_dir='models/olmoe')"
```

Then set `model_name="models/olmoe"` in the `CONFIG` block of
`scripts/run_experiment.py`.

## Usage

```bash

# Start run
python scripts/run_experiment.py

# Or start run with DPO fitness (recommended)
python scripts/run_experiment.py --fitness dpo

# Resume a previous run
python scripts/run_experiment.py --resume outputs/olmoe_router_dpo/<write_your_xp_folder_name_here>
```

Hyperparameters are in the `CONFIG` block at the top of
`scripts/run_experiment.py`.

## Project Structure

```
evo_moe/
├── src/
│   ├── config.py            # Dataclass-based configuration
│   ├── model_wrapper.py     # OLMoE wrapper: activation caching, KV cache, router I/O
│   ├── data.py              # Preference dataset loading and tokenisation
│   ├── evolution.py         # OpenAI-ES optimizer (ask/tell interface)
│   ├── fitness_dpo.py       # DPO fitness (activation cache, no generation)
│   ├── fitness_reward.py    # Reward-model fitness (generation + scoring)
│   └── reward_model.py      # Reward model wrapper (GRM-Llama3.2-3B)
├── scripts/
│   ├── run_experiment.py    # Main entry point
│   └── plot_results.py      # Plotting utilities
├── tests/
│   └── test_pipeline.py
├── requirements.txt
└── README.md
```

## Results

Current results are limited. I ran a few experiments showing that the ES process is able to find routers that *significantly increase* the DPO/Reward Model fitness scores. However these early experiments lacked the implementation of a KL penalty and evolved routers that quickly learned fitness hacking, mostly by producing non-human gibberish output.

I ran a few more experiments by adding a KL penalty to the fitness function, this time ensuring that the evolved routers did not produce gibberish sequences. However, the fitness gains were *less significant* (0.15 baseline -> 0.17 using DPO, which I wouldn't risk interpreting yet), and analyzing the output sentences did not reveal obvious improvements in quality, although they were different from the sequences produced by the baseline router.

## Future Directions

- **LoRA-compressed experts.** Applying LoRA to expert MLPs would compress each
  expert's trainable parameters to a few thousand, opening the door to
  co-evolving experts alongside the router, leading to gradient-free LoRA fine-tuning.

## Hardware

Tested on NVIDIA RTX 2080 Ti (12 GB VRAM), 48 GB system RAM.
OLMoE-1B-7B in NF4 4-bit uses ~3.5 GB VRAM. Both fitness modes fit on a
single 12 GB GPU.
