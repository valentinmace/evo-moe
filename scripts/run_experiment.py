"""
Main entry point for the evolutionary MoE router search.

Two fitness modes are supported (set via CONFIG or ``--fitness`` CLI flag):

  **reward** — generate responses with the LLM, score with a reward model.
    Slow (~minutes/candidate) but gives a direct alignment signal.

  **dpo** — compare log-probabilities of pre-existing chosen/rejected
    responses using cached activations.  Very fast (~seconds/candidate)
    but an indirect proxy signal.

The evolution loop is identical for both: only the setup phase and
evaluator differ.

Usage
-----
    # New experiment (reward fitness, the default)
    python scripts/run_experiment.py

    # New experiment with DPO fitness
    python scripts/run_experiment.py --fitness dpo

    # Resume from a previous run
    python scripts/run_experiment.py --resume outputs/olmoe_router_reward/20260317_111719
"""

import argparse
import json
import logging
import os
import sys
import time
from dataclasses import replace
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import DataConfig, EvolutionConfig, ExperimentConfig, ModelConfig
from src.data import load_preference_sequences, load_prompts
from src.evolution import OpenAIES
from src.fitness_dpo import DPOFitnessEvaluator
from src.fitness_reward import RewardFitnessEvaluator
from src.model_wrapper import OLMoEWrapper
from src.reward_model import RewardModel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Experiment configuration — edit here for new experiments
# ---------------------------------------------------------------------------

CONFIG = ExperimentConfig(
    model=ModelConfig(
        model_name="allenai/OLMoE-1B-7B-0924-Instruct",
        load_in_4bit=True,
        device="cuda",
        target_layer_idx=-1,
    ),
    data=DataConfig(
        dataset_name="HuggingFaceH4/ultrafeedback_binarized",
        dataset_split="train_prefs",
        num_samples=128,
        max_prompt_length=256,
        max_response_length=128,
    ),
    evolution=EvolutionConfig(
        population_size=40,
        sigma=0.01,
        learning_rate=0.01,
        num_generations=100,
        antithetic_sampling=True,
        fitness_batch_size=128,
        seed=42,
    ),
    output_dir="outputs",
    #experiment_name="olmoe_router_reward",
    experiment_name="olmoe_router_dpo",
    fitness_type="dpo",
    reward_model_name="Ray2333/GRM-Llama3.2-3B-rewardmodel-ft",
    max_new_tokens=25,
    log_every=2,
    sample_every=2,
    num_sample_prompts=5,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _create_run_dir(config: ExperimentConfig) -> Path:
    """Create a timestamped run directory under outputs/<experiment_name>/."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(config.output_dir) / config.experiment_name / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "checkpoint").mkdir(exist_ok=True)
    (run_dir / "samples").mkdir(exist_ok=True)
    return run_dir


def _save_config(config: ExperimentConfig, run_dir: Path) -> None:
    path = run_dir / "config.json"
    with open(path, "w") as f:
        json.dump(config.to_dict(), f, indent=2)
    logger.info(f"Config saved → {path}")


def _load_config(run_dir: Path) -> ExperimentConfig:
    path = run_dir / "config.json"
    with open(path) as f:
        return ExperimentConfig.from_dict(json.load(f))


def _save_checkpoint(
    es: OpenAIES,
    baseline_fitness: float,
    router_shape: tuple[int, int],
    run_dir: Path,
) -> None:
    """Atomically save the ES checkpoint (write tmp then rename)."""
    ckpt = es.get_checkpoint()
    ckpt["baseline_fitness"] = baseline_fitness
    ckpt["router_shape"] = router_shape

    tmp_path = run_dir / "checkpoint" / "latest.pt.tmp"
    final_path = run_dir / "checkpoint" / "latest.pt"
    torch.save(ckpt, tmp_path)
    os.replace(tmp_path, final_path)


def _save_best_router(
    es: OpenAIES,
    router_shape: tuple[int, int],
    run_dir: Path,
) -> None:
    torch.save(
        {
            "weights": es.state.best_theta.copy(),
            "router_shape": router_shape,
            "fitness": es.state.best_fitness,
            "generation": es.state.generation,
        },
        run_dir / "best_router.pt",
    )


def _save_history(history: dict, run_dir: Path) -> None:
    with open(run_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)


def _generate_responses(
    wrapper: OLMoEWrapper,
    prompts: list[str],
    max_new_tokens: int,
) -> list[str]:
    """Generate a response for each prompt using the currently-set router."""
    responses: list[str] = []
    for prompt_text in prompts:
        tokenizer = wrapper.tokenizer
        if tokenizer.chat_template is not None:
            text = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt_text}],
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            text = prompt_text

        input_ids = tokenizer(text, return_tensors="pt").input_ids.to(wrapper.device)
        with torch.no_grad():
            out = wrapper.model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )
        generated = out[0, input_ids.shape[1]:]
        responses.append(tokenizer.decode(generated, skip_special_tokens=True))
    return responses


def _save_samples(
    sample_prompts: list[str],
    baseline_responses: list[str],
    evolved_responses: list[str],
    generation: int,
    best_fitness: float,
    run_dir: Path,
) -> None:
    samples = [
        {
            "prompt": p,
            "baseline_response": b,
            "evolved_response": e,
        }
        for p, b, e in zip(sample_prompts, baseline_responses, evolved_responses)
    ]
    data = {
        "generation": generation,
        "best_fitness": best_fitness,
        "samples": samples,
    }
    path = run_dir / "samples" / f"gen_{generation:04d}.json"
    with open(path, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def plot_fitness_curve(history: dict, run_dir: Path) -> None:
    """Generate and save the fitness curve plot."""
    gens_data = history["generations"]
    if not gens_data:
        return

    generations = [g["generation"] for g in gens_data]
    mean_fit = [g["mean_fitness"] for g in gens_data]
    max_fit = [g["max_fitness"] for g in gens_data]
    best_ever = [g["best_ever"] for g in gens_data]
    baseline = history["baseline_fitness"]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(generations, best_ever, linewidth=2, label="Best ever (cumulative)")
    ax.plot(generations, max_fit, linewidth=1, alpha=0.7, label="Max (per gen)")
    ax.plot(generations, mean_fit, linewidth=1, alpha=0.5, label="Mean (per gen)")
    ax.axhline(baseline, color="gray", linestyle="--", linewidth=1, label=f"Baseline ({baseline:.2f})")

    ax.set_xlabel("Generation")
    ax.set_ylabel("Fitness")
    ax.set_title("Evolutionary Router Search — Fitness over Generations")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(run_dir / "fitness_curve.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(config: ExperimentConfig, run_dir: Path, resume_ckpt: dict | None = None) -> None:
    # ------------------------------------------------------------------ 1. LLM
    logger.info("Loading LLM…")
    wrapper = OLMoEWrapper(config.model)

    # ------------------------------------------------------------------ 2. Prompts (always load for qualitative samples)
    logger.info("Loading prompts…")
    prompts = load_prompts(config.data)
    sample_prompts = prompts[: config.num_sample_prompts]

    # ------------------------------------------------------------------ 3. Fitness evaluator (branched by type)
    if config.fitness_type == "reward":
        logger.info("Loading reward model…")
        reward_model = RewardModel(
            model_name=config.reward_model_name,
            load_in_4bit=config.model.load_in_4bit,
            device=config.model.device,
        )
        evaluator = RewardFitnessEvaluator(
            wrapper=wrapper,
            reward_model=reward_model,
            prompts=prompts,
            max_new_tokens=config.max_new_tokens,
            rm_batch_size=config.evolution.fitness_batch_size,
        )
    elif config.fitness_type == "dpo":
        logger.info("Loading preference sequences for DPO fitness…")
        sequences = load_preference_sequences(config.data, wrapper.tokenizer)
        logger.info("Caching activations (one-time forward passes)…")
        cache = wrapper.cache_activations([vars(s) for s in sequences])
        evaluator = DPOFitnessEvaluator(
            wrapper=wrapper,
            cache=cache,
            batch_size=config.evolution.fitness_batch_size,
            beta=config.dpo_beta,
        )
    else:
        raise ValueError(
            f"Unknown fitness_type {config.fitness_type!r} — "
            f"expected 'reward' or 'dpo'"
        )

    # ------------------------------------------------------------------ 4. Baseline or resume
    if resume_ckpt is not None:
        baseline_fitness = resume_ckpt["baseline_fitness"]
        start_gen = resume_ckpt["generation"]
        es = OpenAIES.from_checkpoint(resume_ckpt)
        logger.info(
            f"Resumed from generation {start_gen} | "
            f"baseline={baseline_fitness:.4f} | "
            f"best_ever={es.state.best_fitness:.4f}"
        )

        # Load existing history so we append to it
        history_path = run_dir / "history.json"
        if history_path.exists():
            with open(history_path) as f:
                history = json.load(f)
        else:
            history = {"baseline_fitness": baseline_fitness, "generations": []}
    else:
        logger.info("Evaluating baseline (pre-trained router)…")
        baseline_fitness = evaluator.evaluate(wrapper.get_router_weights())
        logger.info(f"Baseline fitness ({config.fitness_type}): {baseline_fitness:.4f}")
        start_gen = 0

        es = OpenAIES(
            theta_init=wrapper.get_router_weights(),
            sigma=config.evolution.sigma,
            learning_rate=config.evolution.learning_rate,
            population_size=config.evolution.population_size,
            antithetic_sampling=config.evolution.antithetic_sampling,
            seed=config.evolution.seed,
        )

        history: dict = {"baseline_fitness": baseline_fitness, "generations": []}

        # Generate baseline samples once
        logger.info(f"Generating baseline samples for {len(sample_prompts)} prompts…")
        wrapper.reset_router()
        baseline_responses = _generate_responses(wrapper, sample_prompts, config.max_new_tokens)
        baseline_data = {
            "baseline_fitness": baseline_fitness,
            "samples": [
                {"prompt": p, "response": r}
                for p, r in zip(sample_prompts, baseline_responses)
            ],
        }
        with open(run_dir / "samples" / "baseline.json", "w") as f:
            json.dump(baseline_data, f, indent=2, ensure_ascii=False)

    # Load baseline responses for comparison in later samples
    baseline_samples_path = run_dir / "samples" / "baseline.json"
    if baseline_samples_path.exists():
        with open(baseline_samples_path) as f:
            baseline_responses = [s["response"] for s in json.load(f)["samples"]]
    else:
        baseline_responses = ["(not available)"] * len(sample_prompts)

    # ------------------------------------------------------------------ 5. Evolution
    num_gens = config.evolution.num_generations
    pop_size = config.evolution.population_size
    logger.info(
        f"Starting evolutionary search ({config.fitness_type} fitness) — "
        f"gen {start_gen}→{num_gens} × {pop_size} candidates"
    )

    prev_best = es.state.best_fitness

    gen_bar = tqdm(range(start_gen, num_gens), desc="Evolution", unit="gen")
    for gen in gen_bar:
        population = es.ask()

        cand_bar = tqdm(
            total=pop_size,
            desc=f"  Gen {gen:3d} eval",
            unit="cand",
            leave=False,
        )
        gen_start = time.perf_counter()
        fitnesses: list[float] = []
        for candidate in population:
            fitnesses.append(evaluator.evaluate(candidate, progress_bar=cand_bar))
        gen_elapsed = time.perf_counter() - gen_start
        cand_bar.close()

        es.tell(fitnesses)

        gen_stats = {
            "generation": gen,
            "mean_fitness": float(np.mean(fitnesses)),
            "max_fitness": float(np.max(fitnesses)),
            "min_fitness": float(np.min(fitnesses)),
            "best_ever": es.state.best_fitness,
        }
        history["generations"].append(gen_stats)

        gen_bar.set_postfix_str(
            f"mean={gen_stats['mean_fitness']:+.2f}  "
            f"max={gen_stats['max_fitness']:+.2f}  "
            f"best={gen_stats['best_ever']:+.2f}  "
            f"({gen_elapsed:.0f}s/gen)"
        )

        # --- Save checkpoint every generation (atomic overwrite) ---
        _save_checkpoint(es, baseline_fitness, wrapper.router_shape, run_dir)
        _save_history(history, run_dir)

        # --- Update best_router.pt when we find a new best ---
        if es.state.best_fitness > prev_best:
            _save_best_router(es, wrapper.router_shape, run_dir)
            logger.info(
                f"  New best router! fitness={es.state.best_fitness:.4f} "
                f"(was {prev_best:.4f})"
            )
            prev_best = es.state.best_fitness

        # --- Periodic logging ---
        if gen % config.log_every == 0:
            logger.info(
                f"Gen {gen:4d} | "
                f"mean: {gen_stats['mean_fitness']:+.4f} | "
                f"max: {gen_stats['max_fitness']:+.4f} | "
                f"best ever: {gen_stats['best_ever']:+.4f} | "
                f"{gen_elapsed:.0f}s"
            )
            plot_fitness_curve(history, run_dir)

        # --- Periodic response samples ---
        if config.sample_every > 0 and gen % config.sample_every == 0 and gen > start_gen:
            wrapper.set_router_weights(es.state.best_theta)
            evolved_responses = _generate_responses(wrapper, sample_prompts, config.max_new_tokens)
            _save_samples(
                sample_prompts, baseline_responses, evolved_responses,
                gen, es.state.best_fitness, run_dir,
            )
            logger.info(f"  Response samples saved → samples/gen_{gen:04d}.json")

    # ------------------------------------------------------------------ 6. Final outputs
    _save_best_router(es, wrapper.router_shape, run_dir)
    _save_history(history, run_dir)
    plot_fitness_curve(history, run_dir)

    # Final response samples
    wrapper.set_router_weights(es.state.best_theta)
    evolved_responses = _generate_responses(wrapper, sample_prompts, config.max_new_tokens)
    _save_samples(
        sample_prompts, baseline_responses, evolved_responses,
        num_gens, es.state.best_fitness, run_dir,
    )

    improvement = es.state.best_fitness - baseline_fitness
    logger.info(
        f"\n{'=' * 50}\n"
        f"  Baseline fitness: {baseline_fitness:.4f}\n"
        f"  Best fitness    : {es.state.best_fitness:.4f}\n"
        f"  Improvement     : {improvement:+.4f}\n"
        f"  Run directory   : {run_dir}\n"
        f"{'=' * 50}"
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Evolutionary MoE router search")
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to a previous run directory to resume from",
    )
    parser.add_argument(
        "--fitness",
        choices=["reward", "dpo"],
        default=None,
        help="Fitness type (overrides CONFIG for new experiments, ignored on --resume)",
    )
    args = parser.parse_args()

    if args.resume:
        run_dir = Path(args.resume)
        if not (run_dir / "config.json").exists():
            parser.error(f"No config.json found in {run_dir}")
        if not (run_dir / "checkpoint" / "latest.pt").exists():
            parser.error(f"No checkpoint/latest.pt found in {run_dir}")

        config = _load_config(run_dir)
        resume_ckpt = torch.load(run_dir / "checkpoint" / "latest.pt", weights_only=False)
        logger.info(f"Resuming experiment from {run_dir} (fitness={config.fitness_type})")
        run(config, run_dir, resume_ckpt=resume_ckpt)
    else:
        config = CONFIG
        if args.fitness and args.fitness != config.fitness_type:
            config = replace(
                config,
                fitness_type=args.fitness,
                experiment_name=f"olmoe_router_{args.fitness}",
            )
        run_dir = _create_run_dir(config)
        _save_config(config, run_dir)
        logger.info(f"New experiment → {run_dir} (fitness={config.fitness_type})")
        run(config, run_dir)


if __name__ == "__main__":
    main()
