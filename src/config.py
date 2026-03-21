from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass
class ModelConfig:
    """Configuration for the language model."""

    model_name: str = "allenai/OLMoE-1B-7B-0924-Instruct"
    load_in_4bit: bool = True
    device: str = "cuda"
    # Index of the MoE layer whose router we evolve (-1 = last layer)
    target_layer_idx: int = -1


@dataclass
class DataConfig:
    """Configuration for the preference dataset."""

    dataset_name: str = "HuggingFaceH4/ultrafeedback_binarized"
    dataset_split: str = "train_prefs"
    # Number of preference pairs to use for fitness evaluation
    num_samples: int = 256
    max_prompt_length: int = 256
    max_response_length: int = 128


@dataclass
class EvolutionConfig:
    """Configuration for the OpenAI-ES optimiser."""

    population_size: int = 50
    # Must be even when antithetic_sampling=True
    sigma: float = 0.01
    learning_rate: float = 0.01
    num_generations: int = 200
    # Antithetic sampling halves variance at no extra cost
    antithetic_sampling: bool = True
    # Mini-batch size when computing fitness (limits GPU memory)
    fitness_batch_size: int = 16
    seed: int = 42


@dataclass
class ExperimentConfig:
    """Top-level experiment configuration."""

    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    evolution: EvolutionConfig = field(default_factory=EvolutionConfig)
    output_dir: str = "outputs"
    experiment_name: str = "router_evolution"
    # "reward" = generate + reward-model scoring (slow, direct signal)
    # "dpo"    = activation-cache log-prob difference (fast proxy)
    fitness_type: str = "reward"
    # KL penalty strength for DPO fitness (0 = disabled).
    # Penalises router candidates whose log-probs deviate from the reference.
    dpo_beta: float = 0.1
    reward_model_name: str = "Ray2333/GRM-Llama3.2-3B-rewardmodel-ft"
    max_new_tokens: int = 50
    log_every: int = 10
    # Generate sample responses every N generations (0 = disabled)
    sample_every: int = 10
    num_sample_prompts: int = 5

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> ExperimentConfig:
        return cls(
            model=ModelConfig(**d["model"]),
            data=DataConfig(**d["data"]),
            evolution=EvolutionConfig(**d["evolution"]),
            output_dir=d["output_dir"],
            experiment_name=d["experiment_name"],
            fitness_type=d.get("fitness_type", "reward"),
            dpo_beta=d.get("dpo_beta", 0.1),
            reward_model_name=d["reward_model_name"],
            max_new_tokens=d["max_new_tokens"],
            log_every=d["log_every"],
            sample_every=d["sample_every"],
            num_sample_prompts=d["num_sample_prompts"],
        )
