"""
Reward model wrapper.

We use GRM-Llama3.2-3B (Ray2333/GRM-Llama3.2-3B-rewardmodel-ft), a 3B
sequence-classification model that outputs a scalar quality score for a
(prompt, response) pair.

The model is loaded in 4-bit NF4 quantisation to keep VRAM usage low (~1.5 GB),
leaving headroom for the main OLMoE model (~3.5 GB) on a 12 GB GPU.

Usage
-----
    rm = RewardModel("Ray2333/GRM-Llama3.2-3B-rewardmodel-ft")
    score = rm.score("What is 2+2?", "The answer is 4.")
    scores = rm.score_batch([("What is 2+2?", "4"), ("Hello", "Hi!")])
"""

import logging

import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BitsAndBytesConfig,
    PreTrainedTokenizerBase,
)

logger = logging.getLogger(__name__)

# Maximum tokens fed to the reward model.
# GRM supports up to 4096; we stay conservative to save VRAM.
_MAX_RM_LENGTH = 2048


class RewardModel:
    """
    Thin wrapper around a HuggingFace sequence-classification reward model.

    The reward is read from `model(**inputs).logits`, shape [batch, 1].
    We squeeze it to a scalar.
    """

    def __init__(
        self,
        model_name: str = "Ray2333/GRM-Llama3.2-3B-rewardmodel-ft",
        load_in_4bit: bool = True,
        device: str = "cuda",
    ) -> None:
        self.model_name = model_name
        self.load_in_4bit = load_in_4bit
        self.device = torch.device(device)

        logger.info(f"Loading reward model: {model_name}")
        self.tokenizer, self.model = self._load()
        logger.info("Reward model ready.")

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def _load(self) -> tuple[PreTrainedTokenizerBase, AutoModelForSequenceClassification]:
        bnb_config = (
            BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
            if self.load_in_4bit
            else None
        )

        model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            quantization_config=bnb_config,
            device_map={"": 0},
            torch_dtype=torch.float16 if not self.load_in_4bit else None,
            num_labels=1,
        )
        model.eval()

        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        return tokenizer, model

    # ------------------------------------------------------------------
    # Formatting
    # ------------------------------------------------------------------

    def _format_input(self, prompt: str, response: str) -> str:
        """Format a (prompt, response) pair using the model's chat template."""
        if self.tokenizer.chat_template is not None:
            return self.tokenizer.apply_chat_template(
                [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": response},
                ],
                tokenize=False,
                add_generation_prompt=False,
            )
        # Fallback plain format
        return f"User: {prompt}\n\nAssistant: {response}"

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    @torch.no_grad()
    def score(self, prompt: str, response: str) -> float:
        """
        Return the scalar reward for a single (prompt, response) pair.

        Higher is better.
        """
        text = self._format_input(prompt, response)
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=_MAX_RM_LENGTH,
        ).to(self.device)

        outputs = self.model(**inputs)
        return outputs.logits.squeeze().item()

    @torch.no_grad()
    def score_batch(self, pairs: list[tuple[str, str]]) -> list[float]:
        """
        Score a list of (prompt, response) pairs in one batched forward pass.

        Use small batch sizes (4–8) to avoid OOM.
        """
        texts = [self._format_input(p, r) for p, r in pairs]
        inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            truncation=True,
            max_length=_MAX_RM_LENGTH,
            padding=True,
        ).to(self.device)

        outputs = self.model(**inputs)
        return outputs.logits.squeeze(-1).tolist()
