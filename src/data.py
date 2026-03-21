"""
Dataset loading and tokenisation for the evolutionary router search.

We use UltraFeedback binarized (HuggingFaceH4/ultrafeedback_binarized) which
provides (prompt, chosen_response, rejected_response) triples.  Each pair is
tokenised into a full sequence (prompt + response) and we record where the
response starts so the fitness function can compute log-probs over the response
tokens only.
"""

import logging
from dataclasses import dataclass

import torch
from datasets import load_dataset
from transformers import PreTrainedTokenizerBase

from .config import DataConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class PreferenceSequence:
    """One tokenised preference pair ready to be fed to OLMoEWrapper."""

    chosen_ids: torch.Tensor           # [chosen_seq_len]  LongTensor
    rejected_ids: torch.Tensor         # [rejected_seq_len] LongTensor
    # Index of the first response token inside the full sequence
    chosen_response_start: int
    rejected_response_start: int


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_assistant_content(messages: list[dict]) -> str:
    """Return the text of the last assistant turn in a chat message list."""
    for msg in reversed(messages):
        if msg.get("role") == "assistant":
            return msg.get("content", "")
    return ""


def _tokenize_pair(
    tokenizer: PreTrainedTokenizerBase,
    prompt: str,
    response: str,
    max_length: int,
) -> tuple[torch.Tensor, int]:
    """
    Tokenise a prompt+response pair and return:
      - input_ids : LongTensor [seq_len]
      - response_start : index of the first response token

    We apply the tokenizer's chat template when available so that the model
    sees its expected special tokens.
    """
    if tokenizer.chat_template is not None:
        # Full sequence: prompt turn + response turn
        full_text: str = tokenizer.apply_chat_template(
            [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": response},
            ],
            tokenize=False,
            add_generation_prompt=False,
        )
        # Prompt-only: used to find where the response begins
        prompt_text: str = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
    else:
        full_text = f"{prompt}\n\n{response}"
        prompt_text = f"{prompt}\n\n"

    full_ids: torch.Tensor = tokenizer(
        full_text,
        max_length=max_length,
        truncation=True,
        return_tensors="pt",
        add_special_tokens=True,
    ).input_ids.squeeze(0)

    prompt_ids: torch.Tensor = tokenizer(
        prompt_text,
        return_tensors="pt",
        add_special_tokens=True,
    ).input_ids.squeeze(0)

    # Clamp in case the prompt alone is longer than max_length
    response_start = min(len(prompt_ids), len(full_ids) - 1)

    return full_ids, response_start


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_prompts(
    config: DataConfig,
    max_prompt_chars: int = 512,
) -> list[str]:
    """
    Load plain-text prompts from the dataset (no preference pairs needed).

    Used by the reward-model fitness which generates and scores responses
    rather than comparing pre-existing chosen/rejected pairs.

    Args:
        config           : DataConfig — dataset name, split, num_samples.
        max_prompt_chars : truncate prompts longer than this many characters
                           to keep tokenised lengths reasonable.

    Returns:
        List of prompt strings.
    """
    logger.info(f"Loading prompts: {config.dataset_name} ({config.dataset_split})")
    dataset = load_dataset(config.dataset_name, split=config.dataset_split)

    n = min(config.num_samples, len(dataset))
    dataset = dataset.select(range(n))

    prompts: list[str] = []
    for sample in dataset:
        prompt: str = sample.get("prompt", "")
        if not prompt:
            continue
        # Truncate very long prompts
        prompts.append(prompt[:max_prompt_chars])

    logger.info(f"Loaded {len(prompts)} prompts.")
    return prompts


def load_preference_sequences(
    config: DataConfig,
    tokenizer: PreTrainedTokenizerBase,
) -> list[PreferenceSequence]:
    """
    Download (or load from cache) the preference dataset and tokenise the
    first ``config.num_samples`` pairs.

    Returns a list of PreferenceSequence objects, one per sample.
    """
    logger.info(f"Loading dataset: {config.dataset_name} ({config.dataset_split})")
    dataset = load_dataset(config.dataset_name, split=config.dataset_split)

    n = min(config.num_samples, len(dataset))
    dataset = dataset.select(range(n))
    logger.info(f"Using {n} preference pairs.")

    max_len = config.max_prompt_length + config.max_response_length
    sequences: list[PreferenceSequence] = []
    skipped = 0

    for sample in dataset:
        prompt: str = sample.get("prompt", "")
        chosen_text = _extract_assistant_content(sample["chosen"])
        rejected_text = _extract_assistant_content(sample["rejected"])

        if not chosen_text or not rejected_text:
            skipped += 1
            continue

        chosen_ids, chosen_start = _tokenize_pair(
            tokenizer, prompt, chosen_text, max_len
        )
        rejected_ids, rejected_start = _tokenize_pair(
            tokenizer, prompt, rejected_text, max_len
        )

        # Skip sequences where the response is so long it got completely
        # truncated (degenerate case)
        if chosen_start >= len(chosen_ids) or rejected_start >= len(rejected_ids):
            skipped += 1
            continue

        sequences.append(
            PreferenceSequence(
                chosen_ids=chosen_ids,
                rejected_ids=rejected_ids,
                chosen_response_start=chosen_start,
                rejected_response_start=rejected_start,
            )
        )

    if skipped:
        logger.warning(f"Skipped {skipped} malformed samples.")

    logger.info(f"Tokenised {len(sequences)} valid preference pairs.")
    return sequences
