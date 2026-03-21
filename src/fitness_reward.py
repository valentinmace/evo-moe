"""
Reward-model fitness for the evolutionary router search.

How it works
------------
For each candidate router θ in the population:

  1. Set the last MoE layer's router weights to θ.
  2. For each prompt in the evaluation set:
       a. Generate a response with the LLM (using the candidate router).
       b. Score (prompt, response) with the reward model → scalar reward.
  3. fitness(θ) = mean reward over all prompts.

Generation speed-up via prompt KV cache
-----------------------------------------
The KV values at all transformer layers depend only on the *input* to each
layer, never on the layer's own MoE output.  Because we freeze every layer
except the last one's router, all KV values for the prompt tokens are
completely identical across router candidates.

We therefore compute the prompt KV cache ONCE (during __init__) and reuse it
for every generation call, avoiding redundant attention computation over the
prompt for each of the N × pop_size generation calls.

  Without optimisation : pop_size × n_prompts × (prompt_len + gen_len) steps
  With KV-cache reuse  :            n_prompts × (prompt_len − 1) steps
                        + pop_size × n_prompts × (1 + gen_len) steps
  Typical speed-up     : ~4×  (prompt_len=200, gen_len=50)
"""

import logging
import time

import numpy as np
import torch
from tqdm import tqdm
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .model_wrapper import OLMoEWrapper
    from .reward_model import RewardModel

logger = logging.getLogger(__name__)


class RewardFitnessEvaluator:
    """
    Stateful evaluator that caches prompt KV tensors on construction and
    reuses them for every subsequent candidate evaluation.

    Args:
        wrapper         : OLMoEWrapper — used for generation.
        reward_model    : RewardModel  — used for scoring.
        prompts         : list of plain-text instruction prompts.
        max_new_tokens  : maximum response length in tokens.
        rm_batch_size   : number of (prompt, response) pairs scored at once
                          by the reward model (tune to VRAM).
    """

    def __init__(
        self,
        wrapper: "OLMoEWrapper",
        reward_model: "RewardModel",
        prompts: list[str],
        max_new_tokens: int = 50,
        rm_batch_size: int = 4,
    ) -> None:
        self.wrapper = wrapper
        self.reward_model = reward_model
        self.prompts = prompts
        self.max_new_tokens = max_new_tokens
        self.rm_batch_size = rm_batch_size

        # Tokenise prompts and pre-compute their KV caches (one-time cost)
        logger.info(f"Pre-computing prompt KV caches for {len(prompts)} prompts…")
        self._tokenized_prompts, self._prompt_kv_caches = self._prepare_prompt_caches()
        logger.info("Prompt KV caches ready.")

    # ------------------------------------------------------------------
    # One-time setup
    # ------------------------------------------------------------------

    def _tokenize_prompt(self, prompt: str) -> torch.Tensor:
        """Tokenise a single prompt using the LLM's chat template."""
        tokenizer = self.wrapper.tokenizer
        if tokenizer.chat_template is not None:
            text = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            text = prompt

        return tokenizer(
            text,
            return_tensors="pt",
            add_special_tokens=True,
        ).input_ids  # [1, seq_len]

    def _prepare_prompt_caches(
        self,
    ) -> tuple[list[torch.Tensor], list[list[tuple[torch.Tensor, torch.Tensor]]]]:
        tokenized: list[torch.Tensor] = []
        kv_caches: list[list[tuple[torch.Tensor, torch.Tensor]]] = []

        for prompt in tqdm(self.prompts, desc="Caching prompt KVs"):
            ids = self._tokenize_prompt(prompt)
            tokenized.append(ids)
            kv_caches.append(self.wrapper.cache_prompt_kv(ids))

        return tokenized, kv_caches

    # ------------------------------------------------------------------
    # Per-candidate evaluation
    # ------------------------------------------------------------------

    def evaluate(
        self,
        router_weights_flat: np.ndarray,
        progress_bar: tqdm | None = None,
    ) -> float:
        """
        Generate one response per prompt with the given router, score all
        responses with the reward model, and return the mean reward.

        Args:
            router_weights_flat : flat float32 numpy array of router weights.
            progress_bar        : optional outer tqdm bar to update after each
                                  major step (generation, scoring).

        Returns:
            Mean scalar reward over all prompts (higher is better).
        """
        self.wrapper.set_router_weights(router_weights_flat)

        # --- Generation: one response per prompt ---
        t0 = time.perf_counter()
        responses: list[str] = []
        for prompt_ids, kv_cache in zip(self._tokenized_prompts, self._prompt_kv_caches):
            generated_ids = self.wrapper.generate_from_prompt_kv(
                prompt_ids,
                kv_cache,
                max_new_tokens=self.max_new_tokens,
            )
            response_text = self.wrapper.tokenizer.decode(
                generated_ids, skip_special_tokens=True
            )
            responses.append(response_text)
        gen_time = time.perf_counter() - t0

        # --- Reward scoring: batch calls to the reward model ---
        t0 = time.perf_counter()
        pairs = list(zip(self.prompts, responses))
        all_scores: list[float] = []

        for start in range(0, len(pairs), self.rm_batch_size):
            batch = pairs[start : start + self.rm_batch_size]
            all_scores.extend(self.reward_model.score_batch(batch))
        score_time = time.perf_counter() - t0

        reward = float(np.mean(all_scores))

        if progress_bar is not None:
            progress_bar.set_postfix_str(
                f"gen={gen_time:.1f}s  score={score_time:.1f}s  reward={reward:+.2f}"
            )
            progress_bar.update(1)

        return reward
