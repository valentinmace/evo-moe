"""
DPO-style fitness for the evolutionary router search.

This is an *activation-cache-based* fitness that does NOT require generating
text or running a reward model.  It is very fast and works as follows:

    fitness_DPO(θ) = E_{(chosen, rejected) ~ D} [
        mean_log_prob_θ(chosen) − mean_log_prob_θ(rejected)
    ]

Intuitively: a better router should assign higher log-probability to the
preferred (chosen) response than to the dispreferred (rejected) one.

Because only the last MoE layer's router changes, we pre-cache the post-
attention activations entering that layer (see model_wrapper.py).  Each
candidate evaluation then only runs the cheap MoE tail — no attention, no
early layers.

This fitness is useful as a fast proxy signal or for ablations.  The reward-
model fitness in fitness_reward.py is slower but more direct.
"""

import logging
import time

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .model_wrapper import ActivationCache, OLMoEWrapper

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mean_response_log_prob(
    logits: torch.Tensor,        # [seq_len, vocab_size]
    token_ids: torch.Tensor,     # [seq_len]
    response_mask: torch.Tensor, # [seq_len] bool — True at response positions
) -> float:
    """
    Mean per-token log-probability over the response tokens.

    logits[t] predicts token token_ids[t+1], so we shift everything by one
    before applying the response mask.
    """
    shifted_logits = logits[:-1]                       # [seq_len-1, vocab_size]
    target_ids     = token_ids[1:].to(logits.device)   # [seq_len-1]
    shifted_mask   = response_mask[1:]                  # [seq_len-1]

    log_probs        = F.log_softmax(shifted_logits, dim=-1)
    token_log_probs  = log_probs[torch.arange(len(target_ids)), target_ids]
    response_lp      = token_log_probs[shifted_mask]

    if response_lp.numel() == 0:
        return 0.0
    return response_lp.mean().item()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_router_dpo(
    router_weights_flat: np.ndarray,
    wrapper: "OLMoEWrapper",
    cache: "ActivationCache",
    batch_size: int = 16,
) -> float:
    """
    Evaluate a router candidate using the DPO implicit reward.

    Args:
        router_weights_flat : flat float32 numpy array (router weights).
        wrapper             : OLMoEWrapper — router is set temporarily.
        cache               : ActivationCache built by wrapper.cache_activations().
        batch_size          : sequences per GPU batch (tune to VRAM).

    Returns:
        Scalar fitness (higher = better router).
    """
    wrapper.set_router_weights(router_weights_flat)

    n = len(cache.chosen_activations)
    scores: list[float] = []

    for start in range(0, n, batch_size):
        sl = slice(start, start + batch_size)
        for j in range(len(cache.chosen_activations[sl])):
            idx = start + j
            chosen_logits   = wrapper.compute_logits_from_cache(cache.chosen_activations[idx])
            rejected_logits = wrapper.compute_logits_from_cache(cache.rejected_activations[idx])

            chosen_lp   = _mean_response_log_prob(
                chosen_logits, cache.chosen_input_ids[idx], cache.chosen_response_masks[idx]
            )
            rejected_lp = _mean_response_log_prob(
                rejected_logits, cache.rejected_input_ids[idx], cache.rejected_response_masks[idx]
            )
            scores.append(chosen_lp - rejected_lp)

    return float(np.mean(scores))


class DPOFitnessEvaluator:
    """
    Stateful evaluator that wraps the activation-cache-based DPO fitness
    with the same .evaluate() interface as RewardFitnessEvaluator.

    Includes a KL penalty that prevents the evolved router from drifting
    too far from the reference (pre-trained) router.  The penalty is:

        penalty_i = 0.5 * ( |chosen_lp_i − ref_chosen_lp_i|
                           + |rejected_lp_i − ref_rejected_lp_i| )

        fitness = mean( dpo_score_i − β × penalty_i )

    Reference log-probs are computed once in __init__ and reused for all
    candidates, so the overhead is negligible.

    Args:
        wrapper   : OLMoEWrapper — router is set per candidate.
        cache     : ActivationCache built by wrapper.cache_activations().
        batch_size: sequences per GPU batch (tune to VRAM).
        beta      : KL penalty strength (0 = no penalty).
    """

    def __init__(
        self,
        wrapper: "OLMoEWrapper",
        cache: "ActivationCache",
        batch_size: int = 16,
        beta: float = 0.1,
    ) -> None:
        self.wrapper = wrapper
        self.cache = cache
        self.batch_size = batch_size
        self.beta = beta

        self._ref_chosen_lps, self._ref_rejected_lps = (
            self._compute_reference_log_probs()
        )

        n_pairs = len(cache.chosen_activations)
        logger.info(
            f"DPOFitnessEvaluator ready — {n_pairs} preference pairs, β={beta}"
        )

    # ------------------------------------------------------------------
    # One-time reference computation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _compute_reference_log_probs(
        self,
    ) -> tuple[list[float], list[float]]:
        """Compute per-sequence mean log-probs under the reference router."""
        self.wrapper.reset_router()

        ref_chosen: list[float] = []
        ref_rejected: list[float] = []

        n = len(self.cache.chosen_activations)
        for idx in range(n):
            chosen_logits = self.wrapper.compute_logits_from_cache(
                self.cache.chosen_activations[idx]
            )
            rejected_logits = self.wrapper.compute_logits_from_cache(
                self.cache.rejected_activations[idx]
            )
            ref_chosen.append(_mean_response_log_prob(
                chosen_logits,
                self.cache.chosen_input_ids[idx],
                self.cache.chosen_response_masks[idx],
            ))
            ref_rejected.append(_mean_response_log_prob(
                rejected_logits,
                self.cache.rejected_input_ids[idx],
                self.cache.rejected_response_masks[idx],
            ))

        logger.info(
            f"Reference log-probs — "
            f"chosen: {np.mean(ref_chosen):.4f}, "
            f"rejected: {np.mean(ref_rejected):.4f}"
        )
        return ref_chosen, ref_rejected

    # ------------------------------------------------------------------
    # Per-candidate evaluation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def evaluate(
        self,
        router_weights_flat: np.ndarray,
        progress_bar: tqdm | None = None,
    ) -> float:
        t0 = time.perf_counter()

        self.wrapper.set_router_weights(router_weights_flat)

        n = len(self.cache.chosen_activations)
        scores: list[float] = []

        for start in range(0, n, self.batch_size):
            sl = slice(start, start + self.batch_size)
            for j in range(len(self.cache.chosen_activations[sl])):
                idx = start + j
                chosen_logits = self.wrapper.compute_logits_from_cache(
                    self.cache.chosen_activations[idx]
                )
                rejected_logits = self.wrapper.compute_logits_from_cache(
                    self.cache.rejected_activations[idx]
                )

                chosen_lp = _mean_response_log_prob(
                    chosen_logits,
                    self.cache.chosen_input_ids[idx],
                    self.cache.chosen_response_masks[idx],
                )
                rejected_lp = _mean_response_log_prob(
                    rejected_logits,
                    self.cache.rejected_input_ids[idx],
                    self.cache.rejected_response_masks[idx],
                )

                dpo_score = chosen_lp - rejected_lp
                kl_penalty = 0.5 * (
                    abs(chosen_lp - self._ref_chosen_lps[idx])
                    + abs(rejected_lp - self._ref_rejected_lps[idx])
                )
                scores.append(dpo_score - self.beta * kl_penalty)

        score = float(np.mean(scores))
        elapsed = time.perf_counter() - t0

        if progress_bar is not None:
            progress_bar.set_postfix_str(
                f"dpo={elapsed:.2f}s  fitness={score:+.4f}"
            )
            progress_bar.update(1)

        return score
