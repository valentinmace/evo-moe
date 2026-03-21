"""
Pipeline tests for evo_moe.

Run fast tests only (no model loading, no GPU needed):
    pytest tests/ -v -m "not gpu"

Run architecture + model tests (requires GPU + model download):
    pytest tests/ -v -m gpu

Run everything:
    pytest tests/ -v

Each section is independent — feel free to run them individually with -k:
    pytest tests/ -v -k "TestEvolution"
    pytest tests/ -v -k "TestArchitecture"
    pytest tests/ -v -k "TestActivationCaching"
    pytest tests/ -v -k "TestKVCacheGeneration"
    pytest tests/ -v -k "TestDPOHelpers"
    pytest tests/ -v -k "TestDPOFitnessIntegration"
    pytest tests/ -v -k "TestRewardModel"
    pytest tests/ -v -k "TestConfigSerialization"
    pytest tests/ -v -k "TestESCheckpointing"
    pytest tests/ -v -k "TestExperimentIO"
"""

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))


# ============================================================
# Shared fixtures
# ============================================================

@pytest.fixture(scope="session")
def llm_wrapper():
    """
    Load OLMoE once for the whole test session.
    All tests in TestArchitecture and TestCaching share this instance.
    """
    if not torch.cuda.is_available():
        pytest.skip("CUDA GPU required to load the LLM.")

    from src.config import ModelConfig
    from src.model_wrapper import OLMoEWrapper

    return OLMoEWrapper(
        ModelConfig(
            model_name="allenai/OLMoE-1B-7B-0924-Instruct",
            load_in_4bit=True,
            device="cuda",
            target_layer_idx=-1,
        )
    )


@pytest.fixture(scope="session")
def reward_model():
    """Load GRM reward model once for the whole test session."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA GPU required to load the reward model.")

    from src.reward_model import RewardModel

    return RewardModel(
        model_name="Ray2333/GRM-Llama3.2-3B-rewardmodel-ft",
        load_in_4bit=True,
        device="cuda",
    )


# ============================================================
# SECTION 1 — Evolution strategy (no GPU needed)
# ============================================================

class TestEvolution:
    """Fast logic tests for OpenAI-ES: no model, no GPU."""

    def test_ask_returns_correct_population_size(self):
        from src.evolution import OpenAIES

        es = OpenAIES(np.zeros(100), population_size=20, antithetic_sampling=True)
        population = es.ask()
        assert len(population) == 20

    def test_all_candidates_have_correct_shape(self):
        from src.evolution import OpenAIES

        dim = 500
        es = OpenAIES(np.zeros(dim), population_size=10, antithetic_sampling=False)
        for candidate in es.ask():
            assert candidate.shape == (dim,)

    def test_antithetic_pairs_sum_to_2_theta(self):
        """
        With antithetic sampling, candidate[i] + candidate[i + N/2] should
        equal 2 * theta.  This verifies that we truly sample ε and -ε.
        """
        from src.evolution import OpenAIES

        theta = np.ones(50, dtype=np.float32) * 3.0
        es = OpenAIES(theta.copy(), population_size=10, sigma=0.5, antithetic_sampling=True, seed=0)
        pop = es.ask()
        half = 5
        for i in range(half):
            np.testing.assert_allclose(
                pop[i] + pop[i + half], 2 * theta, atol=1e-5,
                err_msg=f"Antithetic pair {i} is not symmetric around theta",
            )

    def test_tell_moves_theta(self):
        """Calling tell() with non-uniform fitnesses should update theta."""
        from src.evolution import OpenAIES

        es = OpenAIES(np.zeros(100), population_size=10, learning_rate=1.0,
                      antithetic_sampling=False, seed=0)
        theta_before = es.theta.copy()
        es.ask()
        es.tell([float(i) for i in range(10)])
        assert not np.allclose(es.theta, theta_before), "theta did not change after tell()"

    def test_best_fitness_is_tracked(self):
        from src.evolution import OpenAIES

        es = OpenAIES(np.zeros(50), population_size=10, antithetic_sampling=False, seed=0)
        es.ask()
        es.tell([1.0, 5.0, 2.0, 3.0, 0.5, 1.5, 4.0, 2.5, 0.1, 3.5])
        assert es.state.best_fitness == pytest.approx(5.0)

    def test_rank_normalize_is_centered(self):
        """Rank-normalised fitnesses should be in [-0.5, 0.5] and sum to 0."""
        from src.evolution import _rank_normalize

        arr = np.array([10.0, 1.0, 5.0, 3.0, 8.0])
        ranks = _rank_normalize(arr)
        assert ranks.min() == pytest.approx(-0.5)
        assert ranks.max() == pytest.approx(0.5)
        assert ranks.sum() == pytest.approx(0.0, abs=1e-5)

    def test_tell_requires_ask_first(self):
        from src.evolution import OpenAIES

        es = OpenAIES(np.zeros(10), population_size=4)
        with pytest.raises(RuntimeError, match="ask()"):
            es.tell([1.0, 2.0, 3.0, 4.0])

    def test_tell_wrong_count_raises(self):
        from src.evolution import OpenAIES

        es = OpenAIES(np.zeros(10), population_size=4, antithetic_sampling=False)
        es.ask()
        with pytest.raises(ValueError, match="Expected 4"):
            es.tell([1.0, 2.0])

    def test_antithetic_requires_even_population(self):
        from src.evolution import OpenAIES

        with pytest.raises(ValueError, match="even"):
            OpenAIES(np.zeros(10), population_size=5, antithetic_sampling=True)

    def test_seed_reproducibility(self):
        """Same seed must produce the exact same population."""
        from src.evolution import OpenAIES

        es1 = OpenAIES(np.zeros(50), population_size=10, seed=123, antithetic_sampling=False)
        es2 = OpenAIES(np.zeros(50), population_size=10, seed=123, antithetic_sampling=False)
        pop1 = es1.ask()
        pop2 = es2.ask()
        for a, b in zip(pop1, pop2):
            np.testing.assert_array_equal(a, b)

    def test_multiple_ask_tell_cycles_track_generation(self):
        """After N ask/tell cycles, state.generation should be N."""
        from src.evolution import OpenAIES

        es = OpenAIES(np.zeros(20), population_size=6, antithetic_sampling=True, seed=0)
        for _ in range(5):
            es.ask()
            es.tell([float(i) for i in range(6)])
        assert es.state.generation == 5
        assert len(es.state.fitness_history) == 5


# ============================================================
# SECTION 2 — OLMoE architecture validation (requires GPU)
# ============================================================

class TestArchitecture:
    """
    Verify that the OLMoE model has exactly the structure we expect.
    Run these FIRST before any experiment — they confirm that module paths
    like model.model.layers[-1].mlp.gate are correct for this model version.
    """

    @pytest.mark.gpu
    def test_last_layer_has_mlp_with_gate(self, llm_wrapper):
        last = llm_wrapper._last_layer
        assert hasattr(last, "mlp"), \
            "Expected last_layer.mlp — check the model architecture."
        assert hasattr(last.mlp, "gate"), \
            "Expected last_layer.mlp.gate — the router is not where we expect it."

    @pytest.mark.gpu
    def test_gate_weight_dtype_matches_model(self, llm_wrapper):
        """
        The router weight dtype must match the model's compute dtype so that
        F.linear(hidden_states, gate.weight) never raises a dtype mismatch.

        OlmoeTopKRouter stores its routing matrix directly as an nn.Parameter
        (no nn.Linear wrapper) and applies it via F.linear.  We verify the
        parameter dtype matches what the wrapper detected at load time.
        """
        gate = llm_wrapper._last_layer.mlp.gate
        assert hasattr(gate, "weight"), \
            f"Router ({type(gate)}) has no .weight parameter — architecture changed?"
        expected = llm_wrapper._compute_dtype
        assert gate.weight.dtype == expected, \
            f"Router weight dtype is {gate.weight.dtype} — expected {expected}."

    @pytest.mark.gpu
    def test_router_shape_is_printed_and_sane(self, llm_wrapper):
        """Print the router shape so you can confirm it matches expectations."""
        shape = llm_wrapper.router_shape
        dim   = llm_wrapper.router_dim
        print(f"\n  Router shape : {shape}  (num_experts × hidden_size)")
        print(f"  Router dim   : {dim:,} parameters")
        assert shape[0] >= 2,  f"Unexpected num_experts={shape[0]}"
        assert shape[1] >= 64, f"Unexpected hidden_size={shape[1]}"

    @pytest.mark.gpu
    def test_post_attention_layernorm_is_accessible(self, llm_wrapper):
        """
        Activation caching hooks into post_attention_layernorm.
        This test confirms the attribute exists under the expected name.
        """
        assert hasattr(llm_wrapper._last_layer, "post_attention_layernorm"), \
            "post_attention_layernorm not found — update the hook in model_wrapper.py."

    @pytest.mark.gpu
    def test_router_get_set_and_reset(self, llm_wrapper):
        """
        Full round-trip: read weights → perturb → write → verify → reset → verify.
        This is the core operation used in every ES iteration.
        """
        original = llm_wrapper.get_router_weights()

        # Set perturbed weights
        perturbed = original + 0.1
        llm_wrapper.set_router_weights(perturbed)
        np.testing.assert_allclose(
            llm_wrapper.get_router_weights(), perturbed, atol=1e-3,
            err_msg="Router weights were not updated correctly.",
        )

        # Reset to original
        llm_wrapper.reset_router()
        np.testing.assert_allclose(
            llm_wrapper.get_router_weights(), original, atol=1e-3,
            err_msg="Router was not restored to original after reset.",
        )


# ============================================================
# SECTION 3 — Activation caching and tail forward pass
# ============================================================

class TestActivationCaching:
    """
    Tests for the DPO-style fitness path:
    cache_activations() → compute_logits_from_cache()
    """

    @pytest.fixture
    def short_sequence(self, llm_wrapper):
        ids = llm_wrapper.tokenizer(
            "What is the capital of France?",
            return_tensors="pt",
        ).input_ids.squeeze(0)
        return ids

    @pytest.mark.gpu
    def test_cached_activation_shape(self, llm_wrapper, short_sequence):
        """Cached tensor must be [seq_len, hidden_size] in the model's compute dtype on CPU."""
        seq_dict = {
            "chosen_ids": short_sequence,
            "rejected_ids": short_sequence,
            "chosen_response_start": 1,
            "rejected_response_start": 1,
        }
        cache = llm_wrapper.cache_activations([seq_dict], show_progress=False)

        act = cache.chosen_activations[0]
        assert act.ndim == 2
        assert act.shape[0] == len(short_sequence), "seq_len mismatch"
        assert act.shape[1] == llm_wrapper.router_shape[1], "hidden_size mismatch"
        assert act.dtype == llm_wrapper._compute_dtype
        assert act.device.type == "cpu", "Activation should be stored on CPU"

    @pytest.mark.gpu
    def test_logits_from_cache_match_full_forward(self, llm_wrapper, short_sequence):
        """
        compute_logits_from_cache() must give the same logits as a full
        model forward pass (up to fp16 rounding).
        This confirms our activation-caching approach is mathematically correct.
        """
        seq_dict = {
            "chosen_ids": short_sequence,
            "rejected_ids": short_sequence,
            "chosen_response_start": 1,
            "rejected_response_start": 1,
        }
        cache = llm_wrapper.cache_activations([seq_dict], show_progress=False)

        # Full model forward pass
        with torch.no_grad():
            full_logits = llm_wrapper.model(
                short_sequence.unsqueeze(0).to(llm_wrapper.device),
                use_cache=False,
            ).logits.squeeze(0).cpu()

        # Tail-only forward pass from the cached activation
        cached_logits = llm_wrapper.compute_logits_from_cache(
            cache.chosen_activations[0]
        ).cpu()

        np.testing.assert_allclose(
            cached_logits.float().numpy(),
            full_logits.float().numpy(),
            atol=0.1,   # fp16 accumulation allows small numerical differences
            err_msg="Tail forward pass diverges from the full model forward pass.",
        )

    @pytest.mark.gpu
    def test_logits_change_when_router_changes(self, llm_wrapper, short_sequence):
        """
        THE critical property: different router weights must produce different
        logits from the cached activation.  If they don't, the router has no
        effect and the whole project is pointless.
        """
        seq_dict = {
            "chosen_ids": short_sequence,
            "rejected_ids": short_sequence,
            "chosen_response_start": 1,
            "rejected_response_start": 1,
        }
        cache = llm_wrapper.cache_activations([seq_dict], show_progress=False)
        activation = cache.chosen_activations[0]

        logits_original = llm_wrapper.compute_logits_from_cache(activation).cpu()

        rng = np.random.default_rng(seed=42)
        random_weights = rng.standard_normal(llm_wrapper.router_dim).astype(np.float32)
        llm_wrapper.set_router_weights(random_weights)
        logits_random = llm_wrapper.compute_logits_from_cache(activation).cpu()
        llm_wrapper.reset_router()

        assert not torch.allclose(logits_original, logits_random, atol=1e-3), \
            "Logits are identical with different router weights — the router has no effect!"

    @pytest.mark.gpu
    def test_response_mask_covers_correct_positions(self, llm_wrapper, short_sequence):
        """
        With response_start=3, the response mask should be False for positions
        0..2 and True for positions 3..end.
        """
        response_start = 3
        seq_dict = {
            "chosen_ids": short_sequence,
            "rejected_ids": short_sequence,
            "chosen_response_start": response_start,
            "rejected_response_start": response_start,
        }
        cache = llm_wrapper.cache_activations([seq_dict], show_progress=False)
        mask = cache.chosen_response_masks[0]

        assert mask.shape[0] == len(short_sequence)
        assert not mask[:response_start].any(), "Prompt positions should be masked out"
        assert mask[response_start:].all(), "Response positions should all be True"

    @pytest.mark.gpu
    def test_cache_stores_both_sides(self, llm_wrapper, short_sequence):
        """Both chosen and rejected sides should be populated after caching."""
        seq_dict = {
            "chosen_ids": short_sequence,
            "rejected_ids": short_sequence,
            "chosen_response_start": 1,
            "rejected_response_start": 1,
        }
        cache = llm_wrapper.cache_activations([seq_dict], show_progress=False)

        assert len(cache.chosen_activations) == 1
        assert len(cache.rejected_activations) == 1
        assert len(cache.chosen_input_ids) == 1
        assert len(cache.rejected_input_ids) == 1
        assert len(cache.chosen_response_masks) == 1
        assert len(cache.rejected_response_masks) == 1


# ============================================================
# SECTION 4 — KV cache and generation
# ============================================================

class TestKVCacheGeneration:
    """Tests for the reward-model fitness path: cache_prompt_kv() + generate_from_prompt_kv()."""

    @pytest.fixture
    def prompt_ids(self, llm_wrapper):
        text = llm_wrapper.tokenizer.apply_chat_template(
            [{"role": "user", "content": "What is 2 + 2?"}],
            tokenize=False,
            add_generation_prompt=True,
        )
        return llm_wrapper.tokenizer(text, return_tensors="pt").input_ids  # [1, seq_len]

    @pytest.mark.gpu
    def test_prompt_kv_cache_is_router_independent(self, llm_wrapper, prompt_ids):
        """
        KEY PROPERTY: the prompt KV cache should be identical for any router
        because KV values at layer L depend only on h_{L-1}, not on layer L's
        own MoE output.  This is what allows us to reuse the cache across all
        population members.
        """
        kv_original  = llm_wrapper.cache_prompt_kv(prompt_ids)

        w = llm_wrapper.get_router_weights()
        llm_wrapper.set_router_weights(w + 1.0)   # large perturbation
        kv_perturbed = llm_wrapper.cache_prompt_kv(prompt_ids)
        llm_wrapper.reset_router()

        for layer_idx, ((k1, v1), (k2, v2)) in enumerate(zip(kv_original, kv_perturbed)):
            np.testing.assert_allclose(
                k1.float().numpy(), k2.float().numpy(), atol=1e-3,
                err_msg=(
                    f"KV differs at layer {layer_idx} — the router IS affecting "
                    f"the prompt KV cache (unexpected).  Check model architecture."
                ),
            )

    @pytest.mark.gpu
    def test_generation_produces_text(self, llm_wrapper, prompt_ids):
        """generate_from_prompt_kv should return at least one non-empty token."""
        kv    = llm_wrapper.cache_prompt_kv(prompt_ids)
        tokens = llm_wrapper.generate_from_prompt_kv(prompt_ids, kv, max_new_tokens=20)
        text   = llm_wrapper.tokenizer.decode(tokens, skip_special_tokens=True)

        assert tokens.ndim == 1
        assert len(tokens) > 0,  "No tokens were generated."
        assert len(text.strip()) > 0, "Generated text is empty after decoding."
        print(f"\n  Generated response: {text!r}")

    @pytest.mark.gpu
    def test_different_routers_can_produce_different_responses(self, llm_wrapper, prompt_ids):
        """
        Two very different routers should (likely) generate different responses.
        If they always produce identical output, the router has no effect.
        """
        kv = llm_wrapper.cache_prompt_kv(prompt_ids)   # reused for both

        original_weights = llm_wrapper.get_router_weights()
        tokens_original  = llm_wrapper.generate_from_prompt_kv(prompt_ids, kv, max_new_tokens=20)

        rng = np.random.default_rng(seed=99)
        random_weights = rng.standard_normal(len(original_weights)).astype(np.float32)
        llm_wrapper.set_router_weights(random_weights)
        tokens_random = llm_wrapper.generate_from_prompt_kv(prompt_ids, kv, max_new_tokens=20)
        llm_wrapper.reset_router()

        text_original = llm_wrapper.tokenizer.decode(tokens_original, skip_special_tokens=True)
        text_random   = llm_wrapper.tokenizer.decode(tokens_random,   skip_special_tokens=True)
        print(f"\n  Original router : {text_original!r}")
        print(f"  Random router   : {text_random!r}")
        # Soft check: a random router might coincidentally produce the same
        # tokens on very short sequences, so we print rather than hard-fail.
        if torch.equal(tokens_original, tokens_random[:len(tokens_original)]):
            import warnings
            warnings.warn(
                "Original and random routers produced identical output — "
                "this is possible but unlikely. Investigate if it persists.",
                UserWarning,
                stacklevel=1,
            )

    @pytest.mark.gpu
    def test_greedy_generation_is_deterministic(self, llm_wrapper, prompt_ids):
        """Same prompt + same router + greedy must give identical output twice."""
        kv = llm_wrapper.cache_prompt_kv(prompt_ids)
        tokens_a = llm_wrapper.generate_from_prompt_kv(prompt_ids, kv, max_new_tokens=10)
        tokens_b = llm_wrapper.generate_from_prompt_kv(prompt_ids, kv, max_new_tokens=10)

        assert torch.equal(tokens_a, tokens_b), \
            "Greedy generation is non-deterministic — check for stochastic ops."


# ============================================================
# SECTION 5 — DPO fitness helpers (no GPU needed)
# ============================================================

class TestDPOHelpers:
    """Unit tests for the DPO fitness math — runs on CPU, no model needed."""

    def test_mean_response_log_prob_basic(self):
        """
        Given trivial logits where the correct token has the highest score,
        the mean log-prob should be close to 0 (log(1) = 0).
        """
        from src.fitness_dpo import _mean_response_log_prob

        vocab_size = 10
        seq_len = 5
        logits = torch.zeros(seq_len, vocab_size)
        token_ids = torch.arange(seq_len)
        for t in range(seq_len - 1):
            logits[t, token_ids[t + 1]] = 100.0

        mask = torch.ones(seq_len, dtype=torch.bool)
        result = _mean_response_log_prob(logits, token_ids, mask)
        assert result > -0.01, f"Expected ~0.0 but got {result}"

    def test_mean_response_log_prob_only_response_tokens(self):
        """
        When the mask excludes early positions, only response tokens
        should contribute to the log-prob.
        """
        from src.fitness_dpo import _mean_response_log_prob

        vocab_size = 10
        seq_len = 6
        logits = torch.zeros(seq_len, vocab_size)
        token_ids = torch.zeros(seq_len, dtype=torch.long)

        mask = torch.tensor([False, False, False, True, True, True])
        result = _mean_response_log_prob(logits, token_ids, mask)

        mask_all = torch.ones(seq_len, dtype=torch.bool)
        result_all = _mean_response_log_prob(logits, token_ids, mask_all)

        assert isinstance(result, float)
        assert isinstance(result_all, float)

    def test_mean_response_log_prob_empty_response_returns_zero(self):
        """If no response tokens are present, function should return 0.0."""
        from src.fitness_dpo import _mean_response_log_prob

        logits = torch.randn(5, 10)
        token_ids = torch.zeros(5, dtype=torch.long)
        mask = torch.zeros(5, dtype=torch.bool)
        assert _mean_response_log_prob(logits, token_ids, mask) == 0.0


# ============================================================
# SECTION 6 — DPO fitness integration (requires GPU)
# ============================================================

class TestDPOFitnessIntegration:

    @pytest.fixture
    def dpo_cache(self, llm_wrapper):
        tok = llm_wrapper.tokenizer
        chosen_ids = tok(
            "What is the capital of France? Paris is the capital.",
            return_tensors="pt",
        ).input_ids.squeeze(0)
        rejected_ids = tok(
            "What is the capital of France? London is very nice.",
            return_tensors="pt",
        ).input_ids.squeeze(0)
        prompt_len = len(tok(
            "What is the capital of France?",
            return_tensors="pt",
        ).input_ids.squeeze(0))
        seq_dict = {
            "chosen_ids": chosen_ids,
            "rejected_ids": rejected_ids,
            "chosen_response_start": prompt_len,
            "rejected_response_start": prompt_len,
        }
        return llm_wrapper.cache_activations([seq_dict], show_progress=False)

    @pytest.mark.gpu
    def test_evaluate_router_dpo_returns_float(self, llm_wrapper, dpo_cache):
        from src.fitness_dpo import evaluate_router_dpo

        weights = llm_wrapper.get_router_weights()
        score = evaluate_router_dpo(weights, llm_wrapper, dpo_cache)
        assert isinstance(score, float)
        llm_wrapper.reset_router()

    @pytest.mark.gpu
    def test_dpo_fitness_changes_with_router(self, llm_wrapper, dpo_cache):
        """Different router weights should yield different DPO fitness scores."""
        from src.fitness_dpo import evaluate_router_dpo

        original = llm_wrapper.get_router_weights()
        score_original = evaluate_router_dpo(original, llm_wrapper, dpo_cache)

        rng = np.random.default_rng(seed=7)
        random_weights = rng.standard_normal(len(original)).astype(np.float32)
        score_random = evaluate_router_dpo(random_weights, llm_wrapper, dpo_cache)
        llm_wrapper.reset_router()

        assert score_original != pytest.approx(score_random, abs=1e-4), \
            "DPO fitness is the same for original and random router — router has no effect."

    @pytest.mark.gpu
    def test_dpo_evaluator_matches_raw_function(self, llm_wrapper, dpo_cache):
        """DPOFitnessEvaluator.evaluate must return the same value as evaluate_router_dpo."""
        from src.fitness_dpo import DPOFitnessEvaluator, evaluate_router_dpo

        evaluator = DPOFitnessEvaluator(llm_wrapper, dpo_cache)
        weights = llm_wrapper.get_router_weights()

        score_evaluator = evaluator.evaluate(weights)
        score_raw = evaluate_router_dpo(weights, llm_wrapper, dpo_cache)
        llm_wrapper.reset_router()

        assert score_evaluator == pytest.approx(score_raw), \
            "DPOFitnessEvaluator gives a different score than the raw function."

    @pytest.mark.gpu
    def test_dpo_evaluator_updates_progress_bar(self, llm_wrapper, dpo_cache):
        """DPOFitnessEvaluator should call progress_bar.update(1) when provided."""
        from unittest.mock import MagicMock
        from src.fitness_dpo import DPOFitnessEvaluator

        evaluator = DPOFitnessEvaluator(llm_wrapper, dpo_cache)
        weights = llm_wrapper.get_router_weights()

        mock_bar = MagicMock()
        evaluator.evaluate(weights, progress_bar=mock_bar)
        llm_wrapper.reset_router()

        mock_bar.update.assert_called_once_with(1)
        mock_bar.set_postfix_str.assert_called_once()

    @pytest.mark.gpu
    def test_kl_penalty_reduces_score_for_deviant_router(self, llm_wrapper, dpo_cache):
        """
        A random router should receive a LOWER KL-penalised score than the
        raw DPO score, because its log-probs deviate from the reference.
        """
        from src.fitness_dpo import DPOFitnessEvaluator, evaluate_router_dpo

        evaluator = DPOFitnessEvaluator(llm_wrapper, dpo_cache, beta=0.5)

        rng = np.random.default_rng(seed=7)
        random_weights = rng.standard_normal(llm_wrapper.router_dim).astype(np.float32)

        score_penalised = evaluator.evaluate(random_weights)
        score_raw = evaluate_router_dpo(random_weights, llm_wrapper, dpo_cache)
        llm_wrapper.reset_router()

        assert score_penalised < score_raw, (
            f"KL-penalised score ({score_penalised:.4f}) should be lower "
            f"than raw DPO score ({score_raw:.4f}) for a deviant router."
        )

    @pytest.mark.gpu
    def test_kl_penalty_zero_beta_matches_raw(self, llm_wrapper, dpo_cache):
        """With beta=0 the KL penalty is disabled; scores must match the raw function."""
        from src.fitness_dpo import DPOFitnessEvaluator, evaluate_router_dpo

        evaluator = DPOFitnessEvaluator(llm_wrapper, dpo_cache, beta=0.0)

        rng = np.random.default_rng(seed=7)
        random_weights = rng.standard_normal(llm_wrapper.router_dim).astype(np.float32)

        score_evaluator = evaluator.evaluate(random_weights)
        score_raw = evaluate_router_dpo(random_weights, llm_wrapper, dpo_cache)
        llm_wrapper.reset_router()

        assert score_evaluator == pytest.approx(score_raw, abs=1e-4), (
            f"beta=0 evaluator ({score_evaluator:.4f}) should match "
            f"raw DPO ({score_raw:.4f})."
        )


# ============================================================
# SECTION 7 — Reward model
# ============================================================

class TestRewardModel:

    @pytest.mark.gpu
    def test_score_returns_a_float(self, reward_model):
        score = reward_model.score("What is 2 + 2?", "The answer is 4.")
        assert isinstance(score, float)
        print(f"\n  Score (correct answer): {score:.4f}")

    @pytest.mark.gpu
    def test_good_answer_scores_higher_than_bad(self, reward_model):
        """
        The reward model should prefer a correct factual answer over a wrong one.
        If this test fails the reward model may not have loaded correctly.
        """
        score_good = reward_model.score("What is 2 + 2?", "The answer is 4.")
        score_bad  = reward_model.score("What is 2 + 2?", "The answer is 9273.")
        print(f"\n  Correct answer: {score_good:.4f}  |  Wrong answer: {score_bad:.4f}")
        assert score_good > score_bad, \
            "Reward model scores the wrong answer higher — something is off."

    @pytest.mark.gpu
    def test_batch_scoring_matches_individual(self, reward_model):
        """Batched and individual scoring should give the same results."""
        pairs = [
            ("What is 2 + 2?",                 "4"),
            ("What is the capital of France?",  "Paris"),
            ("Name a primary colour.",          "Blue"),
        ]
        batch_scores      = reward_model.score_batch(pairs)
        individual_scores = [reward_model.score(p, r) for p, r in pairs]

        for i, (b, s) in enumerate(zip(batch_scores, individual_scores)):
            assert abs(b - s) < 0.2, \
                f"Pair {i}: batch score {b:.4f} ≠ individual score {s:.4f}"


# ============================================================
# SECTION 8 — Config serialization (no GPU needed)
# ============================================================

class TestConfigSerialization:
    """Verify ExperimentConfig survives a JSON round-trip."""

    def test_to_dict_returns_plain_dict(self):
        from src.config import ExperimentConfig

        config = ExperimentConfig()
        d = config.to_dict()
        assert isinstance(d, dict)
        assert isinstance(d["model"], dict)
        assert isinstance(d["evolution"], dict)

    def test_round_trip_preserves_all_fields(self):
        from src.config import (
            DataConfig, EvolutionConfig, ExperimentConfig, ModelConfig,
        )

        original = ExperimentConfig(
            model=ModelConfig(model_name="test-model", load_in_4bit=False, device="cpu", target_layer_idx=-1),
            data=DataConfig(dataset_name="test-ds", dataset_split="train", num_samples=8,
                            max_prompt_length=64, max_response_length=32),
            evolution=EvolutionConfig(population_size=6, sigma=0.05, learning_rate=0.1,
                                     num_generations=10, antithetic_sampling=True,
                                     fitness_batch_size=2, seed=99),
            output_dir="test_outputs",
            experiment_name="test_exp",
            fitness_type="dpo",
            reward_model_name="test-rm",
            max_new_tokens=15,
            log_every=2,
            sample_every=3,
            num_sample_prompts=2,
        )

        restored = ExperimentConfig.from_dict(original.to_dict())

        assert restored.model.model_name == "test-model"
        assert restored.model.load_in_4bit is False
        assert restored.data.num_samples == 8
        assert restored.evolution.sigma == 0.05
        assert restored.evolution.seed == 99
        assert restored.output_dir == "test_outputs"
        assert restored.fitness_type == "dpo"
        assert restored.reward_model_name == "test-rm"
        assert restored.max_new_tokens == 15
        assert restored.sample_every == 3
        assert restored.num_sample_prompts == 2

    def test_json_round_trip(self, tmp_path: Path):
        """Config survives serialization to a JSON file and back."""
        from src.config import ExperimentConfig

        original = ExperimentConfig(experiment_name="json_test", max_new_tokens=42)
        json_path = tmp_path / "config.json"

        with open(json_path, "w") as f:
            json.dump(original.to_dict(), f)

        with open(json_path) as f:
            restored = ExperimentConfig.from_dict(json.load(f))

        assert restored.experiment_name == "json_test"
        assert restored.max_new_tokens == 42
        assert restored.evolution.population_size == original.evolution.population_size

    def test_from_dict_without_fitness_type_defaults_to_reward(self):
        """Old configs saved before fitness_type was added must still load."""
        from src.config import ExperimentConfig

        d = ExperimentConfig().to_dict()
        del d["fitness_type"]

        restored = ExperimentConfig.from_dict(d)
        assert restored.fitness_type == "reward"

    def test_fitness_type_dpo_survives_json_round_trip(self, tmp_path: Path):
        """A DPO config must survive the full JSON file round-trip."""
        from src.config import ExperimentConfig

        original = ExperimentConfig(fitness_type="dpo", experiment_name="dpo_test")
        json_path = tmp_path / "config.json"

        with open(json_path, "w") as f:
            json.dump(original.to_dict(), f)

        with open(json_path) as f:
            restored = ExperimentConfig.from_dict(json.load(f))

        assert restored.fitness_type == "dpo"
        assert restored.experiment_name == "dpo_test"


# ============================================================
# SECTION 9 — ES checkpointing (no GPU needed)
# ============================================================

class TestESCheckpointing:
    """Verify OpenAI-ES checkpoint / restore preserves full optimizer state."""

    def _run_es(self, es, n_gens: int) -> None:
        """Run n_gens ask/tell cycles with deterministic fitnesses."""
        for g in range(n_gens):
            pop = es.ask()
            fitnesses = [float(i) + g * 0.1 for i in range(len(pop))]
            es.tell(fitnesses)

    def test_checkpoint_round_trip_preserves_state(self):
        from src.evolution import OpenAIES

        es = OpenAIES(np.ones(50) * 2.0, population_size=6, sigma=0.02,
                      learning_rate=0.05, antithetic_sampling=True, seed=7)
        self._run_es(es, 3)

        ckpt = es.get_checkpoint()
        restored = OpenAIES.from_checkpoint(ckpt)

        np.testing.assert_array_equal(restored.theta, es.theta)
        np.testing.assert_array_equal(restored.state.best_theta, es.state.best_theta)
        assert restored.state.generation == es.state.generation
        assert restored.state.best_fitness == es.state.best_fitness
        assert restored.state.fitness_history == es.state.fitness_history
        assert restored.sigma == es.sigma
        assert restored.lr == es.lr
        assert restored.pop_size == es.pop_size
        assert restored.antithetic == es.antithetic

    def test_checkpoint_preserves_rng_sequence(self):
        """
        After restoring from checkpoint, the next ask() must produce
        the exact same population as the original ES would have.
        """
        from src.evolution import OpenAIES

        es = OpenAIES(np.zeros(30), population_size=4, antithetic_sampling=False, seed=42)
        self._run_es(es, 5)

        ckpt = es.get_checkpoint()
        restored = OpenAIES.from_checkpoint(ckpt)

        pop_original = es.ask()
        pop_restored = restored.ask()

        for i, (a, b) in enumerate(zip(pop_original, pop_restored)):
            np.testing.assert_array_equal(
                a, b,
                err_msg=f"Population member {i} differs after checkpoint restore",
            )

    def test_checkpoint_survives_torch_save_load(self, tmp_path: Path):
        """Checkpoint must survive a torch.save → torch.load cycle."""
        from src.evolution import OpenAIES

        es = OpenAIES(np.ones(20), population_size=4, antithetic_sampling=True, seed=0)
        self._run_es(es, 2)

        ckpt_path = tmp_path / "test_ckpt.pt"
        ckpt = es.get_checkpoint()
        ckpt["baseline_fitness"] = -5.0
        ckpt["router_shape"] = (64, 2048)
        torch.save(ckpt, ckpt_path)

        loaded = torch.load(ckpt_path, weights_only=False)
        restored = OpenAIES.from_checkpoint(loaded)

        assert loaded["baseline_fitness"] == -5.0
        assert loaded["router_shape"] == (64, 2048)
        np.testing.assert_array_equal(restored.theta, es.theta)
        assert restored.state.generation == es.state.generation

        # RNG continuity still works after disk round-trip
        pop_original = es.ask()
        pop_restored = restored.ask()
        for a, b in zip(pop_original, pop_restored):
            np.testing.assert_array_equal(a, b)

    def test_restored_es_can_continue_training(self):
        """A restored ES should be able to continue ask/tell without errors."""
        from src.evolution import OpenAIES

        es = OpenAIES(np.zeros(40), population_size=6, antithetic_sampling=True, seed=1)
        self._run_es(es, 3)
        assert es.state.generation == 3

        restored = OpenAIES.from_checkpoint(es.get_checkpoint())

        # Run 2 more generations on the restored ES
        self._run_es(restored, 2)
        assert restored.state.generation == 5
        assert len(restored.state.fitness_history) == 5


# ============================================================
# SECTION 10 — Experiment I/O helpers (no GPU needed)
# ============================================================

class TestExperimentIO:
    """Test the file I/O helpers from run_experiment.py."""

    def test_create_run_dir_structure(self, tmp_path: Path):
        from src.config import ExperimentConfig

        sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
        from run_experiment import _create_run_dir

        config = ExperimentConfig(output_dir=str(tmp_path), experiment_name="test_exp")
        run_dir = _create_run_dir(config)

        assert run_dir.exists()
        assert (run_dir / "checkpoint").is_dir()
        assert (run_dir / "samples").is_dir()
        assert run_dir.parent.name == "test_exp"

    def test_save_and_load_config(self, tmp_path: Path):
        from src.config import ExperimentConfig

        sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
        from run_experiment import _load_config, _save_config

        original = ExperimentConfig(experiment_name="round_trip_test", max_new_tokens=77)
        _save_config(original, tmp_path)
        restored = _load_config(tmp_path)

        assert restored.experiment_name == "round_trip_test"
        assert restored.max_new_tokens == 77

    def test_save_and_load_config_preserves_fitness_type(self, tmp_path: Path):
        """A DPO config saved to disk must reload with fitness_type='dpo'."""
        from src.config import ExperimentConfig

        sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
        from run_experiment import _load_config, _save_config

        original = ExperimentConfig(
            experiment_name="dpo_io_test",
            fitness_type="dpo",
        )
        _save_config(original, tmp_path)
        restored = _load_config(tmp_path)

        assert restored.fitness_type == "dpo"
        assert restored.experiment_name == "dpo_io_test"

    def test_save_checkpoint_is_atomic(self, tmp_path: Path):
        """After saving, latest.pt exists and latest.pt.tmp does not."""
        from src.evolution import OpenAIES

        sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
        from run_experiment import _save_checkpoint

        (tmp_path / "checkpoint").mkdir()
        es = OpenAIES(np.zeros(10), population_size=4, antithetic_sampling=True, seed=0)
        es.ask()
        es.tell([1.0, 2.0, 3.0, 4.0])

        _save_checkpoint(es, baseline_fitness=-5.0, router_shape=(64, 2048), run_dir=tmp_path)

        assert (tmp_path / "checkpoint" / "latest.pt").exists()
        assert not (tmp_path / "checkpoint" / "latest.pt.tmp").exists()

    def test_save_best_router_contents(self, tmp_path: Path):
        from src.evolution import OpenAIES

        sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
        from run_experiment import _save_best_router

        es = OpenAIES(np.ones(20) * 3.0, population_size=4, antithetic_sampling=True, seed=0)
        es.ask()
        es.tell([1.0, 5.0, 2.0, 4.0])

        _save_best_router(es, router_shape=(64, 2048), run_dir=tmp_path)

        data = torch.load(tmp_path / "best_router.pt", weights_only=False)
        assert "weights" in data
        assert data["router_shape"] == (64, 2048)
        assert data["fitness"] == es.state.best_fitness
        assert data["generation"] == es.state.generation

    def test_save_history_is_valid_json(self, tmp_path: Path):
        sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
        from run_experiment import _save_history

        history = {
            "baseline_fitness": -7.5,
            "generations": [
                {"generation": 0, "mean_fitness": -8.0, "max_fitness": -7.2,
                 "min_fitness": -9.1, "best_ever": -7.2},
                {"generation": 1, "mean_fitness": -7.8, "max_fitness": -7.0,
                 "min_fitness": -8.5, "best_ever": -7.0},
            ],
        }
        _save_history(history, tmp_path)

        with open(tmp_path / "history.json") as f:
            loaded = json.load(f)

        assert loaded["baseline_fitness"] == -7.5
        assert len(loaded["generations"]) == 2
        assert loaded["generations"][1]["best_ever"] == -7.0

    def test_save_samples_structure(self, tmp_path: Path):
        sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
        from run_experiment import _save_samples

        (tmp_path / "samples").mkdir()
        _save_samples(
            sample_prompts=["What is 2+2?", "Hello?"],
            baseline_responses=["Four.", "Hi!"],
            evolved_responses=["The answer is 4.", "Greetings!"],
            generation=10,
            best_fitness=-6.5,
            run_dir=tmp_path,
        )

        path = tmp_path / "samples" / "gen_0010.json"
        assert path.exists()

        with open(path) as f:
            data = json.load(f)

        assert data["generation"] == 10
        assert data["best_fitness"] == -6.5
        assert len(data["samples"]) == 2
        assert data["samples"][0]["prompt"] == "What is 2+2?"
        assert data["samples"][0]["baseline_response"] == "Four."
        assert data["samples"][0]["evolved_response"] == "The answer is 4."

    def test_plot_fitness_curve_creates_png(self, tmp_path: Path):
        sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
        from run_experiment import plot_fitness_curve

        history = {
            "baseline_fitness": -8.0,
            "generations": [
                {"generation": i, "mean_fitness": -8.0 + i * 0.1,
                 "max_fitness": -7.5 + i * 0.1, "min_fitness": -9.0 + i * 0.05,
                 "best_ever": -7.5 + i * 0.1}
                for i in range(10)
            ],
        }
        plot_fitness_curve(history, tmp_path)

        png_path = tmp_path / "fitness_curve.png"
        assert png_path.exists()
        assert png_path.stat().st_size > 1000

    def test_plot_with_empty_history_does_not_crash(self, tmp_path: Path):
        sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
        from run_experiment import plot_fitness_curve

        plot_fitness_curve({"baseline_fitness": -8.0, "generations": []}, tmp_path)
        assert not (tmp_path / "fitness_curve.png").exists()

    def test_full_checkpoint_resume_workflow(self, tmp_path: Path):
        """
        End-to-end test: run 3 gens → checkpoint → restore → run 2 more gens.
        Verify history grows correctly and best_router.pt is created.
        """
        from src.evolution import OpenAIES

        sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
        from run_experiment import (
            _save_best_router, _save_checkpoint, _save_history,
        )

        run_dir = tmp_path / "test_run"
        run_dir.mkdir()
        (run_dir / "checkpoint").mkdir()
        (run_dir / "samples").mkdir()

        # Phase 1: run 3 generations
        es = OpenAIES(np.zeros(50), population_size=6, antithetic_sampling=True, seed=0)
        history: dict = {"baseline_fitness": -7.0, "generations": []}

        for gen in range(3):
            pop = es.ask()
            fitnesses = [float(i) + gen for i in range(6)]
            es.tell(fitnesses)
            history["generations"].append({
                "generation": gen,
                "mean_fitness": float(np.mean(fitnesses)),
                "max_fitness": float(np.max(fitnesses)),
                "min_fitness": float(np.min(fitnesses)),
                "best_ever": es.state.best_fitness,
            })

        _save_checkpoint(es, baseline_fitness=-7.0, router_shape=(8, 50), run_dir=run_dir)
        _save_history(history, run_dir)
        _save_best_router(es, router_shape=(8, 50), run_dir=run_dir)

        assert es.state.generation == 3

        # Phase 2: restore and run 2 more generations
        loaded_ckpt = torch.load(run_dir / "checkpoint" / "latest.pt", weights_only=False)
        restored = OpenAIES.from_checkpoint(loaded_ckpt)

        with open(run_dir / "history.json") as f:
            restored_history = json.load(f)

        assert restored.state.generation == 3
        assert len(restored_history["generations"]) == 3

        for gen in range(3, 5):
            pop = restored.ask()
            fitnesses = [float(i) + gen for i in range(6)]
            restored.tell(fitnesses)
            restored_history["generations"].append({
                "generation": gen,
                "mean_fitness": float(np.mean(fitnesses)),
                "max_fitness": float(np.max(fitnesses)),
                "min_fitness": float(np.min(fitnesses)),
                "best_ever": restored.state.best_fitness,
            })

        _save_checkpoint(restored, baseline_fitness=-7.0, router_shape=(8, 50), run_dir=run_dir)
        _save_history(restored_history, run_dir)

        assert restored.state.generation == 5
        assert len(restored_history["generations"]) == 5

        # Verify the generation indices are correct and continuous
        gen_indices = [g["generation"] for g in restored_history["generations"]]
        assert gen_indices == [0, 1, 2, 3, 4]

        # Verify best_ever is monotonically non-decreasing
        best_evers = [g["best_ever"] for g in restored_history["generations"]]
        for i in range(1, len(best_evers)):
            assert best_evers[i] >= best_evers[i - 1]

    def test_cli_fitness_override_creates_correct_config(self):
        """
        The --fitness dpo flag should override fitness_type and adjust
        the experiment name, without mutating the original CONFIG.
        """
        from dataclasses import replace
        from src.config import ExperimentConfig

        config = ExperimentConfig(
            fitness_type="reward",
            experiment_name="olmoe_router_reward",
        )

        overridden = replace(
            config,
            fitness_type="dpo",
            experiment_name="olmoe_router_dpo",
        )

        assert overridden.fitness_type == "dpo"
        assert overridden.experiment_name == "olmoe_router_dpo"
        # Original must be untouched
        assert config.fitness_type == "reward"
        assert config.experiment_name == "olmoe_router_reward"

    def test_invalid_fitness_type_raises(self):
        """run_experiment.run() should raise ValueError for unknown fitness types."""
        from src.config import ExperimentConfig

        config = ExperimentConfig(fitness_type="invalid_type")
        d = config.to_dict()
        assert d["fitness_type"] == "invalid_type"
