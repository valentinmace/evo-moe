"""
Wrapper around OLMoE that exposes:
  - Activation caching: run a full forward pass and store the post-attention
    residuals of the target MoE layer (one-time cost, done before evolution).
  - Forward tail: given cached residuals, run *only* the MoE FFN + final norm
    + LM head under arbitrary router weights (cheap, called per candidate).
  - Router weight getter / setter for the evolutionary loop.

Architecture reminder (OlmoeMoeDecoderLayer.forward):
    residual = hidden_states
    hidden_states = input_layernorm(hidden_states)
    hidden_states = self_attn(hidden_states, ...)
    hidden_states = residual + hidden_states          # ← post-attention residual
    residual = hidden_states
    hidden_states = post_attention_layernorm(hidden_states)
    hidden_states, router_logits = mlp(hidden_states)
    hidden_states = residual + hidden_states

We hook post_attention_layernorm to capture its *input* (= post-attention
residual).  During evolution we replay from that point, varying only the router.
"""

import logging
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from .config import ModelConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data container for the activation cache
# ---------------------------------------------------------------------------

@dataclass
class ActivationCache:
    """
    Stores the post-attention residuals captured just before the target MoE
    layer's FFN, together with the token ids and response masks needed to
    compute log-probabilities.

    All activation tensors are kept in fp16 on CPU to save VRAM.
    """

    # One tensor per sequence, shape [seq_len, hidden_size] in fp16
    chosen_activations: list[torch.Tensor] = field(default_factory=list)
    rejected_activations: list[torch.Tensor] = field(default_factory=list)

    # Token ids for the full sequence (prompt + response), shape [seq_len]
    chosen_input_ids: list[torch.Tensor] = field(default_factory=list)
    rejected_input_ids: list[torch.Tensor] = field(default_factory=list)

    # Boolean mask: True at positions belonging to the response tokens
    chosen_response_masks: list[torch.Tensor] = field(default_factory=list)
    rejected_response_masks: list[torch.Tensor] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Main wrapper
# ---------------------------------------------------------------------------

class OLMoEWrapper:
    """Thin wrapper that exposes router manipulation and activation caching."""

    def __init__(self, config: ModelConfig) -> None:
        self.config = config
        self.device = torch.device(config.device)
        self.model, self.tokenizer = self._load_model()

        # Resolve negative index and validate that it is the last layer.
        # Activation caching (compute_logits_from_cache) replays only the
        # target layer's FFN then jumps straight to the final norm + LM head,
        # skipping any subsequent layers.  KV cache reuse also relies on every
        # layer before the target being frozen.  Both assumptions hold only
        # when the target is the very last decoder layer.
        num_layers = len(self.model.model.layers)
        self._target_idx: int = config.target_layer_idx % num_layers
        if self._target_idx != num_layers - 1:
            raise ValueError(
                f"target_layer_idx resolved to {self._target_idx}, but "
                f"activation caching and KV cache reuse only work for the "
                f"last decoder layer (index {num_layers - 1}). Evolving an "
                f"intermediate layer would require replaying all subsequent "
                f"layers, which is not implemented."
            )
        self._last_layer = self.model.model.layers[self._target_idx]

        # Detect the model's actual compute dtype from a non-quantized layer.
        # With load_in_4bit=True and no explicit torch_dtype, OLMoE loads its
        # non-quantized parameters in bfloat16 (its native checkpoint dtype).
        self._compute_dtype: torch.dtype = self.model.model.embed_tokens.weight.dtype

        # Ensure the router weight dtype matches the rest of the model so that
        # F.linear(hidden_states, gate.weight) never hits a dtype mismatch.
        self._normalize_gate_dtype()

        # Keep a copy of the original pre-trained router weights as a baseline.
        self._original_router_weights: np.ndarray = self.get_router_weights()

        logger.info(
            "OLMoEWrapper ready. "
            f"Evolving router of layer {self._target_idx} "
            f"(shape {self.router_shape}, dim={self.router_dim:,})."
        )

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def _load_model(self) -> tuple[PreTrainedModel, PreTrainedTokenizerBase]:
        logger.info(f"Loading model: {self.config.model_name}")

        if self.config.load_in_4bit:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
        else:
            bnb_config = None

        model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            quantization_config=bnb_config,
            # Map everything onto a single GPU; the 4-bit model fits in ~3.5 GB
            device_map={"": 0},
            torch_dtype=torch.float16 if not self.config.load_in_4bit else None,
        )
        model.eval()

        tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        return model, tokenizer

    # ------------------------------------------------------------------
    # Gate (router) management
    # ------------------------------------------------------------------

    def _normalize_gate_dtype(self) -> None:
        """
        Cast the router weight to match the model's compute dtype.

        OlmoeTopKRouter stores its routing matrix as a raw nn.Parameter used
        via F.linear.  The checkpoint dtype (typically bfloat16 for OLMoE) may
        differ from what we would naively assume, so we align explicitly using
        the dtype detected from the embedding layer rather than hardcoding.
        """
        gate = self._last_layer.mlp.gate
        gate.weight.data = gate.weight.data.to(self._compute_dtype)
        logger.info(f"Router weight dtype normalised to {self._compute_dtype}.")

    @property
    def router_shape(self) -> tuple[int, int]:
        """(num_experts, hidden_size)"""
        w = self._last_layer.mlp.gate.weight
        return (w.shape[0], w.shape[1])

    @property
    def router_dim(self) -> int:
        """Total number of router parameters (flattened)."""
        s = self.router_shape
        return s[0] * s[1]

    def get_router_weights(self) -> np.ndarray:
        """Return a flattened fp32 numpy copy of the current router weights."""
        return (
            self._last_layer.mlp.gate.weight.data
            .detach()
            .cpu()
            .float()
            .numpy()
            .flatten()
        )

    def set_router_weights(self, weights_flat: np.ndarray) -> None:
        """Set router weights from a flat float32 numpy array."""
        w = (
            torch.from_numpy(weights_flat)
            .reshape(self.router_shape)
            .to(device=self.device, dtype=self._compute_dtype)
        )
        self._last_layer.mlp.gate.weight.data = w

    def reset_router(self) -> None:
        """Restore the original pre-trained router weights."""
        self.set_router_weights(self._original_router_weights)

    # ------------------------------------------------------------------
    # Activation caching (one-time pass over the dataset)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def cache_activations(
        self,
        sequences: list[dict],
        show_progress: bool = True,
    ) -> ActivationCache:
        """
        Run a full forward pass for each sequence and capture the
        post-attention residual entering the target MoE layer.

        Args:
            sequences: list of dicts with keys:
                - 'chosen_ids'             : LongTensor [seq_len]
                - 'rejected_ids'           : LongTensor [seq_len]
                - 'chosen_response_start'  : int
                - 'rejected_response_start': int

        Returns:
            ActivationCache with FP16 activations stored on CPU.
        """
        cache = ActivationCache()

        iterator = tqdm(sequences, desc="Caching activations") if show_progress else sequences

        for seq in iterator:
            for side in ("chosen", "rejected"):
                input_ids = seq[f"{side}_ids"].unsqueeze(0).to(self.device)
                response_start: int = seq[f"{side}_response_start"]

                # Capture the input to post_attention_layernorm via a hook
                captured: list[torch.Tensor] = []

                def _hook(
                    module: nn.Module,
                    inp: tuple[torch.Tensor, ...],
                    out: torch.Tensor,
                    buf: list = captured,
                ) -> None:
                    # inp[0] is the post-attention residual (before layernorm)
                    buf.append(inp[0].detach().squeeze(0).cpu().to(self._compute_dtype))

                hook_handle = (
                    self._last_layer.post_attention_layernorm
                    .register_forward_hook(_hook)
                )

                self.model(input_ids, use_cache=False)
                hook_handle.remove()

                activation = captured[0]  # [seq_len, hidden_size]
                seq_len = activation.shape[0]

                response_mask = torch.zeros(seq_len, dtype=torch.bool)
                response_mask[response_start:] = True

                getattr(cache, f"{side}_activations").append(activation)
                getattr(cache, f"{side}_input_ids").append(seq[f"{side}_ids"])
                getattr(cache, f"{side}_response_masks").append(response_mask)

        return cache

    # ------------------------------------------------------------------
    # Forward tail (cheap per-candidate evaluation)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def compute_logits_from_cache(
        self,
        post_attn_residual: torch.Tensor,  # [seq_len, hidden_size], compute dtype
    ) -> torch.Tensor:  # [seq_len, vocab_size]
        """
        Run only the MoE FFN portion of the target layer plus the final norm
        and LM head, starting from a cached post-attention residual.

        The router weights used are whatever is currently set in the model,
        so call set_router_weights() before this to evaluate a candidate.
        """
        x = post_attn_residual.unsqueeze(0).to(device=self.device, dtype=self._compute_dtype)
        residual = x

        # MoE FFN sub-block: post_attention_layernorm → mlp → residual add
        x_normed = self._last_layer.post_attention_layernorm(x)
        mlp_output = self._last_layer.mlp(x_normed)

        # OlmoeMoE always returns (hidden_states, router_logits)
        if isinstance(mlp_output, tuple):
            x = residual + mlp_output[0]
        else:
            x = residual + mlp_output

        # Final model norm and vocabulary projection
        x = self.model.model.norm(x)
        logits = self.model.lm_head(x)

        return logits.squeeze(0)  # [seq_len, vocab_size]

    # ------------------------------------------------------------------
    # KV-cache utilities for generation-based fitness
    # ------------------------------------------------------------------

    @torch.no_grad()
    def cache_prompt_kv(
        self,
        prompt_ids: torch.Tensor,  # [1, prompt_len]
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """
        Run a forward pass on prompt_ids[:-1] and return the KV cache.

        Why is this reusable across router candidates?
        -----------------------------------------------
        The KV values at layer L are computed from the HIDDEN STATES that
        enter that layer (= output of layer L-1).  Because we only modify the
        LAST layer's MoE router, layers 0..N-2 are fully frozen.  Therefore
        all KV tensors are identical for every router candidate.

        We store the cache on CPU to free VRAM between generation calls.

        Returns
        -------
        List of (K, V) tensors per layer, all on CPU in fp16.
        """
        output = self.model(
            prompt_ids[:, :-1].to(self.device),  # all but last token
            use_cache=True,
            output_attentions=False,
        )
        # DynamicCache stores per-layer data in .layers, each with .keys/.values.
        cpu_kv = [
            (layer.keys.detach().cpu(), layer.values.detach().cpu())
            for layer in output.past_key_values.layers
        ]
        return cpu_kv

    @torch.no_grad()
    def generate_from_prompt_kv(
        self,
        prompt_ids: torch.Tensor,                          # [1, prompt_len]
        prompt_kv_cpu: list[tuple[torch.Tensor, torch.Tensor]],
        max_new_tokens: int = 50,
        do_sample: bool = False,
        temperature: float = 1.0,
    ) -> torch.Tensor:  # [new_tokens]
        """
        Generate a response using a pre-computed prompt KV cache.

        Uses manual autoregressive decoding rather than model.generate()
        because the high-level API no longer supports injecting a pre-filled
        DynamicCache cleanly (it conflicts with the internal prefill logic).

        The router weights currently set in the model are used throughout
        generation — call set_router_weights() before this.

        Args:
            prompt_ids      : full tokenised prompt, shape [1, prompt_len].
            prompt_kv_cpu   : KV cache from cache_prompt_kv(), stored on CPU.
            max_new_tokens  : maximum response length in tokens.
            do_sample       : greedy (False) or sampling (True).
            temperature     : sampling temperature (ignored when do_sample=False).

        Returns:
            1-D LongTensor of generated token ids (without the prompt).
        """
        # Reconstruct a DynamicCache from the CPU-stored (K, V) pairs.
        from transformers import DynamicCache  # noqa: PLC0415
        gpu_kv_data = [
            (k.to(self.device, dtype=self._compute_dtype),
             v.to(self.device, dtype=self._compute_dtype))
            for k, v in prompt_kv_cpu
        ]
        kv_cache = DynamicCache(ddp_cache_data=gpu_kv_data)

        # The cache covers positions 0..prompt_len-2.  The last prompt token
        # is our first input, and its position = prompt_len - 1.
        current_token = prompt_ids[:, -1:].to(self.device)
        generated: list[torch.Tensor] = []

        for _ in range(max_new_tokens):
            output = self.model(
                input_ids=current_token,
                past_key_values=kv_cache,
                use_cache=True,
            )
            kv_cache = output.past_key_values

            next_logits = output.logits[:, -1, :]
            if do_sample:
                probs = torch.softmax(next_logits / temperature, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = next_logits.argmax(dim=-1, keepdim=True)

            generated.append(next_token.squeeze(0))

            if next_token.item() == self.tokenizer.eos_token_id:
                break

            current_token = next_token

        if not generated:
            return torch.tensor([], dtype=torch.long)
        return torch.cat(generated)
