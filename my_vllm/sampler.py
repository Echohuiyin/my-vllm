"""Sampler: applies sampling strategies to model logits."""

from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn.functional as F

from my_vllm.sampling_params import SamplingParams


class Sampler:
    """Samples next tokens from logits given per-request SamplingParams."""

    def __call__(
        self,
        logits: torch.Tensor,
        sampling_params_list: List[SamplingParams],
    ) -> List[int]:
        """
        Args:
            logits: (batch_size, vocab_size) in float32
            sampling_params_list: one SamplingParams per batch entry

        Returns:
            list of sampled token IDs, one per batch entry
        """
        results = []
        for i, sp in enumerate(sampling_params_list):
            row_logits = logits[i]
            token_id = self._sample_one(row_logits, sp)
            results.append(token_id)
        return results

    def _sample_one(
        self,
        logits: torch.Tensor,
        sp: SamplingParams,
    ) -> int:
        if sp.is_greedy:
            return logits.argmax(dim=-1).item()

        logits = logits / sp.temperature

        if sp.top_k > 0:
            logits = self._apply_top_k(logits, sp.top_k)

        if sp.top_p < 1.0:
            logits = self._apply_top_p(logits, sp.top_p)

        probs = F.softmax(logits, dim=-1)

        if sp.seed is not None:
            generator = torch.Generator(device=logits.device)
            generator.manual_seed(sp.seed)
            token_id = torch.multinomial(probs, num_samples=1, generator=generator)
        else:
            token_id = torch.multinomial(probs, num_samples=1)

        return token_id.item()

    @staticmethod
    def _apply_top_k(logits: torch.Tensor, top_k: int) -> torch.Tensor:
        top_k = min(top_k, logits.size(-1))
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1:]
        logits = logits.masked_fill(indices_to_remove, float("-inf"))
        return logits

    @staticmethod
    def _apply_top_p(logits: torch.Tensor, top_p: float) -> torch.Tensor:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(
            F.softmax(sorted_logits, dim=-1), dim=-1
        )
        sorted_mask = cumulative_probs - F.softmax(sorted_logits, dim=-1) >= top_p
        sorted_logits[sorted_mask] = float("-inf")
        logits = logits.scatter(-1, sorted_indices, sorted_logits)
        return logits
