"""Exponential Moving Average (EMA) of model weights.

Maintains a shadow copy of model parameters, updated each step via:
    shadow = decay * shadow + (1 - decay) * model

At evaluation time, swap model -> shadow (with backup), evaluate, then
restore. Yields smoother predictions equivalent to averaging late-epoch
checkpoints. EMA is the only legal "pseudo-ensemble" allowed by the
contest rules.

Notes:
- ``dense_only=True`` skips Embedding tables -> ~halves shadow memory at
  the cost of EMA-ing the dense backbone only. Useful when GPU memory
  is tight.
- After cold-restart of high-cardinality embeddings, the corresponding
  EMA shadows must also be reset via ``reset_for_keys`` to avoid the
  shadow lagging behind reinitialized weights.
"""

import logging
from typing import Dict, Iterable, Optional, Set

import torch
import torch.nn as nn


class EMAModel:
    """Tracks a shadow copy of model parameters via exponential moving average."""

    def __init__(
        self,
        model: nn.Module,
        decay: float = 0.999,
        dense_only: bool = False,
    ) -> None:
        self.decay: float = float(decay)
        self.dense_only: bool = bool(dense_only)
        self.shadow: Dict[str, torch.Tensor] = {}
        self.backup: Optional[Dict[str, torch.Tensor]] = None
        self._tracked_keys: Set[str] = set()
        self._init_shadow(model)

    def _embedding_param_ids(self, model: nn.Module) -> Set[int]:
        """Return ``data_ptr()`` set of all nn.Embedding weight tensors."""
        ids: Set[int] = set()
        for module in model.modules():
            if isinstance(module, nn.Embedding):
                ids.add(module.weight.data_ptr())
        return ids

    def _is_tracked(
        self, name: str, param: torch.Tensor, emb_ids: Set[int]
    ) -> bool:
        if not param.requires_grad:
            return False
        if self.dense_only and param.data_ptr() in emb_ids:
            return False
        return True

    def _init_shadow(self, model: nn.Module) -> None:
        emb_ids = self._embedding_param_ids(model) if self.dense_only else set()
        for name, p in model.named_parameters():
            if not self._is_tracked(name, p, emb_ids):
                continue
            self.shadow[name] = p.detach().clone()
            self._tracked_keys.add(name)
        n_params = sum(int(t.numel()) for t in self.shadow.values())
        n_total = sum(int(p.numel()) for p in model.parameters())
        logging.info(
            f"EMAModel initialized: tracking {len(self.shadow)} param tensors "
            f"({n_params:,} / {n_total:,} elements, "
            f"{100.0*n_params/max(1,n_total):.1f}%), "
            f"decay={self.decay}, dense_only={self.dense_only}"
        )

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        """Apply one EMA step using current model weights."""
        d = self.decay
        for name, p in model.named_parameters():
            if name not in self._tracked_keys:
                continue
            self.shadow[name].mul_(d).add_(p.detach(), alpha=1.0 - d)

    @torch.no_grad()
    def swap_in(self, model: nn.Module) -> None:
        """Replace model params with EMA shadow; backs up originals."""
        assert self.backup is None, "swap_in called twice without restore()"
        self.backup = {}
        for name, p in model.named_parameters():
            if name in self._tracked_keys:
                self.backup[name] = p.detach().clone()
                p.data.copy_(self.shadow[name])

    @torch.no_grad()
    def restore(self, model: nn.Module) -> None:
        """Restore original model params from backup."""
        assert self.backup is not None, "restore() called without prior swap_in()"
        for name, p in model.named_parameters():
            if name in self._tracked_keys:
                p.data.copy_(self.backup[name])
        self.backup = None

    @torch.no_grad()
    def reset_for_keys(
        self, model: nn.Module, keys: Iterable[str]
    ) -> None:
        """Reset shadow to current model weights for the listed param names.

        Call this after cold-restart of embeddings so the shadow does not
        keep stale pre-reset values.
        """
        param_dict = dict(model.named_parameters())
        n_reset = 0
        for k in keys:
            if k in self._tracked_keys and k in param_dict:
                self.shadow[k] = param_dict[k].detach().clone()
                n_reset += 1
        if n_reset > 0:
            logging.info(f"EMAModel: reset shadow for {n_reset} keys after embedding cold-restart")

    @torch.no_grad()
    def reset_by_data_ptrs(
        self, model: nn.Module, ptrs: Set[int]
    ) -> None:
        """Reset shadow for params whose ``data_ptr()`` is in ``ptrs``.

        Used after cold-restart, where reinit replaces tensor storage; the
        old shadow keys still exist by name, so we resync via name lookup
        guided by data_ptr matches.
        """
        keys_to_reset = []
        for name, p in model.named_parameters():
            if name in self._tracked_keys and p.data_ptr() in ptrs:
                keys_to_reset.append(name)
        self.reset_for_keys(model, keys_to_reset)

    def state_dict(self) -> Dict[str, torch.Tensor]:
        return {k: v.detach().clone() for k, v in self.shadow.items()}

    def load_state_dict(self, state: Dict[str, torch.Tensor]) -> None:
        self.shadow = {k: v.clone() for k, v in state.items()}
        self._tracked_keys = set(state.keys())
        self.backup = None
