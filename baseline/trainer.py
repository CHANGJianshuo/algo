"""PCVRHyFormer pointwise trainer (binary-classification, AUC-monitored).

Despite the historical "Ranking" suffix in the class name, the training loop
uses pointwise BCE / Focal loss and evaluates Binary AUC + binary logloss.
"""

import os
import glob
import math
import shutil
import logging
import contextlib
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import roc_auc_score

from utils import sigmoid_focal_loss, EarlyStopping
from model import ModelInput
from ema import EMAModel


def _make_warmup_cosine_lambda(
    warmup_steps: int,
    decay_steps: int,
    min_ratio: float = 0.1,
):
    """Builds a multiplier function for ``LambdaLR``.

    - step < warmup_steps     : linear ramp from 0 -> 1
    - decay_steps == 0        : stay at 1.0 after warmup (warmup-only)
    - decay_steps > 0         : cosine decay from 1 -> min_ratio over
                                ``decay_steps`` steps measured from end of
                                warmup; floor at min_ratio thereafter.
    """
    warmup_steps = max(0, int(warmup_steps))
    decay_steps = max(0, int(decay_steps))
    min_ratio = max(0.0, min(1.0, float(min_ratio)))

    def lr_lambda(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        if decay_steps == 0:
            return 1.0
        progress = (step - warmup_steps) / float(decay_steps)
        if progress >= 1.0:
            return min_ratio
        cos_v = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_ratio + (1.0 - min_ratio) * cos_v

    return lr_lambda


class PCVRHyFormerRankingTrainer:
    """PCVRHyFormer trainer for pointwise binary classification.

    Uses PCVR data layout:
    - user_int_feats, user_dense_feats
    - item_int_feats, item_dense_feats
    - seq_a, seq_b, seq_c, seq_d (each with *_len companion)
    - label (binary)

    Loss: BCEWithLogitsLoss or Focal Loss.
    Metrics: BinaryAUROC + binary logloss.
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        valid_loader: DataLoader,
        lr: float,
        num_epochs: int,
        device: str,
        save_dir: str,
        early_stopping: EarlyStopping,
        loss_type: str = 'bce',
        focal_alpha: float = 0.1,
        focal_gamma: float = 2.0,
        sparse_lr: float = 0.05,
        sparse_weight_decay: float = 0.0,
        reinit_sparse_after_epoch: int = 1,
        reinit_cardinality_threshold: int = 0,
        ckpt_params: Optional[Dict[str, Any]] = None,
        writer: Optional[Any] = None,
        schema_path: Optional[str] = None,
        ns_groups_path: Optional[str] = None,
        eval_every_n_steps: int = 0,
        train_config: Optional[Dict[str, Any]] = None,
        use_bf16: bool = False,
        use_ema: bool = False,
        ema_decay: float = 0.999,
        ema_dense_only: bool = False,
        use_lr_warmup: bool = False,
        warmup_steps: int = 1000,
        lr_decay_steps: int = 0,
        lr_min_ratio: float = 0.1,
        measure_latency: bool = True,
        latency_warmup: int = 50,
        latency_iters: int = 200,
    ) -> None:
        self.model: nn.Module = model
        self.train_loader: DataLoader = train_loader
        self.valid_loader: DataLoader = valid_loader
        self.writer = writer
        # schema_path is copied alongside every checkpoint so that infer.py can
        # rebuild the exact same feature schema the model was trained with.
        self.schema_path: Optional[str] = schema_path
        # ns_groups_path is optional; copied next to schema.json when provided
        # and points at an existing file. Keeping the JSON inside the ckpt dir
        # makes the checkpoint self-contained for evaluation environments that
        # do not ship ns_groups.json separately.
        self.ns_groups_path: Optional[str] = ns_groups_path

        # Dual optimizer: Adagrad for sparse Embeddings, AdamW for dense params.
        self.sparse_optimizer: Optional[torch.optim.Optimizer]
        if hasattr(model, 'get_sparse_params'):
            sparse_params = model.get_sparse_params()
            dense_params = model.get_dense_params()
            sparse_param_count = sum(p.numel() for p in sparse_params)
            dense_param_count = sum(p.numel() for p in dense_params)
            logging.info(f"Sparse params: {len(sparse_params)} tensors, {sparse_param_count:,} parameters (Adagrad lr={sparse_lr})")
            logging.info(f"Dense params: {len(dense_params)} tensors, {dense_param_count:,} parameters (AdamW lr={lr})")
            self.sparse_optimizer = torch.optim.Adagrad(
                sparse_params, lr=sparse_lr, weight_decay=sparse_weight_decay
            )
            self.dense_optimizer: torch.optim.Optimizer = torch.optim.AdamW(
                dense_params, lr=lr, betas=(0.9, 0.98)
            )
        else:
            self.sparse_optimizer = None
            self.dense_optimizer = torch.optim.AdamW(
                model.parameters(), lr=lr, betas=(0.9, 0.98)
            )

        self.num_epochs: int = num_epochs
        self.device: str = device
        self.save_dir: str = save_dir
        self.early_stopping: EarlyStopping = early_stopping
        self.loss_type: str = loss_type
        self.focal_alpha: float = focal_alpha
        self.focal_gamma: float = focal_gamma
        self.reinit_sparse_after_epoch: int = reinit_sparse_after_epoch
        self.reinit_cardinality_threshold: int = reinit_cardinality_threshold
        self.sparse_lr: float = sparse_lr
        self.sparse_weight_decay: float = sparse_weight_decay
        self.ckpt_params: Dict[str, Any] = ckpt_params or {}
        self.eval_every_n_steps: int = eval_every_n_steps
        self.train_config: Optional[Dict[str, Any]] = train_config

        # bfloat16 mixed-precision: covers forward + loss only;
        # backward and optimizer step stay fp32. Embedding lookup is in
        # autocast's natural-fp32 list, so sparse Adagrad keeps fp32 grads.
        self.use_bf16: bool = bool(use_bf16) and torch.cuda.is_available()

        # EMA: maintains a shadow copy of model weights, evaluated/saved
        # under the shadow. Only legal "pseudo-ensemble" allowed by contest.
        self.ema: Optional[EMAModel] = None
        if use_ema:
            self.ema = EMAModel(model, decay=ema_decay, dense_only=ema_dense_only)

        # Latency harness config (run after train() finishes).
        self.measure_latency: bool = bool(measure_latency)
        self.latency_warmup: int = int(latency_warmup)
        self.latency_iters: int = int(latency_iters)

        # LR scheduler: linear warmup -> (optional) cosine decay.
        # Applied only to dense_optimizer (AdamW); Adagrad sparse keeps fixed lr.
        self.dense_lr_scheduler: Optional[torch.optim.lr_scheduler.LambdaLR] = None
        if use_lr_warmup:
            lr_lambda = _make_warmup_cosine_lambda(
                warmup_steps=int(warmup_steps),
                decay_steps=int(lr_decay_steps),
                min_ratio=float(lr_min_ratio),
            )
            self.dense_lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
                self.dense_optimizer, lr_lambda=lr_lambda
            )

        logging.info(f"PCVRHyFormerRankingTrainer loss_type={loss_type}, "
                     f"focal_alpha={focal_alpha}, focal_gamma={focal_gamma}, "
                     f"reinit_sparse_after_epoch={reinit_sparse_after_epoch}, "
                     f"use_bf16={self.use_bf16}, "
                     f"use_ema={self.ema is not None}, "
                     f"ema_decay={ema_decay if self.ema is not None else 'n/a'}, "
                     f"use_lr_warmup={self.dense_lr_scheduler is not None}, "
                     f"warmup_steps={warmup_steps if self.dense_lr_scheduler else 'n/a'}, "
                     f"lr_decay_steps={lr_decay_steps if self.dense_lr_scheduler else 'n/a'}")

    def _autocast(self):
        """Context manager: bfloat16 autocast if enabled, otherwise no-op."""
        if self.use_bf16:
            return torch.cuda.amp.autocast(dtype=torch.bfloat16)
        return contextlib.nullcontext()

    def _build_step_dir_name(self, global_step: int, is_best: bool = False) -> str:
        """Build a checkpoint sub-directory name such as
        ``global_step2500.layer=2.head=4.hidden=64[.best_model]``.
        """
        parts = [f"global_step{global_step}"]
        for key in ("layer", "head", "hidden"):
            if key in self.ckpt_params:
                parts.append(f"{key}={self.ckpt_params[key]}")
        name = ".".join(parts)
        if is_best:
            name += ".best_model"
        return name

    def _write_sidecar_files(self, ckpt_dir: str) -> None:
        """Write sidecar files next to a ``model.pt``.

        Currently persists up to three files, all overwritten on every call:

        - ``schema.json`` (copied from ``self.schema_path``): feature layout
          metadata needed to rebuild the Parquet dataset.
        - ``ns_groups.json`` (copied from ``self.ns_groups_path`` when set
          and the file exists): NS-token grouping used to construct the
          tokenizer. Making a per-ckpt copy lets evaluation environments
          consume the checkpoint without having to ship the original
          project-level ``ns_groups.json``.
        - ``train_config.json`` (serialized from ``self.train_config``):
          full set of training-time hyperparameters. When ``ns_groups.json``
          is copied into ``ckpt_dir``, the ``ns_groups_json`` field is
          rewritten to the bare filename so that ``infer.py`` resolves it
          against ``ckpt_dir`` rather than the original absolute path on
          the training machine.
        """
        os.makedirs(ckpt_dir, exist_ok=True)
        if self.schema_path and os.path.exists(self.schema_path):
            shutil.copy2(self.schema_path, ckpt_dir)

        ns_groups_copied = False
        if self.ns_groups_path and os.path.exists(self.ns_groups_path):
            shutil.copy2(self.ns_groups_path, ckpt_dir)
            ns_groups_copied = True

        if self.train_config:
            import json
            cfg_to_dump = self.train_config
            if ns_groups_copied:
                # Override the stored path to a filename relative to ckpt_dir;
                # infer.py already falls back to `<ckpt_dir>/<basename>` when
                # the recorded path is not absolute, which keeps the ckpt
                # portable across hosts.
                cfg_to_dump = dict(self.train_config)
                cfg_to_dump['ns_groups_json'] = os.path.basename(
                    self.ns_groups_path)
            with open(os.path.join(ckpt_dir, 'train_config.json'), 'w') as f:
                json.dump(cfg_to_dump, f, indent=2)

    def _save_step_checkpoint(
        self,
        global_step: int,
        is_best: bool = False,
        skip_model_file: bool = False,
    ) -> str:
        """Save ``model.pt`` plus sidecar files under a ``global_step`` sub-dir.

        Args:
            global_step: current global step used to name the directory.
            is_best: whether this is a new-best checkpoint.
            skip_model_file: if True, skip writing ``model.pt`` (because the
                caller, e.g. EarlyStopping, has already persisted it to the
                same path). Sidecar files are still (re)written.

        Returns:
            The absolute path of the checkpoint directory.
        """
        dir_name = self._build_step_dir_name(global_step, is_best=is_best)
        ckpt_dir = os.path.join(self.save_dir, dir_name)
        os.makedirs(ckpt_dir, exist_ok=True)
        if not skip_model_file:
            torch.save(self.model.state_dict(), os.path.join(ckpt_dir, "model.pt"))
        self._write_sidecar_files(ckpt_dir)
        logging.info(f"Saved checkpoint to {ckpt_dir}/model.pt")
        return ckpt_dir

    def _remove_old_best_dirs(self) -> None:
        """Delete stale ``*.best_model`` directories so that only the latest
        best checkpoint is kept on disk.
        """
        pattern = os.path.join(self.save_dir, "global_step*.best_model")
        for old_dir in glob.glob(pattern):
            shutil.rmtree(old_dir)
            logging.info(f"Removed old best_model dir: {old_dir}")

    def _batch_to_device(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """Move all tensors in ``batch`` to ``self.device`` (``non_blocking=True``,
        to cooperate with ``pin_memory``). Non-tensor values pass through.
        """
        device_batch: Dict[str, Any] = {}
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                device_batch[k] = v.to(self.device, non_blocking=True)
            else:
                device_batch[k] = v
        return device_batch

    def _handle_validation_result(
        self,
        total_step: int,
        val_auc: float,
        val_logloss: float,
    ) -> None:
        """Wraps :meth:`_handle_validation_result_impl` with EMA swap so the
        checkpoint persisted by EarlyStopping reflects the EMA weights when
        EMA is enabled."""
        if self.ema is not None:
            self.ema.swap_in(self.model)
        try:
            self._handle_validation_result_impl(total_step, val_auc, val_logloss)
        finally:
            if self.ema is not None:
                self.ema.restore(self.model)

    def _handle_validation_result_impl(
        self,
        total_step: int,
        val_auc: float,
        val_logloss: float,
    ) -> None:
        """Persist a new-best checkpoint atomically.

        Flow (ordered to avoid leaving empty sidecar-only directories on disk):

        1. Decide whether ``val_auc`` is *likely* to beat the current best
           using the same threshold as ``EarlyStopping._is_not_improved``,
           so our pre-cleanup and EarlyStopping's internal save decision
           stay in sync.
        2. If unlikely, short-circuit: do nothing on disk. We must NOT
           touch ``self.early_stopping.checkpoint_path`` or call
           ``_write_sidecar_files`` because the target directory may not
           exist yet (sidecar-only dirs would otherwise be created here,
           producing checkpoints with missing ``model.pt``).
        3. If likely, point ``EarlyStopping`` at the canonical
           ``global_stepN.best_model/model.pt`` path, remove any stale
           ``*.best_model`` dirs, then run ``EarlyStopping`` (which writes
           ``model.pt`` when it actually confirms a new best).
        4. Only after ``EarlyStopping`` has confirmed a new best
           (``best_score != old_best``) do we write the sidecar files into
           the freshly-created directory; this is guarded so that a
           razor-close score that tripped ``is_likely_new_best`` but not
           ``EarlyStopping``'s own gate does not create a stray dir.
        """
        old_best = self.early_stopping.best_score
        is_likely_new_best = (
            old_best is None
            or val_auc > old_best + self.early_stopping.delta
        )
        if not is_likely_new_best:
            # No new best anticipated: leave disk untouched. The previous
            # best_model dir (with its model.pt + sidecars) remains valid.
            self.early_stopping(val_auc, self.model, {
                "best_val_AUC": val_auc,
                "best_val_logloss": val_logloss,
            })
            return

        # Point EarlyStopping at the canonical best-model location for this
        # step. Only done on the likely-new-best branch so that a skipped
        # save never leaks the unused path into EarlyStopping state.
        best_dir = os.path.join(
            self.save_dir,
            self._build_step_dir_name(total_step, is_best=True),
        )
        self.early_stopping.checkpoint_path = os.path.join(best_dir, "model.pt")

        # Remove stale best dirs first so EarlyStopping's write is the only
        # I/O needed when a new best is confirmed.
        self._remove_old_best_dirs()

        self.early_stopping(val_auc, self.model, {
            "best_val_AUC": val_auc,
            "best_val_logloss": val_logloss,
        })

        # Write sidecar files only when EarlyStopping actually confirmed a
        # new best and wrote model.pt. If the score tripped our heuristic
        # but EarlyStopping internally declined to save, skip to avoid
        # creating an empty (sidecar-only) checkpoint directory.
        if self.early_stopping.best_score != old_best and os.path.exists(
            self.early_stopping.checkpoint_path
        ):
            self._save_step_checkpoint(
                total_step, is_best=True, skip_model_file=True)

    def train(self) -> None:
        """Main training loop: iterates over epochs, performs step-level and
        epoch-level validation, triggers EarlyStopping and the periodic sparse
        re-initialization strategy.
        """
        print("Start training (PCVRHyFormer)")
        self.model.train()
        total_step = 0

        for epoch in range(1, self.num_epochs + 1):
            train_pbar = tqdm(enumerate(self.train_loader), total=len(self.train_loader),
                              dynamic_ncols=True)
            loss_sum = 0.0

            for step, batch in train_pbar:
                loss = self._train_step(batch)
                total_step += 1
                loss_sum += loss

                if self.writer:
                    self.writer.add_scalar('Loss/train', loss, total_step)

                train_pbar.set_postfix({"loss": f"{loss:.4f}"})

                # Step-level validation (only when eval_every_n_steps > 0).
                if self.eval_every_n_steps > 0 and total_step % self.eval_every_n_steps == 0:
                    logging.info(f"Evaluating at step {total_step}")
                    val_auc, val_logloss = self.evaluate(epoch=epoch)
                    self.model.train()
                    torch.cuda.empty_cache()

                    logging.info(f"Step {total_step} Validation | AUC: {val_auc}, LogLoss: {val_logloss}")

                    if self.writer:
                        self.writer.add_scalar('AUC/valid', val_auc, total_step)
                        self.writer.add_scalar('LogLoss/valid', val_logloss, total_step)

                    self._handle_validation_result(total_step, val_auc, val_logloss)

                    if self.early_stopping.early_stop:
                        logging.info(f"Early stopping at step {total_step}")
                        return

            logging.info(f"Epoch {epoch}, Average Loss: {loss_sum / len(self.train_loader)}")

            val_auc, val_logloss = self.evaluate(epoch=epoch)
            self.model.train()
            torch.cuda.empty_cache()

            logging.info(f"Epoch {epoch} Validation | AUC: {val_auc}, LogLoss: {val_logloss}")

            if self.writer:
                self.writer.add_scalar('AUC/valid', val_auc, total_step)
                self.writer.add_scalar('LogLoss/valid', val_logloss, total_step)

            self._handle_validation_result(total_step, val_auc, val_logloss)

            if self.early_stopping.early_stop:
                logging.info(f"Early stopping at epoch {epoch}")
                break

            # After the configured epoch, reinitialize high-cardinality sparse
            # params (Embeddings) as a form of cold restart to reduce overfit.
            # Reference: KuaiShou Tech., "MultiEpoch: Reusing Training Data
            # for Click-Through Rate Prediction",
            # https://arxiv.org/pdf/2305.19531
            if epoch >= self.reinit_sparse_after_epoch and self.sparse_optimizer is not None:
                # Snapshot Adagrad state per parameter via data_ptr, so state
                # of low-cardinality embeddings can be preserved across rebuild.
                old_state: Dict[int, Any] = {}
                for group in self.sparse_optimizer.param_groups:
                    for p in group['params']:
                        if p.data_ptr() in self.sparse_optimizer.state:
                            old_state[p.data_ptr()] = self.sparse_optimizer.state[p]

                reinit_ptrs = self.model.reinit_high_cardinality_params(self.reinit_cardinality_threshold)
                # Resync EMA shadow for any reinitialized embedding so it does
                # not keep stale pre-reset values.
                if self.ema is not None:
                    self.ema.reset_by_data_ptrs(self.model, reinit_ptrs)
                sparse_params = self.model.get_sparse_params()
                self.sparse_optimizer = torch.optim.Adagrad(
                    sparse_params, lr=self.sparse_lr, weight_decay=self.sparse_weight_decay
                )
                # Restore optimizer state for low-cardinality embeddings only.
                restored = 0
                for p in sparse_params:
                    if p.data_ptr() not in reinit_ptrs and p.data_ptr() in old_state:
                        self.sparse_optimizer.state[p] = old_state[p.data_ptr()]
                        restored += 1
                logging.info(f"Rebuilt Adagrad optimizer after epoch {epoch}, "
                             f"restored optimizer state for {restored} low-cardinality params")

        # After training (or early stop) finishes, optionally measure
        # single-request inference latency. This builds the per-commit
        # latency baseline required by the contest's hard latency cap.
        if self.measure_latency:
            try:
                self.measure_inference_latency(
                    num_warmup=self.latency_warmup,
                    num_iters=self.latency_iters,
                    log_label="Post-training inference latency",
                )
            except Exception as e:
                logging.warning(f"[Latency] measurement skipped due to error: {e}")

    def _make_model_input(self, device_batch: Dict[str, Any]) -> ModelInput:
        """Construct a ``ModelInput`` NamedTuple from a device_batch dict."""
        seq_domains = device_batch['_seq_domains']
        seq_data: Dict[str, torch.Tensor] = {}
        seq_lens: Dict[str, torch.Tensor] = {}
        seq_time_buckets: Dict[str, torch.Tensor] = {}
        for domain in seq_domains:
            seq_data[domain] = device_batch[domain]
            seq_lens[domain] = device_batch[f'{domain}_len']
            B = device_batch[domain].shape[0]
            L = device_batch[domain].shape[2]
            seq_time_buckets[domain] = device_batch.get(
                f'{domain}_time_bucket',
                torch.zeros(B, L, dtype=torch.long, device=self.device))
        return ModelInput(
            user_int_feats=device_batch['user_int_feats'],
            item_int_feats=device_batch['item_int_feats'],
            user_dense_feats=device_batch['user_dense_feats'],
            item_dense_feats=device_batch['item_dense_feats'],
            seq_data=seq_data,
            seq_lens=seq_lens,
            seq_time_buckets=seq_time_buckets,
        )

    def _train_step(self, batch: Dict[str, Any]) -> float:
        """Run a single training step and return the scalar loss value."""
        device_batch = self._batch_to_device(batch)
        label = device_batch['label'].float()

        self.dense_optimizer.zero_grad()
        if self.sparse_optimizer is not None:
            self.sparse_optimizer.zero_grad()

        model_input = self._make_model_input(device_batch)
        with self._autocast():
            logits = self.model(model_input)  # (B, 1)
            logits = logits.squeeze(-1)  # (B,)

            if self.loss_type == 'focal':
                loss = sigmoid_focal_loss(logits, label, alpha=self.focal_alpha, gamma=self.focal_gamma)
            else:
                loss = F.binary_cross_entropy_with_logits(logits, label)
        # backward outside autocast: bf16 doesn't need GradScaler; grads are fp32.
        loss.backward()
        # foreach=False: avoids a PyTorch _foreach_norm CUDA kernel bug observed
        # with certain tensor shapes in this project.
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0, foreach=False)

        self.dense_optimizer.step()
        if self.sparse_optimizer is not None:
            self.sparse_optimizer.step()

        if self.dense_lr_scheduler is not None:
            self.dense_lr_scheduler.step()

        if self.ema is not None:
            self.ema.update(self.model)

        return loss.item()

    def evaluate(self, epoch: Optional[int] = None) -> Tuple[float, float]:
        """Run validation over ``self.valid_loader`` and return ``(AUC, logloss)``.

        NaN predictions (which can arise from exploding gradients) are filtered
        out before computing both metrics. When EMA is enabled, model weights
        are temporarily swapped with the EMA shadow for the duration of eval.
        """
        print("Start Evaluation (PCVRHyFormer) - validation")
        self.model.eval()
        if not epoch:
            epoch = -1

        # EMA: swap to shadow weights for evaluation; restore on exit.
        if self.ema is not None:
            self.ema.swap_in(self.model)
        try:
            return self._evaluate_impl()
        finally:
            if self.ema is not None:
                self.ema.restore(self.model)

    def _evaluate_impl(self) -> Tuple[float, float]:
        """Inner evaluation loop (no EMA bookkeeping)."""
        pbar = tqdm(enumerate(self.valid_loader), total=len(self.valid_loader))

        all_logits_list = []
        all_labels_list = []

        with torch.no_grad():
            for step, batch in pbar:
                logits, labels = self._evaluate_step(batch)
                all_logits_list.append(logits.detach().cpu())
                all_labels_list.append(labels.detach().cpu())

        all_logits = torch.cat(all_logits_list, dim=0)
        all_labels = torch.cat(all_labels_list, dim=0).long()

        # Binary AUC via sklearn.
        probs = torch.sigmoid(all_logits).numpy()
        labels_np = all_labels.numpy()

        # Filter NaN predictions (may appear if gradients explode).
        nan_mask = np.isnan(probs)
        if nan_mask.any():
            n_nan = int(nan_mask.sum())
            logging.warning(f"[Evaluate] {n_nan}/{len(probs)} predictions are NaN, filtering them out")
            valid_mask = ~nan_mask
            probs = probs[valid_mask]
            labels_np = labels_np[valid_mask]

        if len(probs) == 0 or len(np.unique(labels_np)) < 2:
            auc = 0.0
        else:
            auc = float(roc_auc_score(labels_np, probs))

        # Binary logloss (same NaN filtering).
        valid_logits = all_logits[~torch.isnan(all_logits)]
        valid_labels = all_labels[~torch.isnan(all_logits)]
        if len(valid_logits) > 0:
            logloss = F.binary_cross_entropy_with_logits(valid_logits, valid_labels.float()).item()
        else:
            logloss = float('inf')

        return auc, logloss

    def _slice_batch_to_one(self, device_batch: Dict[str, Any]) -> Dict[str, Any]:
        """Return a copy of ``device_batch`` with the leading dim sliced to 1.

        Used by latency measurement to simulate single-request inference.
        Non-tensor values pass through unchanged.
        """
        out: Dict[str, Any] = {}
        for k, v in device_batch.items():
            if isinstance(v, torch.Tensor):
                out[k] = v[:1].contiguous()
            else:
                out[k] = v
        return out

    def measure_inference_latency(
        self,
        num_warmup: int = 50,
        num_iters: int = 200,
        log_label: str = "Inference Latency",
    ) -> Dict[str, float]:
        """Measure model inference latency at batch_size=1.

        Uses CUDA events when available (sub-ms precision after sync),
        otherwise wall-clock ``time.perf_counter``. EMA weights are applied
        for the duration of the measurement when enabled. Returns a dict
        with ``mean / p50 / p95 / p99`` keys (milliseconds).
        """
        import time
        self.model.eval()
        if self.ema is not None:
            self.ema.swap_in(self.model)
        try:
            # Pull the first available batch, slice to B=1.
            first_batch = None
            for batch in self.valid_loader:
                first_batch = batch
                break
            if first_batch is None:
                logging.warning("[Latency] valid_loader is empty; skipping measurement")
                return {'mean': float('nan'), 'p50': float('nan'),
                        'p95': float('nan'), 'p99': float('nan')}

            device_batch = self._batch_to_device(first_batch)
            b1 = self._slice_batch_to_one(device_batch)
            model_input = self._make_model_input(b1)

            use_cuda = torch.cuda.is_available() and 'cuda' in str(self.device)

            with torch.no_grad():
                # Warmup.
                for _ in range(num_warmup):
                    with self._autocast():
                        self.model.predict(model_input)
                if use_cuda:
                    torch.cuda.synchronize()

                # Measure.
                latencies_ms = []
                if use_cuda:
                    for _ in range(num_iters):
                        start_evt = torch.cuda.Event(enable_timing=True)
                        end_evt = torch.cuda.Event(enable_timing=True)
                        start_evt.record()
                        with self._autocast():
                            self.model.predict(model_input)
                        end_evt.record()
                        torch.cuda.synchronize()
                        latencies_ms.append(float(start_evt.elapsed_time(end_evt)))
                else:
                    for _ in range(num_iters):
                        t0 = time.perf_counter()
                        with self._autocast():
                            self.model.predict(model_input)
                        latencies_ms.append((time.perf_counter() - t0) * 1000.0)

            latencies_ms.sort()
            n = len(latencies_ms)
            stats = {
                'mean': sum(latencies_ms) / n,
                'p50': latencies_ms[int(n * 0.50)],
                'p95': latencies_ms[min(n - 1, int(n * 0.95))],
                'p99': latencies_ms[min(n - 1, int(n * 0.99))],
                'min': latencies_ms[0],
                'max': latencies_ms[-1],
            }
            logging.info(
                f"=== {log_label} (B=1, n={n}, warmup={num_warmup}, "
                f"device={self.device}, bf16={self.use_bf16}) ==="
            )
            logging.info(f"  Mean: {stats['mean']:.3f} ms")
            logging.info(f"  P50:  {stats['p50']:.3f} ms")
            logging.info(f"  P95:  {stats['p95']:.3f} ms")
            logging.info(f"  P99:  {stats['p99']:.3f} ms")
            logging.info(f"  Min:  {stats['min']:.3f} ms / Max: {stats['max']:.3f} ms")
            if self.writer is not None:
                for k, v in stats.items():
                    self.writer.add_scalar(f'Latency/{k}_ms', v, 0)
            return stats
        finally:
            if self.ema is not None:
                self.ema.restore(self.model)

    def _evaluate_step(
        self, batch: Dict[str, Any]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run a single validation step and return ``(logits, labels)``."""
        device_batch = self._batch_to_device(batch)
        label = device_batch['label']

        model_input = self._make_model_input(device_batch)
        with self._autocast():
            logits, _ = self.model.predict(model_input)  # (B, 1), (B, D)
        # Cast back to fp32 for downstream sklearn AUC (numpy can't read bf16).
        logits = logits.float().squeeze(-1)  # (B,)

        return logits, label
