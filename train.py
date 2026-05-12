"""
Training, inference, checkpointing, and BLEU evaluation for Assignment 3.
"""

from __future__ import annotations

import math
import os
from collections import Counter
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import EOS_IDX, PAD_IDX, SOS_IDX, Multi30kDataset, make_collate_fn
from lr_scheduler import NoamScheduler
from model import Transformer, make_src_mask, make_tgt_mask


class LabelSmoothingLoss(nn.Module):
    """
    KL-divergence label smoothing loss.

    The target class receives probability 1 - smoothing. Other non-pad
    classes share the smoothing mass. Pad targets are ignored.
    """

    def __init__(self, vocab_size: int, pad_idx: int, smoothing: float = 0.1) -> None:
        super().__init__()
        if not 0.0 <= smoothing < 1.0:
            raise ValueError("smoothing must be in [0, 1).")

        self.vocab_size = vocab_size
        self.pad_idx = pad_idx
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: shape [batch * tgt_len, vocab_size]
            target: shape [batch * tgt_len]
        """
        log_probs = F.log_softmax(logits, dim=-1)
        non_pad_mask = target.ne(self.pad_idx)

        if non_pad_mask.sum() == 0:
            return logits.new_tensor(0.0)

        with torch.no_grad():
            true_dist = torch.zeros_like(log_probs)
            smooth_value = self.smoothing / max(self.vocab_size - 2, 1)
            true_dist.fill_(smooth_value)
            true_dist[:, self.pad_idx] = 0.0
            true_dist.scatter_(1, target.unsqueeze(1), self.confidence)
            true_dist.masked_fill_(~non_pad_mask.unsqueeze(1), 0.0)

        loss = F.kl_div(log_probs, true_dist, reduction="sum")
        return loss / non_pad_mask.sum()


def run_epoch(
    data_iter,
    model: Transformer,
    loss_fn: nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    scheduler=None,
    epoch_num: int = 0,
    is_train: bool = True,
    device: str = "cpu",
    wandb_run=None,
    log_prefix: str = "",
    log_every: int = 100,
    global_step_start: int = 0,
    grad_log_steps: int = 1000,
) -> float:
    """
    Run one epoch of training or evaluation.
    """
    model.to(device)
    model.train(is_train)

    total_loss = 0.0
    total_tokens = 0
    total_confidence = 0.0
    confidence_batches = 0
    global_step = global_step_start
    iterator = tqdm(data_iter, desc=f"{'train' if is_train else 'eval'} {epoch_num}", leave=False)

    for batch_idx, (src, tgt) in enumerate(iterator):
        src = src.to(device)
        tgt = tgt.to(device)

        tgt_input = tgt[:, :-1]
        tgt_output = tgt[:, 1:]

        src_mask = make_src_mask(src, PAD_IDX)
        tgt_mask = make_tgt_mask(tgt_input, PAD_IDX)

        if is_train:
            if optimizer is None:
                raise ValueError("optimizer must be provided when is_train=True.")
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(is_train):
            logits = model(src, tgt_input, src_mask, tgt_mask)
            loss = loss_fn(
                logits.reshape(-1, logits.size(-1)),
                tgt_output.reshape(-1),
            )

            if is_train:
                loss.backward()
                if wandb_run is not None and global_step < grad_log_steps:
                    wandb_run.log(
                        _gradient_norm_metrics(model, prefix=f"{log_prefix}grad_norm/"),
                        step=global_step,
                    )
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()

        ntokens = tgt_output.ne(PAD_IDX).sum().item()
        total_loss += loss.item() * ntokens
        total_tokens += ntokens
        batch_confidence = _prediction_confidence(logits.detach(), tgt_output)
        total_confidence += batch_confidence
        confidence_batches += 1
        iterator.set_postfix(loss=loss.item())

        if wandb_run is not None and (batch_idx % log_every == 0):
            metrics = {
                f"{log_prefix}batch_loss": loss.item(),
                f"{log_prefix}prediction_confidence": batch_confidence,
            }
            if optimizer is not None:
                metrics[f"{log_prefix}lr"] = optimizer.param_groups[0]["lr"]
            wandb_run.log(metrics, step=global_step)

        global_step += 1

    avg_confidence = total_confidence / max(confidence_batches, 1)
    if wandb_run is not None:
        wandb_run.log(
            {
                f"{log_prefix}epoch_loss": total_loss / max(total_tokens, 1),
                f"{log_prefix}epoch_prediction_confidence": avg_confidence,
            },
            step=global_step,
        )

    return total_loss / max(total_tokens, 1)


def _prediction_confidence(logits: torch.Tensor, target: torch.Tensor) -> float:
    probs = F.softmax(logits, dim=-1)
    token_probs = probs.gather(-1, target.unsqueeze(-1)).squeeze(-1)
    mask = target.ne(PAD_IDX)
    if mask.sum() == 0:
        return 0.0
    return token_probs.masked_select(mask).mean().item()


def _gradient_norm_metrics(model: Transformer, prefix: str = "grad_norm/") -> dict[str, float]:
    q_norms = []
    k_norms = []
    for module in model.modules():
        if hasattr(module, "W_q") and module.W_q.weight.grad is not None:
            q_norms.append(module.W_q.weight.grad.detach().norm().item())
        if hasattr(module, "W_k") and module.W_k.weight.grad is not None:
            k_norms.append(module.W_k.weight.grad.detach().norm().item())

    metrics = {}
    if q_norms:
        metrics[f"{prefix}query_weights"] = sum(q_norms) / len(q_norms)
    if k_norms:
        metrics[f"{prefix}key_weights"] = sum(k_norms) / len(k_norms)
    return metrics


def greedy_decode(
    model: Transformer,
    src: torch.Tensor,
    src_mask: torch.Tensor,
    max_len: int,
    start_symbol: int,
    end_symbol: int,
    device: str = "cpu",
) -> torch.Tensor:
    """
    Generate one translation with greedy autoregressive decoding.
    """
    model.eval()
    src = src.to(device)
    src_mask = src_mask.to(device)

    with torch.no_grad():
        memory = model.encode(src, src_mask)
        ys = torch.tensor([[start_symbol]], dtype=torch.long, device=device)

        for _ in range(max_len - 1):
            tgt_mask = make_tgt_mask(ys, PAD_IDX)
            logits = model.decode(memory, src_mask, ys, tgt_mask)
            next_word = torch.argmax(logits[:, -1, :], dim=-1).item()
            ys = torch.cat(
                [ys, torch.tensor([[next_word]], dtype=torch.long, device=device)],
                dim=1,
            )
            if next_word == end_symbol:
                break

    return ys


def _lookup_token(vocab, idx: int) -> str:
    try:
        if hasattr(vocab, "lookup_token"):
            return vocab.lookup_token(idx)
        if hasattr(vocab, "itos"):
            return vocab.itos[idx]
        if isinstance(vocab, dict):
            reverse = {v: k for k, v in vocab.items()}
            return reverse[idx]
    except (IndexError, KeyError):
        return "<unk>"
    return str(idx)


def _tokens_from_indices(indices: list[int], vocab) -> list[str]:
    tokens = []
    for idx in indices:
        token = _lookup_token(vocab, idx)
        if token == "<eos>":
            break
        if token not in {"<sos>", "<pad>", "<unk>"}:
            tokens.append(token)
    return tokens


def _ngram_counts(tokens: list[str], n: int) -> Counter:
    return Counter(tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1))


def _corpus_bleu(predictions: list[list[str]], references: list[list[str]]) -> float:
    if not predictions:
        return 0.0

    pred_len = sum(len(pred) for pred in predictions)
    ref_len = sum(len(ref) for ref in references)
    if pred_len == 0:
        return 0.0

    precisions = []
    for n in range(1, 5):
        clipped = 0
        total = 0
        for pred, ref in zip(predictions, references):
            pred_counts = _ngram_counts(pred, n)
            ref_counts = _ngram_counts(ref, n)
            total += sum(pred_counts.values())
            clipped += sum(min(count, ref_counts[ngram]) for ngram, count in pred_counts.items())

        # Add-one smoothing keeps short/early models from collapsing to BLEU 0.
        precisions.append((clipped + 1.0) / (total + 1.0))

    geo_mean = math.exp(sum(math.log(p) for p in precisions) / 4.0)
    brevity_penalty = 1.0 if pred_len > ref_len else math.exp(1.0 - ref_len / pred_len)
    return 100.0 * brevity_penalty * geo_mean


def evaluate_bleu(
    model: Transformer,
    test_dataloader: DataLoader,
    tgt_vocab,
    device: str = "cpu",
    max_len: int = 100,
) -> float:
    """
    Evaluate translation quality with corpus-level BLEU score.
    """
    model.to(device)
    model.eval()

    predictions = []
    references = []

    with torch.no_grad():
        for src_batch, tgt_batch in tqdm(test_dataloader, desc="bleu", leave=False):
            for src, tgt in zip(src_batch, tgt_batch):
                src = src.unsqueeze(0).to(device)
                src_mask = make_src_mask(src, PAD_IDX)
                pred = greedy_decode(
                    model,
                    src,
                    src_mask,
                    max_len=max_len,
                    start_symbol=SOS_IDX,
                    end_symbol=EOS_IDX,
                    device=device,
                )
                predictions.append(_tokens_from_indices(pred.squeeze(0).tolist(), tgt_vocab))
                references.append(_tokens_from_indices(tgt.tolist(), tgt_vocab))

    return _corpus_bleu(predictions, references)


def log_attention_head_heatmaps(
    model: Transformer,
    sample_batch: tuple[torch.Tensor, torch.Tensor],
    src_vocab,
    wandb_run,
    device: str = "cpu",
    prefix: str = "attention/",
) -> None:
    """
    Log one heatmap per head from the final encoder layer.
    """
    if wandb_run is None:
        return

    try:
        import matplotlib.pyplot as plt
        import wandb
    except ImportError:
        return

    src_batch, _ = sample_batch
    src = src_batch[:1].to(device)
    src_mask = make_src_mask(src, PAD_IDX)

    model.eval()
    with torch.no_grad():
        model.encode(src, src_mask)

    attn = model.encoder.layers[-1].self_attn.attn_weights
    if attn is None:
        return

    attn = attn[0].detach().cpu()
    src_tokens = _tokens_from_indices(src.squeeze(0).detach().cpu().tolist(), src_vocab)
    if not src_tokens:
        src_tokens = [str(idx) for idx in src.squeeze(0).detach().cpu().tolist()]

    images = {}
    for head_idx in range(attn.size(0)):
        matrix = attn[head_idx, : len(src_tokens), : len(src_tokens)]
        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(matrix, cmap="viridis", aspect="auto")
        ax.set_xticks(range(len(src_tokens)))
        ax.set_yticks(range(len(src_tokens)))
        ax.set_xticklabels(src_tokens, rotation=90)
        ax.set_yticklabels(src_tokens)
        ax.set_title(f"Encoder Head {head_idx}")
        fig.colorbar(im, ax=ax)
        fig.tight_layout()
        images[f"{prefix}encoder_head_{head_idx}"] = wandb.Image(fig)
        plt.close(fig)

    wandb_run.log(images)


def save_checkpoint(
    model: Transformer,
    optimizer: torch.optim.Optimizer,
    scheduler,
    epoch: int,
    path: str = "checkpoint.pt",
) -> None:
    """
    Save model + optimizer + scheduler state to disk.
    """
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)

    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
            "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
            "model_config": getattr(model, "model_config", None),
            "src_vocab": getattr(model, "src_vocab", None),
            "tgt_vocab": getattr(model, "tgt_vocab", None),
            "pad_idx": PAD_IDX,
            "sos_idx": SOS_IDX,
            "eos_idx": EOS_IDX,
            "max_decode_len": getattr(model, "max_decode_len", 100),
        },
        path,
    )


def load_checkpoint(
    path: str,
    model: Transformer,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler=None,
) -> int:
    """
    Restore model and optionally optimizer/scheduler state.
    """
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None and checkpoint.get("optimizer_state_dict") is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    if scheduler is not None and checkpoint.get("scheduler_state_dict") is not None:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    return int(checkpoint.get("epoch", 0))


def run_training_experiment(
    config_overrides: Optional[dict] = None,
    run_name: Optional[str] = None,
    group: Optional[str] = None,
) -> None:
    """
    Train a Transformer on Multi30k and log metrics to W&B when available.
    """
    try:
        import wandb
    except ImportError:
        wandb = None

    config = {
        "batch_size": 64,
        "num_epochs": 10,
        "d_model": 512,
        "N": 4,
        "num_heads": 8,
        "d_ff": 2048,
        "dropout": 0.1,
        "warmup_steps": 4000,
        "min_freq": 2,
        "max_len": 100,
        "checkpoint_path": "checkpoint.pt",
        "use_wandb": True,
        "use_noam": True,
        "fixed_lr": 1e-4,
        "label_smoothing": 0.1,
        "use_attention_scaling": True,
        "positional_encoding_type": "sinusoidal",
        "log_every": 100,
        "grad_log_steps": 1000,
        "log_attention_heatmaps": False,
    }
    if config_overrides:
        config.update(config_overrides)

    run = (
        wandb.init(project="da6401-a3", config=config, name=run_name, group=group)
        if wandb is not None and config["use_wandb"]
        else None
    )
    if run is not None:
        config.update(dict(run.config))

    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_data = Multi30kDataset(
        "train",
        min_freq=config["min_freq"],
        max_len=config["max_len"],
    )
    train_data.build_vocab()
    train_data.process_data()

    val_data = Multi30kDataset(
        "validation",
        src_vocab=train_data.src_vocab,
        tgt_vocab=train_data.tgt_vocab,
        max_len=config["max_len"],
    )
    val_data.process_data()

    test_data = Multi30kDataset(
        "test",
        src_vocab=train_data.src_vocab,
        tgt_vocab=train_data.tgt_vocab,
        max_len=config["max_len"],
    )
    test_data.process_data()

    collate = make_collate_fn(PAD_IDX)
    train_loader = DataLoader(
        train_data,
        batch_size=config["batch_size"],
        shuffle=True,
        collate_fn=collate,
    )
    val_loader = DataLoader(
        val_data,
        batch_size=config["batch_size"],
        shuffle=False,
        collate_fn=collate,
    )
    test_loader = DataLoader(
        test_data,
        batch_size=1,
        shuffle=False,
        collate_fn=collate,
    )

    model = Transformer(
        src_vocab_size=len(train_data.src_vocab),
        tgt_vocab_size=len(train_data.tgt_vocab),
        d_model=config["d_model"],
        N=config["N"],
        num_heads=config["num_heads"],
        d_ff=config["d_ff"],
        dropout=config["dropout"],
        use_attention_scaling=config["use_attention_scaling"],
        positional_encoding_type=config["positional_encoding_type"],
    ).to(device)
    model.src_vocab = train_data.src_vocab
    model.tgt_vocab = train_data.tgt_vocab
    model.src_tokenizer = train_data.src_tokenizer
    model.pad_idx = PAD_IDX
    model.sos_idx = SOS_IDX
    model.eos_idx = EOS_IDX
    model.max_decode_len = config["max_len"]

    base_lr = 1.0 if config["use_noam"] else config["fixed_lr"]
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=base_lr,
        betas=(0.9, 0.98),
        eps=1e-9,
    )
    scheduler = (
        NoamScheduler(
            optimizer,
            d_model=config["d_model"],
            warmup_steps=config["warmup_steps"],
        )
        if config["use_noam"]
        else None
    )
    loss_fn = LabelSmoothingLoss(
        vocab_size=len(train_data.tgt_vocab),
        pad_idx=PAD_IDX,
        smoothing=config["label_smoothing"],
    )

    best_val_loss = float("inf")
    global_step = 0
    for epoch in range(config["num_epochs"]):
        train_loss = run_epoch(
            train_loader,
            model,
            loss_fn,
            optimizer,
            scheduler,
            epoch_num=epoch,
            is_train=True,
            device=device,
            wandb_run=run,
            log_prefix="train/",
            log_every=config["log_every"],
            global_step_start=global_step,
            grad_log_steps=config["grad_log_steps"],
        )
        global_step += len(train_loader)
        val_loss = run_epoch(
            val_loader,
            model,
            loss_fn,
            optimizer=None,
            scheduler=None,
            epoch_num=epoch,
            is_train=False,
            device=device,
            wandb_run=run,
            log_prefix="val/",
            log_every=config["log_every"],
            global_step_start=global_step,
            grad_log_steps=0,
        )
        global_step += len(val_loader)

        metrics = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "lr": optimizer.param_groups[0]["lr"],
            "use_noam": config["use_noam"],
            "label_smoothing": config["label_smoothing"],
        }
        if run is not None:
            run.log(metrics, step=global_step)
        print(metrics)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(model, optimizer, scheduler, epoch, config["checkpoint_path"])

    if config["log_attention_heatmaps"]:
        sample_batch = next(iter(val_loader))
        log_attention_head_heatmaps(
            model,
            sample_batch,
            train_data.src_vocab,
            run,
            device=device,
        )

    bleu = evaluate_bleu(model, test_loader, train_data.tgt_vocab, device=device)
    if run is not None:
        run.log({"test_bleu": bleu})
        run.finish()
    print({"test_bleu": bleu})


if __name__ == "__main__":
    run_training_experiment()
