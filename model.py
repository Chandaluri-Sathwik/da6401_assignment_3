"""
Transformer architecture for German-to-English machine translation.

This file intentionally uses only basic PyTorch building blocks rather than
torch.nn.MultiheadAttention.
"""

import copy
import math
import os
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import gdown
except ImportError:  # Optional, only needed for hosted checkpoints.
    gdown = None

try:
    import spacy
except ImportError:  # pragma: no cover - spacy is in requirements for the assignment
    spacy = None


# Put your public Google Drive file id here before Gradescope submission, or set
# environment variable A3_CHECKPOINT_GDRIVE_ID. Do not submit the checkpoint file.
CHECKPOINT_GDRIVE_ID = "1Q2yHrVGIu0hzznmk4ATf1O8gsNp2rUZM"
DEFAULT_CHECKPOINT_PATH = "checkpoint.pt"

def _torch_load(path: str):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _load_spacy_model(model_name: str, language: str):
    if spacy is None:
        return None
    try:
        return spacy.load(model_name)
    except OSError:
        return spacy.blank(language)


def scaled_dot_product_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    use_scaling: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute Attention(Q, K, V) = softmax(Q K^T / sqrt(d_k)) V.

    Args:
        Q: shape (..., seq_q, d_k)
        K: shape (..., seq_k, d_k)
        V: shape (..., seq_k, d_v)
        mask: bool tensor broadcastable to (..., seq_q, seq_k).
              True means masked out.

    Returns:
        output: shape (..., seq_q, d_v)
        attn_w: shape (..., seq_q, seq_k)
    """
    d_k = Q.size(-1)
    scores = torch.matmul(Q, K.transpose(-2, -1))
    if use_scaling:
        scores = scores / math.sqrt(d_k)

    if mask is not None:
        scores = scores.masked_fill(mask, float("-inf"))

    attn_w = F.softmax(scores, dim=-1)
    output = torch.matmul(attn_w, V)
    return output, attn_w


def make_src_mask(
    src: torch.Tensor,
    pad_idx: int = 1,
) -> torch.Tensor:
    """
    Build encoder padding mask.

    Args:
        src: shape [batch, src_len]
        pad_idx: index of the <pad> token

    Returns:
        Bool tensor with shape [batch, 1, 1, src_len].
        True means masked out.
    """
    return (src == pad_idx).unsqueeze(1).unsqueeze(2)


def make_tgt_mask(
    tgt: torch.Tensor,
    pad_idx: int = 1,
) -> torch.Tensor:
    """
    Build decoder padding + causal mask.

    Args:
        tgt: shape [batch, tgt_len]
        pad_idx: index of the <pad> token

    Returns:
        Bool tensor with shape [batch, 1, tgt_len, tgt_len].
        True means masked out.
    """
    _, tgt_len = tgt.size()
    pad_mask = (tgt == pad_idx).unsqueeze(1).unsqueeze(2)
    causal_mask = torch.triu(
        torch.ones((tgt_len, tgt_len), device=tgt.device, dtype=torch.bool),
        diagonal=1,
    )
    return pad_mask | causal_mask


class MultiHeadAttention(nn.Module):
    """
    Multi-head attention using projected Q, K, V tensors.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dropout: float = 0.1,
        use_scaling: bool = True,
    ) -> None:
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.use_scaling = use_scaling

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.attn_weights = None

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            query: shape [batch, seq_q, d_model]
            key: shape [batch, seq_k, d_model]
            value: shape [batch, seq_k, d_model]
            mask: optional bool tensor broadcastable to
                  [batch, num_heads, seq_q, seq_k]

        Returns:
            Tensor with shape [batch, seq_q, d_model].
        """
        batch_size = query.size(0)
        seq_q = query.size(1)

        Q = self.W_q(query)
        K = self.W_k(key)
        V = self.W_v(value)

        Q = Q.view(batch_size, seq_q, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, key.size(1), self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, value.size(1), self.num_heads, self.d_k).transpose(1, 2)

        attn_output, self.attn_weights = scaled_dot_product_attention(
            Q,
            K,
            V,
            mask,
            use_scaling=self.use_scaling,
        )
        attn_output = self.dropout(attn_output)

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_q, self.d_model)
        return self.W_o(attn_output)


class PositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding from "Attention Is All You Need".
    """

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float)
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[: pe[:, 1::2].size(1)])

        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: shape [batch, seq_len, d_model]

        Returns:
            Tensor with the same shape.
        """
        x = x + self.pe[:, : x.size(1), :].to(dtype=x.dtype)
        return self.dropout(x)


class LearnedPositionalEncoding(nn.Module):
    """
    Learned positional embeddings for the report ablation.
    """

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000) -> None:
        super().__init__()
        self.position_embedding = nn.Embedding(max_len, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        positions = torch.arange(x.size(1), device=x.device).unsqueeze(0)
        x = x + self.position_embedding(positions)
        return self.dropout(x)


class PositionwiseFeedForward(nn.Module):
    """
    FFN(x) = Linear(ReLU(Linear(x))).
    """

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.dropout(F.relu(self.linear1(x))))


class EncoderLayer(nn.Module):
    """
    One Transformer encoder layer with post-layer normalization.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        dropout: float = 0.1,
        use_attention_scaling: bool = True,
    ) -> None:
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout, use_attention_scaling)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, src_mask: torch.Tensor) -> torch.Tensor:
        attn_out = self.self_attn(x, x, x, src_mask)
        x = self.norm1(x + self.dropout1(attn_out))

        ffn_out = self.feed_forward(x)
        x = self.norm2(x + self.dropout2(ffn_out))
        return x


class DecoderLayer(nn.Module):
    """
    One Transformer decoder layer with masked self-attention, cross-attention,
    and a feed-forward network.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        dropout: float = 0.1,
        use_attention_scaling: bool = True,
    ) -> None:
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout, use_attention_scaling)
        self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout, use_attention_scaling)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        memory: torch.Tensor,
        src_mask: torch.Tensor,
        tgt_mask: torch.Tensor,
    ) -> torch.Tensor:
        self_attn_out = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout1(self_attn_out))

        cross_attn_out = self.cross_attn(x, memory, memory, src_mask)
        x = self.norm2(x + self.dropout2(cross_attn_out))

        ffn_out = self.feed_forward(x)
        x = self.norm3(x + self.dropout3(ffn_out))
        return x


class Encoder(nn.Module):
    """Stack of N identical EncoderLayer modules with final LayerNorm."""

    def __init__(self, layer: EncoderLayer, N: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList([copy.deepcopy(layer) for _ in range(N)])
        self.norm = nn.LayerNorm(layer.norm2.normalized_shape)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class Decoder(nn.Module):
    """Stack of N identical DecoderLayer modules with final LayerNorm."""

    def __init__(self, layer: DecoderLayer, N: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList([copy.deepcopy(layer) for _ in range(N)])
        self.norm = nn.LayerNorm(layer.norm3.normalized_shape)

    def forward(
        self,
        x: torch.Tensor,
        memory: torch.Tensor,
        src_mask: torch.Tensor,
        tgt_mask: torch.Tensor,
    ) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)


class Transformer(nn.Module):
    """
    Full encoder-decoder Transformer for sequence-to-sequence tasks.
    """

    def __init__(
        self,
        src_vocab_size: int = 10000,
        tgt_vocab_size: int = 10000,
        d_model: int = 512,
        N: int = 6,
        num_heads: int = 8,
        d_ff: int = 2048,
        dropout: float = 0.1,
        checkpoint_path: str = None,
        use_attention_scaling: bool = True,
        positional_encoding_type: str = "sinusoidal",
    ) -> None:
        super().__init__()
        if checkpoint_path is None:
            checkpoint_path = DEFAULT_CHECKPOINT_PATH

        if checkpoint_path is not None and not os.path.exists(checkpoint_path):
            drive_id = CHECKPOINT_GDRIVE_ID
            if drive_id and gdown is not None:
                gdown.download(id=drive_id, output=checkpoint_path, quiet=False)

        if checkpoint_path is not None and os.path.exists(checkpoint_path):
            checkpoint = _torch_load(checkpoint_path)
            model_config = checkpoint.get("model_config")
            if model_config:
                src_vocab_size = model_config.get("src_vocab_size", src_vocab_size)
                tgt_vocab_size = model_config.get("tgt_vocab_size", tgt_vocab_size)
                d_model = model_config.get("d_model", d_model)
                N = model_config.get("N", N)
                num_heads = model_config.get("num_heads", num_heads)
                d_ff = model_config.get("d_ff", d_ff)
                dropout = model_config.get("dropout", dropout)
                use_attention_scaling = model_config.get(
                    "use_attention_scaling",
                    use_attention_scaling,
                )
                positional_encoding_type = model_config.get(
                    "positional_encoding_type",
                    positional_encoding_type,
                )

        self.src_vocab_size = src_vocab_size
        self.tgt_vocab_size = tgt_vocab_size
        self.d_model = d_model
        self.N = N
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.dropout_p = dropout
        self.use_attention_scaling = use_attention_scaling
        self.positional_encoding_type = positional_encoding_type
        self.model_config = {
            "src_vocab_size": src_vocab_size,
            "tgt_vocab_size": tgt_vocab_size,
            "d_model": d_model,
            "N": N,
            "num_heads": num_heads,
            "d_ff": d_ff,
            "dropout": dropout,
            "use_attention_scaling": use_attention_scaling,
            "positional_encoding_type": positional_encoding_type,
        }

        self.src_embed = nn.Embedding(src_vocab_size, d_model)
        self.tgt_embed = nn.Embedding(tgt_vocab_size, d_model)
        if positional_encoding_type == "sinusoidal":
            self.positional_encoding = PositionalEncoding(d_model, dropout)
        elif positional_encoding_type == "learned":
            self.positional_encoding = LearnedPositionalEncoding(d_model, dropout)
        else:
            raise ValueError("positional_encoding_type must be 'sinusoidal' or 'learned'.")

        encoder_layer = EncoderLayer(d_model, num_heads, d_ff, dropout, use_attention_scaling)
        decoder_layer = DecoderLayer(d_model, num_heads, d_ff, dropout, use_attention_scaling)
        self.encoder = Encoder(encoder_layer, N)
        self.decoder = Decoder(decoder_layer, N)
        self.generator = nn.Linear(d_model, tgt_vocab_size)

        self._reset_parameters()

        self.src_vocab = None
        self.tgt_vocab = None
        self.src_tokenizer = _load_spacy_model("de_core_news_sm", "de")
        self.pad_idx = 1
        self.sos_idx = 2
        self.eos_idx = 3
        self.max_decode_len = 100

        if checkpoint_path is not None and os.path.exists(checkpoint_path):
            checkpoint = _torch_load(checkpoint_path)
            state_dict = checkpoint.get("model_state_dict", checkpoint)
            self.load_state_dict(state_dict)
            self.src_vocab = checkpoint.get("src_vocab")
            self.tgt_vocab = checkpoint.get("tgt_vocab")
            self.pad_idx = checkpoint.get("pad_idx", self.pad_idx)
            self.sos_idx = checkpoint.get("sos_idx", self.sos_idx)
            self.eos_idx = checkpoint.get("eos_idx", self.eos_idx)
            self.max_decode_len = checkpoint.get("max_decode_len", self.max_decode_len)

    def _reset_parameters(self) -> None:
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def encode(
        self,
        src: torch.Tensor,
        src_mask: torch.Tensor,
    ) -> torch.Tensor:
        src_emb = self.src_embed(src) * math.sqrt(self.d_model)
        src_emb = self.positional_encoding(src_emb)
        return self.encoder(src_emb, src_mask)

    def decode(
        self,
        memory: torch.Tensor,
        src_mask: torch.Tensor,
        tgt: torch.Tensor,
        tgt_mask: torch.Tensor,
    ) -> torch.Tensor:
        tgt_emb = self.tgt_embed(tgt) * math.sqrt(self.d_model)
        tgt_emb = self.positional_encoding(tgt_emb)
        decoded = self.decoder(tgt_emb, memory, src_mask, tgt_mask)
        return self.generator(decoded)

    def forward(
        self,
        src: torch.Tensor,
        tgt: torch.Tensor,
        src_mask: torch.Tensor,
        tgt_mask: torch.Tensor,
    ) -> torch.Tensor:
        memory = self.encode(src, src_mask)
        return self.decode(memory, src_mask, tgt, tgt_mask)

    def infer(self, src_sentence: str) -> str:
        """
        Text-level inference depends on the project's tokenizers and vocab objects.
        If src_vocab/tgt_vocab are attached to the model, this method performs
        greedy decoding. Otherwise, call train.greedy_decode with tokenized input.
        """
        if self.src_vocab is None or self.tgt_vocab is None:
            return ""

        def token_to_idx(vocab, token: str) -> int:
            if hasattr(vocab, "lookup_indices"):
                return vocab.lookup_indices([token])[0]
            if hasattr(vocab, "stoi"):
                return vocab.stoi.get(token, vocab.stoi.get("<unk>", 0))
            if isinstance(vocab, dict):
                return vocab.get(token, vocab.get("<unk>", 0))
            try:
                return vocab[token]
            except Exception:
                return token_to_idx(vocab, "<unk>") if token != "<unk>" else 0

        def idx_to_token(vocab, idx: int) -> str:
            if hasattr(vocab, "lookup_token"):
                return vocab.lookup_token(idx)
            if hasattr(vocab, "itos"):
                return vocab.itos[idx]
            if isinstance(vocab, dict):
                reverse_vocab = {v: k for k, v in vocab.items()}
                return reverse_vocab[idx]
            return str(idx)

        tokenizer = getattr(self, "src_tokenizer", None)
        if tokenizer is None:
            src_tokens = src_sentence.lower().split()
        else:
            src_tokens = [tok.text.lower() if hasattr(tok, "text") else str(tok).lower()
                          for tok in tokenizer(src_sentence)]

        sos_idx = getattr(self, "sos_idx", token_to_idx(self.tgt_vocab, "<sos>"))
        eos_idx = getattr(self, "eos_idx", token_to_idx(self.tgt_vocab, "<eos>"))
        pad_idx = getattr(self, "pad_idx", token_to_idx(self.src_vocab, "<pad>"))
        max_len = getattr(self, "max_decode_len", 100)
        device = next(self.parameters()).device

        src_ids = [token_to_idx(self.src_vocab, "<sos>")]
        src_ids.extend(token_to_idx(self.src_vocab, token) for token in src_tokens)
        src_ids.append(token_to_idx(self.src_vocab, "<eos>"))

        self.eval()
        with torch.no_grad():
            src = torch.tensor(src_ids, dtype=torch.long, device=device).unsqueeze(0)
            src_mask = make_src_mask(src, pad_idx)
            memory = self.encode(src, src_mask)
            ys = torch.tensor([[sos_idx]], dtype=torch.long, device=device)

            for _ in range(max_len - 1):
                tgt_mask = make_tgt_mask(ys, pad_idx)
                logits = self.decode(memory, src_mask, ys, tgt_mask)
                next_token = torch.argmax(logits[:, -1, :], dim=-1).item()
                ys = torch.cat(
                    [ys, torch.tensor([[next_token]], dtype=torch.long, device=device)],
                    dim=1,
                )
                if next_token == eos_idx:
                    break

        output_tokens = []
        skip_tokens = {"<sos>", "<eos>", "<pad>"}
        for idx in ys.squeeze(0).tolist():
            token = idx_to_token(self.tgt_vocab, idx)
            if token not in skip_tokens:
                output_tokens.append(token)
        return " ".join(output_tokens)
