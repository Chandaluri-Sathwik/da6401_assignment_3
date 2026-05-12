from collections import Counter
from typing import Callable, Iterable, Optional

import spacy
import torch
from datasets import load_dataset
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset


UNK_IDX = 0
PAD_IDX = 1
SOS_IDX = 2
EOS_IDX = 3
SPECIAL_TOKENS = ["<unk>", "<pad>", "<sos>", "<eos>"]


class Vocab:
    def __init__(self, tokens: Iterable[str], min_freq: int = 2) -> None:
        counter = Counter(tokens)
        self.itos = list(SPECIAL_TOKENS)
        self.stoi = {token: idx for idx, token in enumerate(self.itos)}

        for token, freq in sorted(counter.items()):
            if freq >= min_freq and token not in self.stoi:
                self.stoi[token] = len(self.itos)
                self.itos.append(token)

    def __len__(self) -> int:
        return len(self.itos)

    def __getitem__(self, token: str) -> int:
        return self.stoi.get(token, UNK_IDX)

    def lookup_token(self, idx: int) -> str:
        return self.itos[idx]

    def lookup_indices(self, tokens: list[str]) -> list[int]:
        return [self[token] for token in tokens]


def _load_spacy_model(model_name: str, language: str):
    try:
        return spacy.load(model_name)
    except OSError:
        return spacy.blank(language)


def _extract_pair(example: dict) -> tuple[str, str]:
    if "de" in example and "en" in example:
        return example["de"], example["en"]
    if "translation" in example:
        translation = example["translation"]
        return translation["de"], translation["en"]
    raise KeyError("Expected Multi30k example to contain 'de'/'en' or 'translation'.")


class Multi30kDataset(Dataset):
    def __init__(
        self,
        split: str = "train",
        src_vocab: Optional[Vocab] = None,
        tgt_vocab: Optional[Vocab] = None,
        min_freq: int = 2,
        max_len: Optional[int] = None,
    ) -> None:
        """
        Load Multi30k and prepare German/English tokenizers.

        German is the source language and English is the target language.
        Build vocabularies only on the train split, then pass them to
        validation/test datasets to avoid data leakage.
        """
        self.split = split
        self.min_freq = min_freq
        self.max_len = max_len
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab

        self.dataset = load_dataset("bentrevett/multi30k", split=split)
        self.src_tokenizer = _load_spacy_model("de_core_news_sm", "de")
        self.tgt_tokenizer = _load_spacy_model("en_core_web_sm", "en")
        self.data: list[tuple[torch.Tensor, torch.Tensor]] = []

    def _tokenize_src(self, sentence: str) -> list[str]:
        return [tok.text.lower() for tok in self.src_tokenizer(sentence)]

    def _tokenize_tgt(self, sentence: str) -> list[str]:
        return [tok.text.lower() for tok in self.tgt_tokenizer(sentence)]

    def build_vocab(self) -> None:
        """
        Build source and target vocabularies from this dataset's examples.
        Call this on the training split only.
        """
        src_tokens = []
        tgt_tokens = []

        for example in self.dataset:
            src_sentence, tgt_sentence = _extract_pair(example)
            src_tokens.extend(self._tokenize_src(src_sentence))
            tgt_tokens.extend(self._tokenize_tgt(tgt_sentence))

        self.src_vocab = Vocab(src_tokens, min_freq=self.min_freq)
        self.tgt_vocab = Vocab(tgt_tokens, min_freq=self.min_freq)

    def _numericalize(self, tokens: list[str], vocab: Vocab) -> torch.Tensor:
        if self.max_len is not None:
            tokens = tokens[: self.max_len - 2]
        ids = [SOS_IDX]
        ids.extend(vocab[token] for token in tokens)
        ids.append(EOS_IDX)
        return torch.tensor(ids, dtype=torch.long)

    def process_data(self) -> None:
        """
        Tokenize and convert all examples into integer tensors.
        """
        if self.src_vocab is None or self.tgt_vocab is None:
            raise ValueError("Call build_vocab() first or pass src_vocab/tgt_vocab.")

        processed = []
        for example in self.dataset:
            src_sentence, tgt_sentence = _extract_pair(example)
            src_tokens = self._tokenize_src(src_sentence)
            tgt_tokens = self._tokenize_tgt(tgt_sentence)
            src_tensor = self._numericalize(src_tokens, self.src_vocab)
            tgt_tensor = self._numericalize(tgt_tokens, self.tgt_vocab)
            processed.append((src_tensor, tgt_tensor))

        self.data = processed

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.data[idx]


def collate_fn(
    batch: list[tuple[torch.Tensor, torch.Tensor]],
    pad_idx: int = PAD_IDX,
) -> tuple[torch.Tensor, torch.Tensor]:
    src_batch, tgt_batch = zip(*batch)
    src_batch = pad_sequence(src_batch, batch_first=True, padding_value=pad_idx)
    tgt_batch = pad_sequence(tgt_batch, batch_first=True, padding_value=pad_idx)
    return src_batch, tgt_batch


def make_collate_fn(pad_idx: int = PAD_IDX) -> Callable:
    return lambda batch: collate_fn(batch, pad_idx=pad_idx)
