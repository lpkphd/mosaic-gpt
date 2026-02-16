"""Data loading for MOSAIC-GPT training.

Streams data from HuggingFace datasets (FineWeb-Edu by default),
tokenizes on-the-fly, and produces fixed-length chunks for training.
"""

import torch
from torch.utils.data import IterableDataset, DataLoader
from transformers import AutoTokenizer
from datasets import load_dataset
from typing import Iterator


class StreamingTextDataset(IterableDataset):
    """Streams tokenized text chunks from a HuggingFace dataset."""

    def __init__(
        self,
        dataset_name: str = "HuggingFaceFW/fineweb-edu",
        tokenizer_name: str = "gpt2",
        seq_len: int = 1024,
        split: str = "train",
        subset: str = "sample-10BT",
    ):
        self.seq_len = seq_len
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.dataset_name = dataset_name
        self.split = split
        self.subset = subset

    def __iter__(self) -> Iterator[dict]:
        ds = load_dataset(
            self.dataset_name,
            name=self.subset,
            split=self.split,
            streaming=True,
        )

        buffer = []
        for example in ds:
            text = example.get("text", "")
            if not text:
                continue
            tokens = self.tokenizer.encode(text)
            buffer.extend(tokens)

            while len(buffer) >= self.seq_len + 1:
                chunk = buffer[: self.seq_len + 1]
                buffer = buffer[self.seq_len:]
                x = torch.tensor(chunk[:-1], dtype=torch.long)
                y = torch.tensor(chunk[1:], dtype=torch.long)
                yield {"input_ids": x, "labels": y}


class WikiText2Dataset(IterableDataset):
    """WikiText-2 for evaluation."""

    def __init__(
        self,
        tokenizer_name: str = "gpt2",
        seq_len: int = 1024,
        split: str = "test",
    ):
        self.seq_len = seq_len
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.split = split

    def __iter__(self) -> Iterator[dict]:
        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split=self.split)
        buffer = []
        for example in ds:
            text = example.get("text", "")
            if not text.strip():
                continue
            tokens = self.tokenizer.encode(text)
            buffer.extend(tokens)

            while len(buffer) >= self.seq_len + 1:
                chunk = buffer[: self.seq_len + 1]
                buffer = buffer[self.seq_len:]
                x = torch.tensor(chunk[:-1], dtype=torch.long)
                y = torch.tensor(chunk[1:], dtype=torch.long)
                yield {"input_ids": x, "labels": y}


def build_train_loader(cfg, num_workers: int = 2) -> DataLoader:
    ds = StreamingTextDataset(
        dataset_name=cfg.training.dataset,
        tokenizer_name=cfg.training.tokenizer,
        seq_len=cfg.training.seq_len,
    )
    return DataLoader(ds, batch_size=cfg.training.batch_size, num_workers=num_workers)


def build_eval_loader(cfg, split: str = "test") -> DataLoader:
    ds = WikiText2Dataset(
        tokenizer_name=cfg.training.tokenizer,
        seq_len=cfg.training.seq_len,
        split=split,
    )
    return DataLoader(ds, batch_size=cfg.training.batch_size)
