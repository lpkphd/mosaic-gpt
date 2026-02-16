"""Evaluation benchmarks for MOSAIC-GPT.

Implements four standard LLM benchmarks:
  - WikiText-2 perplexity
  - LAMBADA last-word prediction accuracy
  - HellaSwag commonsense reasoning accuracy
  - ARC-Easy science QA accuracy

Each benchmark loads its dataset from HuggingFace, runs evaluation,
and returns a results dict.
"""

import math
from typing import Optional

import torch
import torch.nn.functional as F
from torch.amp import autocast
from transformers import AutoTokenizer
from datasets import load_dataset

from mosaic.model import MosaicGPT


@torch.no_grad()
def evaluate_wikitext2(
    model: MosaicGPT,
    device: torch.device,
    tokenizer: Optional[AutoTokenizer] = None,
    seq_len: int = 1024,
    batch_size: int = 8,
) -> dict:
    """Compute perplexity on WikiText-2 test set.

    Uses a sliding window over the full concatenated test corpus, computing
    cross-entropy loss per token. Returns exp(mean NLL).
    """
    model.eval()
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained("gpt2")

    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join([ex["text"] for ex in ds if ex["text"].strip()])
    token_ids = tokenizer.encode(text)

    total_nll = 0.0
    total_tokens = 0

    use_amp = device.type == "cuda"

    for start in range(0, len(token_ids) - 1, seq_len * batch_size):
        batch_inputs = []
        batch_targets = []
        for b in range(batch_size):
            offset = start + b * seq_len
            if offset + seq_len + 1 > len(token_ids):
                break
            chunk = token_ids[offset : offset + seq_len + 1]
            batch_inputs.append(chunk[:-1])
            batch_targets.append(chunk[1:])

        if not batch_inputs:
            break

        x = torch.tensor(batch_inputs, dtype=torch.long, device=device)
        y = torch.tensor(batch_targets, dtype=torch.long, device=device)

        with autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            logits, _ = model(x)
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)), y.reshape(-1), reduction="sum"
            )

        total_nll += loss.item()
        total_tokens += y.numel()

    avg_nll = total_nll / max(total_tokens, 1)
    ppl = math.exp(min(avg_nll, 20.0))

    return {
        "metric": "perplexity",
        "perplexity": ppl,
        "avg_nll": avg_nll,
        "num_tokens": total_tokens,
    }


@torch.no_grad()
def evaluate_lambada(
    model: MosaicGPT,
    device: torch.device,
    tokenizer: Optional[AutoTokenizer] = None,
    max_examples: int = 0,
) -> dict:
    """Evaluate last-word prediction accuracy on LAMBADA.

    For each passage, tokenize the full text, feed all tokens except the last
    through the model, and check if the model's argmax prediction matches the
    final token.
    """
    model.eval()
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained("gpt2")

    ds = load_dataset("lambada", split="test")

    correct = 0
    total = 0
    seq_len = model.cfg.position.max_seq_len
    use_amp = device.type == "cuda"

    for ex in ds:
        text = ex["text"].strip()
        if not text:
            continue

        tokens = tokenizer.encode(text)
        if len(tokens) < 2:
            continue

        # Truncate from the left if needed, keeping the last token as target
        if len(tokens) > seq_len + 1:
            tokens = tokens[-(seq_len + 1) :]

        input_ids = torch.tensor([tokens[:-1]], dtype=torch.long, device=device)
        target_id = tokens[-1]

        with autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            logits, _ = model(input_ids)

        # Prediction from the last position
        pred_id = logits[0, -1].argmax(dim=-1).item()
        if pred_id == target_id:
            correct += 1
        total += 1

        if max_examples > 0 and total >= max_examples:
            break

    accuracy = correct / max(total, 1)
    return {
        "metric": "accuracy",
        "accuracy": accuracy,
        "num_correct": correct,
        "num_total": total,
    }


def _score_continuation(
    model: MosaicGPT,
    tokenizer: AutoTokenizer,
    context: str,
    continuation: str,
    device: torch.device,
    use_amp: bool,
) -> float:
    """Score a continuation given a context by summing log-probs of continuation tokens.

    Returns the total log-probability of the continuation tokens, normalized
    by the number of continuation tokens.
    """
    ctx_ids = tokenizer.encode(context)
    cont_ids = tokenizer.encode(continuation)

    if not cont_ids:
        return float("-inf")

    full_ids = ctx_ids + cont_ids
    seq_len = model.cfg.position.max_seq_len

    # Truncate from left if too long
    if len(full_ids) > seq_len:
        # Keep enough context that all continuation tokens are present
        full_ids = full_ids[-seq_len:]
        # Recompute how many continuation tokens remain
        n_cont = min(len(cont_ids), len(full_ids) - 1)
    else:
        n_cont = len(cont_ids)

    if n_cont < 1:
        return float("-inf")

    input_ids = torch.tensor([full_ids[:-1]], dtype=torch.long, device=device)
    target_ids = torch.tensor([full_ids[1:]], dtype=torch.long, device=device)

    with autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
        logits, _ = model(input_ids)

    log_probs = F.log_softmax(logits[0], dim=-1)

    # Sum log-probs only over the continuation positions
    cont_start = len(full_ids) - 1 - n_cont
    total_ll = 0.0
    for i in range(cont_start, len(full_ids) - 1):
        total_ll += log_probs[i, target_ids[0, i]].item()

    return total_ll / n_cont


@torch.no_grad()
def evaluate_hellaswag(
    model: MosaicGPT,
    device: torch.device,
    tokenizer: Optional[AutoTokenizer] = None,
    max_examples: int = 0,
) -> dict:
    """Evaluate commonsense reasoning on HellaSwag (validation set).

    For each example, score 4 candidate endings by their length-normalized
    log-likelihood and pick the best.
    """
    model.eval()
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained("gpt2")

    ds = load_dataset("Rowan/hellaswag", split="validation")

    correct = 0
    total = 0
    use_amp = device.type == "cuda"

    for ex in ds:
        ctx = ex.get("ctx", "")
        endings = ex.get("endings", [])
        label = int(ex.get("label", 0))

        if not endings or not ctx:
            continue

        scores = []
        for ending in endings:
            score = _score_continuation(model, tokenizer, ctx, ending, device, use_amp)
            scores.append(score)

        pred = max(range(len(scores)), key=lambda i: scores[i])
        if pred == label:
            correct += 1
        total += 1

        if max_examples > 0 and total >= max_examples:
            break

    accuracy = correct / max(total, 1)
    return {
        "metric": "accuracy",
        "accuracy": accuracy,
        "num_correct": correct,
        "num_total": total,
    }


@torch.no_grad()
def evaluate_arc_easy(
    model: MosaicGPT,
    device: torch.device,
    tokenizer: Optional[AutoTokenizer] = None,
    max_examples: int = 0,
) -> dict:
    """Evaluate on ARC-Easy science QA (test set).

    For each question, score each answer choice by computing the
    length-normalized log-likelihood of "Question: ... Answer: <choice>"
    and pick the highest-scoring choice.
    """
    model.eval()
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained("gpt2")

    ds = load_dataset("allenai/ai2_arc", "ARC-Easy", split="test")

    correct = 0
    total = 0
    use_amp = device.type == "cuda"

    for ex in ds:
        question = ex.get("question", "")
        choices = ex.get("choices", {})
        answer_key = str(ex.get("answerKey", "A"))

        choice_texts = choices.get("text", [])
        choice_labels = choices.get("label", [])

        if not question or not choice_texts:
            continue

        prompt = f"Question: {question}\nAnswer:"

        scores = []
        for choice_text in choice_texts:
            score = _score_continuation(
                model, tokenizer, prompt, " " + choice_text, device, use_amp
            )
            scores.append(score)

        pred_idx = max(range(len(scores)), key=lambda i: scores[i])
        pred_label = choice_labels[pred_idx] if pred_idx < len(choice_labels) else ""

        if pred_label == answer_key:
            correct += 1
        total += 1

        if max_examples > 0 and total >= max_examples:
            break

    accuracy = correct / max(total, 1)
    return {
        "metric": "accuracy",
        "accuracy": accuracy,
        "num_correct": correct,
        "num_total": total,
    }


def run_all(
    model: MosaicGPT,
    device: torch.device,
    tokenizer: Optional[AutoTokenizer] = None,
    max_examples: int = 0,
    benchmarks: Optional[list[str]] = None,
) -> dict[str, dict]:
    """Run all benchmarks and return combined results.

    Parameters
    ----------
    model : MosaicGPT
        Trained model.
    device : torch.device
    tokenizer : AutoTokenizer or None
        GPT-2 tokenizer. Created automatically if None.
    max_examples : int
        Cap per benchmark (0 = no cap). Useful for quick sanity checks.
    benchmarks : list of str or None
        Which benchmarks to run. None = all four.
        Valid names: "wikitext2", "lambada", "hellaswag", "arc_easy".

    Returns
    -------
    dict mapping benchmark name to results dict
    """
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained("gpt2")

    all_benchmarks = {
        "wikitext2": lambda: evaluate_wikitext2(
            model, device, tokenizer, seq_len=model.cfg.training.seq_len
        ),
        "lambada": lambda: evaluate_lambada(model, device, tokenizer, max_examples),
        "hellaswag": lambda: evaluate_hellaswag(model, device, tokenizer, max_examples),
        "arc_easy": lambda: evaluate_arc_easy(model, device, tokenizer, max_examples),
    }

    if benchmarks is None:
        benchmarks = list(all_benchmarks.keys())

    results = {}
    for name in benchmarks:
        if name not in all_benchmarks:
            print(f"Unknown benchmark: {name}")
            continue
        print(f"Running {name}...")
        results[name] = all_benchmarks[name]()
        _print_result(name, results[name])

    return results


def _print_result(name: str, result: dict):
    metric = result["metric"]
    if metric == "perplexity":
        print(f"  {name}: perplexity = {result['perplexity']:.2f} ({result['num_tokens']:,} tokens)")
    else:
        pct = result["accuracy"] * 100
        print(f"  {name}: accuracy = {pct:.2f}% ({result['num_correct']}/{result['num_total']})")
