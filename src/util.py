import json
import random
import subprocess
import time
from contextlib import contextmanager
from typing import Optional, Union

import numpy as np
import torch
from transformers.models.auto.modeling_auto import AutoModelForCausalLM
from transformers.models.auto.tokenization_auto import AutoTokenizer

from model.llama import LlamaForCausalLM
from model.qwen3 import Qwen3ForCausalLM


def parse_config_version(version: str) -> tuple[str, str]:
    """
    parse '<config-type>/version to (config_type, version)
    """
    config_type, version = version.split("/")
    return config_type, version


def convert_dtype(dtype: str) -> torch.dtype:
    match dtype:
        case "bf16":
            return torch.bfloat16
        case "fp16":
            return torch.float16
        case "fp32":
            return torch.float32
        case _:
            raise ValueError(f"Unsupported dtype: {dtype}")


def load_model(name: str, device: torch.device, dtype: torch.dtype):
    model = AutoModelForCausalLM.from_pretrained(
        name, torch_dtype=dtype, device_map=device
    )
    model.eval()
    return model


def load_graph_model(name: str, device: torch.device, dtype: torch.dtype):
    if "llama" in name.lower() or "vicuna" in name.lower():
        model = LlamaForCausalLM.from_pretrained(
            name, torch_dtype=dtype, device_map=device
        )
    elif "qwen3" in name.lower():
        model = Qwen3ForCausalLM.from_pretrained(
            name, torch_dtype=dtype, device_map=device
        )
    else:
        raise ValueError(
            f"Unsupported model: {name}. Only Llama and Qwen models are supported."
        )

    model.eval()
    return model


def load_tokenizer(name: str):
    tokenizer = AutoTokenizer.from_pretrained(name, legacy=False)

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    return tokenizer


def load_dataset(
    name: str, model_name: str | None = None, reasoning: bool | None = False
):
    dataset_name = f"{name.lower()}_prompts"

    try:
        if dataset_name == "specbench_prompts":
            if model_name is None:
                raise ValueError("Model name is required for specbench dataset.")

            file_path = f"data/{dataset_name}.jsonl"
            questions = []
            _tokenizer = load_tokenizer(model_name)

            with open(file_path, "r") as question_file:
                for line in question_file:
                    if line:
                        raw_question = json.loads(line)
                        chat = [{"role": "user", "content": raw_question["turns"][0]}]
                        conv = _tokenizer.apply_chat_template(
                            chat,
                            tokenize=False,
                            enable_thinking=reasoning,
                            add_generation_prompt=True,
                        )
                        questions.append(conv)
            return questions

        file_path = f"data/{dataset_name}.json"
        with open(file_path, "r") as f:
            dataset = json.load(f)
        return [x[1] for x in dataset]
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Missing `data/{dataset_name}.json` file.") from e


def invert_mask(mask: torch.Tensor):
    if mask.min() == 0:
        min_dtype = torch.finfo(mask.dtype).min
        mask = (mask.eq(0.0)).to(dtype=mask.dtype) * min_dtype
    return mask


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.random.manual_seed(seed)


def warn_not_commit():
    """
    Warn that there are files that are not committed.
    """
    try:
        # Check if there are uncommitted changes
        result = subprocess.run(
            ["git", "status", "--porcelain"],  # noqa: S607
            capture_output=True,
            text=True,
            check=True,
        )
        if result.stdout.strip():
            print(
                "Warning: There are uncommitted changes \n\n"
                f"{result.stdout.replace('\n', '\n  ')}"
            )
            proceed = input("Do you want to proceed? (y/N): ")
            if proceed.lower() != "y":
                print("Experiment aborted.")
                exit(0)

    except (subprocess.SubprocessError, FileNotFoundError):
        # If git command fails or git is not installed, silently ignore
        pass


class Timing:
    """
    Time measuring context manager

    Modes:
    - "no-sync": no synchronization
    - "sync": synchronize before and after measuring
    - "event": use torch.cuda.Event for measuring

    Example:
    ```
    with Timing(device="cuda:0", mode="no-sync") as t:
        # do something
    """

    def __init__(
        self,
        device: Optional[Union[torch.device, str]] = None,
        mode="no-sync",
        enabled: bool = True,
    ):
        if mode not in ["no-sync", "sync", "event"]:
            raise ValueError(f"Unsupported mode: {mode}")

        if mode in ["sync", "event"] and device is None:
            raise ValueError("Device must be specified for sync and event modes")

        self.device = device
        self.mode = mode
        self._elapsed = 0.0
        self._enabled = enabled

    def __enter__(self):
        if not self._enabled:
            return self

        if self.mode == "event":
            self.start_event = torch.cuda.Event(enable_timing=True)
            self.end_event = torch.cuda.Event(enable_timing=True)
            self.stream = torch.cuda.default_stream(device=self.device)
            self.start_event.record(self.stream)
        else:
            self.start = time.perf_counter()

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if not self._enabled:
            return

        if self.mode == "sync":
            torch.cuda.synchronize(self.device)
            self._elapsed = (time.perf_counter() - self.start) * 1000
        elif self.mode == "no-sync":
            self._elapsed = (time.perf_counter() - self.start) * 1000
        elif self.mode == "event":
            self.end_event.record(self.stream)
            torch.cuda.synchronize(self.device)
            self._elapsed = self.start_event.elapsed_time(self.end_event)

    @property
    def elapsed(self):
        return self._elapsed


class Profiler:
    def __init__(self, file_name: str):
        self._file_name = file_name
        self._current_stack = []
        self._stack_time = [0]
        self._traces = []

    @contextmanager
    def trace(self, name: str, mode: str, device: Optional[torch.device] = None):
        self._current_stack.append(name)
        self._stack_time.append(0)

        current_stack_str = ";".join(self._current_stack)

        with Timing(mode=mode, device=device) as timer:
            yield

        elapsed = timer.elapsed

        self._stack_time[-2] -= elapsed  # type: ignore
        self._traces.append((current_stack_str, elapsed + self._stack_time[-1]))
        self._current_stack.pop()
        self._stack_time.pop()

    def close(self):
        file = open(f"{self._file_name}", "w")
        for stack_str, total_time in self._traces:
            file.write(f"{stack_str} {total_time:.6f}\n")
        file.close()


profiler = Profiler("test.prof")


def encode(target: torch.Tensor):
    return target.contiguous().cpu().numpy().tobytes()


def decode(target: bytes, device: torch.device, dtype: torch.dtype, shape):
    return torch.frombuffer(target, dtype=dtype).reshape(shape).to(device=device)


@torch.inference_mode()
def sampler_from_logits(
    logits: torch.Tensor, temperature, top_p=0.8, min_tokens_to_keep=1
):
    """
    Performs token sampling from logits
    using top-p (nucleus) sampling or deterministic selection.
    Args:
        logits (torch.Tensor): Logits from a language model.
        temperature (float): Adjusts distribution sharpness (higher = more random);
        top_p (float): Cumulative probability threshold for top-p sampling.
        min_tokens_to_keep (int): Minimum tokens to keep regardless of top_p.
    Returns: Tuple[torch.Tensor, torch.Tensor]: Indices and logprobs of selected tokens.
    """
    if temperature > 0:
        if temperature != 1:
            scores = logits / temperature  # Apply temperature scaling
        else:
            scores = logits  # Potentially modifies logits if masked_fill_ is used later

        if top_p != 1.0:
            # Sort scores in descending order for top-p sampling
            # For logits [B, T, V], sort along V (dim=-1)
            sorted_logits, sorted_indices = torch.sort(scores, descending=True, dim=-1)
            cumulative_probs = torch.softmax(sorted_logits, dim=-1).cumsum(dim=-1)

            # Create a mask to remove logits not in the top-p
            sorted_indices_to_remove = cumulative_probs <= (1 - top_p)
            # Keep at least min_tokens_to_keep tokens
            # (from the tail of sorted, i.e., lowest probability)
            sorted_indices_to_remove[..., -min_tokens_to_keep:] = False

            # Scatter the indices to the original order and mask the logits
            # For scores [B, T, V], scatter along V (dim=-1)
            indices_to_remove = sorted_indices_to_remove.scatter(
                -1, sorted_indices, sorted_indices_to_remove
            )
            scores.masked_fill_(indices_to_remove, -float("inf"))

        # Sampling from the filtered logits
        probs = torch.softmax(scores, dim=-1)  # probs shape: [B, T, V]

        # Original logits shape prefix (B, T) and vocab size V
        logits_prefix_shape = logits.shape[:-1]
        vocab_size = logits.shape[-1]

        # Reshape probs from [B, T, V] to [B*T, V] for multinomial
        probs_flat = probs.view(-1, vocab_size)

        _ = torch.multinomial(probs_flat, 1)  # warmup, output shape [B*T, 1]
        selection_flat = torch.multinomial(probs_flat, 1)  # output shape [B*T, 1]

        # Reshape selection_flat from [B*T, 1] to [B, T]
        selection = selection_flat.view(logits_prefix_shape)

    else:
        # Greedy selection for temperature == 0
        # logits shape: [B, T, V]
        # selection shape: [B, T]
        selection = torch.argmax(logits, dim=-1)

    return selection
