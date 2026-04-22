from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Literal, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


LOGGER = logging.getLogger(__name__)


TorchDTypeName = Literal["auto", "float32", "float16", "bfloat16"]


DTYPE_MAP: Dict[str, Optional[torch.dtype]] = {
    "auto": None,
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


@dataclass(frozen=True)
class ModelConfig:
    name: str
    model_condition: str
    device: str = "cuda"
    torch_dtype: TorchDTypeName = "auto"
    batch_size: int = 4
    max_length: int = 1024
    local_files_only: bool = False
    use_chat_template: bool = False
    prompt_style: str = "plain"


@dataclass(frozen=True)
class PromptBuildResult:
    prefix_ids: List[int]
    full_ids: List[int]
    rendered_prefix: str
    prompt_format: str


class CausalLMWrapper:
    """Thin wrapper around a Hugging Face causal LM for deterministic scoring."""

    def __init__(
        self,
        config: ModelConfig,
        force_bos: bool = False,
        append_eos_to_candidate: bool = False,
    ) -> None:
        self.config = config
        self.force_bos = force_bos
        self.append_eos_to_candidate = append_eos_to_candidate

        dtype = DTYPE_MAP[config.torch_dtype]
        device = config.device
        if device == "cuda" and not torch.cuda.is_available():
            LOGGER.warning("CUDA requested but unavailable. Falling back to CPU for %s", config.name)
            device = "cpu"

        self.device = device
        self._tokenizer = AutoTokenizer.from_pretrained(
            config.name,
            local_files_only=config.local_files_only,
        )
        if self._tokenizer.pad_token_id is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        self._model = AutoModelForCausalLM.from_pretrained(
            config.name,
            local_files_only=config.local_files_only,
            torch_dtype=dtype,
        ).to(self.device)
        self._model.eval()

        LOGGER.info(
            "Loaded model=%s model_condition=%s device=%s torch_dtype=%s use_chat_template=%s",
            config.name,
            config.model_condition,
            self.device,
            config.torch_dtype,
            config.use_chat_template,
        )

    @property
    def tokenizer(self):
        return self._tokenizer

    @property
    def model(self):
        return self._model

    @property
    def eos_token_text(self) -> str:
        return self._tokenizer.eos_token or ""

    @property
    def bos_token_id(self) -> Optional[int]:
        return self._tokenizer.bos_token_id

    @property
    def eos_token_id(self) -> Optional[int]:
        return self._tokenizer.eos_token_id

    @property
    def prompt_format(self) -> str:
        return "chat_template" if self.config.use_chat_template else "plain_text"

    def _apply_length_controls(self, ids: List[int]) -> List[int]:
        if len(ids) <= self.config.max_length:
            return ids
        # Keep the right-most tokens. For scoring, preserving immediate context is most important.
        return ids[-self.config.max_length :]

    def _truncate_aligned_prefix_and_full(
        self,
        prefix_ids: List[int],
        full_ids: List[int],
    ) -> tuple[List[int], List[int]]:
        if len(prefix_ids) >= len(full_ids):
            raise ValueError(
                "Candidate produced zero-length continuation before truncation. "
                f"prefix_len={len(prefix_ids)} full_len={len(full_ids)}"
            )

        max_length = int(self.config.max_length)
        if len(full_ids) <= max_length:
            return prefix_ids, full_ids

        drop_count = len(full_ids) - max_length
        truncated_full = full_ids[drop_count:]
        new_prefix_len = len(prefix_ids) - drop_count

        if new_prefix_len < 1:
            raise ValueError(
                "Configured max_length truncates away all prompt context for conditional scoring. "
                f"prefix_len={len(prefix_ids)} full_len={len(full_ids)} max_length={max_length}"
            )
        if new_prefix_len >= len(truncated_full):
            raise ValueError(
                "Configured max_length truncates away all candidate tokens. "
                f"new_prefix_len={new_prefix_len} truncated_full_len={len(truncated_full)}"
            )

        truncated_prefix = truncated_full[:new_prefix_len]
        return truncated_prefix, truncated_full

    def _prepend_bos_if_needed(self, ids: List[int]) -> List[int]:
        if not self.force_bos:
            return ids
        bos_token_id = self.bos_token_id
        if bos_token_id is None:
            return ids
        if ids and ids[0] == bos_token_id:
            return ids
        return [bos_token_id] + ids

    def _maybe_append_eos(self, ids: List[int]) -> List[int]:
        if not self.append_eos_to_candidate:
            return ids
        eos_token_id = self.eos_token_id
        if eos_token_id is None:
            return ids
        if ids and ids[-1] == eos_token_id:
            return ids
        return ids + [eos_token_id]

    def _build_plain_ids(self, prefix_text: str, candidate_text: str) -> PromptBuildResult:
        prefix_ids = self._tokenizer.encode(prefix_text, add_special_tokens=False)
        full_ids = self._tokenizer.encode(prefix_text + candidate_text, add_special_tokens=False)

        prefix_ids = self._prepend_bos_if_needed(prefix_ids)
        full_ids = self._prepend_bos_if_needed(full_ids)
        full_ids = self._maybe_append_eos(full_ids)

        prefix_ids, full_ids = self._truncate_aligned_prefix_and_full(
            prefix_ids=prefix_ids,
            full_ids=full_ids,
        )

        if len(prefix_ids) >= len(full_ids):
            raise ValueError(
                "Candidate produced zero-length continuation after truncation. "
                f"prefix_len={len(prefix_ids)} full_len={len(full_ids)}"
            )

        if full_ids[: len(prefix_ids)] != prefix_ids:
            raise ValueError(
                "Prefix tokenization is not a prefix of full tokenization in plain-text mode. "
                "Add explicit whitespace before candidates or disable aggressive prompt edits."
            )

        return PromptBuildResult(
            prefix_ids=prefix_ids,
            full_ids=full_ids,
            rendered_prefix=prefix_text,
            prompt_format="plain_text",
        )

    def _build_chat_ids(self, prefix_text: str, candidate_text: str) -> PromptBuildResult:
        if not hasattr(self._tokenizer, "apply_chat_template"):
            raise ValueError(
                f"Tokenizer for {self.config.name} does not expose apply_chat_template."
            )

        user_messages = [{"role": "user", "content": prefix_text}]
        prefix_ids = self._tokenizer.apply_chat_template(
            user_messages,
            tokenize=True,
            add_generation_prompt=True,
        )

        full_messages = [
            {"role": "user", "content": prefix_text},
            {"role": "assistant", "content": candidate_text},
        ]
        full_ids = self._tokenizer.apply_chat_template(
            full_messages,
            tokenize=True,
            add_generation_prompt=False,
        )

        prefix_ids = self._prepend_bos_if_needed(list(prefix_ids))
        full_ids = self._prepend_bos_if_needed(list(full_ids))
        full_ids = self._maybe_append_eos(full_ids)

        prefix_ids, full_ids = self._truncate_aligned_prefix_and_full(
            prefix_ids=prefix_ids,
            full_ids=full_ids,
        )

        if len(prefix_ids) >= len(full_ids):
            raise ValueError(
                "Candidate produced zero-length continuation after chat templating/truncation. "
                f"prefix_len={len(prefix_ids)} full_len={len(full_ids)}"
            )

        if full_ids[: len(prefix_ids)] != prefix_ids:
            raise ValueError(
                "Prefix tokenization is not a prefix of full tokenization in chat-template mode. "
                "Try use_chat_template=false for this run."
            )

        rendered_prefix = self._tokenizer.decode(prefix_ids, skip_special_tokens=False)

        return PromptBuildResult(
            prefix_ids=prefix_ids,
            full_ids=full_ids,
            rendered_prefix=rendered_prefix,
            prompt_format="chat_template",
        )

    def build_prompt_ids(self, prefix_text: str, candidate_text: str) -> PromptBuildResult:
        if self.config.use_chat_template:
            return self._build_chat_ids(prefix_text=prefix_text, candidate_text=candidate_text)
        return self._build_plain_ids(prefix_text=prefix_text, candidate_text=candidate_text)
