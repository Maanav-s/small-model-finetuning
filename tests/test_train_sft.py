"""Unit tests for the pure-logic helpers in scripts/train/train_sft.py.

These cover the length-filtering / drop logic, the effective-batch math, the
tool-response-opener trim, and the percentile helper -- all GPU/model-free so they
run on the dev box (and in CI) without loading Gemma. The tokenizer-dependent
masking (build_labels) is verified separately via `--dry-run`.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts" / "train"))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src" / "gemma"))

import train_sft as t  # noqa: E402


def test_effective_batch_size():
    assert t.effective_batch_size(1, 8, 2) == 16
    assert t.effective_batch_size(2, 4, 2) == 16
    assert t.effective_batch_size(1, 1, 1) == 1


def test_partition_by_length_basic():
    lengths = [10, 100, 16385, 500, 0]
    kept, dropped = t.partition_by_length(lengths, max_length=16384)
    assert kept == [0, 1, 3]
    assert dropped == [2, 4]  # 16385 too long; 0 is a render-failure sentinel


def test_partition_by_length_boundary_inclusive():
    # exactly max_length is KEPT (<=), one over is dropped
    kept, dropped = t.partition_by_length([16384, 16385], max_length=16384)
    assert kept == [0]
    assert dropped == [1]


def test_partition_by_length_zero_dropped():
    kept, dropped = t.partition_by_length([0, 0], max_length=100)
    assert kept == []
    assert dropped == [0, 1]


def test_trim_open_call_end_trims_opener():
    ids = [1, 2, 49, 50]  # ...<tool_call|>(49), <|tool_response>(50)
    assert t.trim_open_call_end(ids, 4, tool_response_open_id=50) == 3


def test_trim_open_call_end_noop_when_absent():
    ids = [1, 2, 49, 99]  # final turn ends with something else
    assert t.trim_open_call_end(ids, 4, tool_response_open_id=50) == 4


def test_trim_open_call_end_empty():
    assert t.trim_open_call_end([], 0, tool_response_open_id=50) == 0


def test_percentiles_monotonic():
    vals = list(range(1, 101))  # 1..100
    p = t.percentiles(vals)
    assert p[50] <= p[90] <= p[95] <= p[99] <= p[100]
    assert p[100] == 100
    assert p[50] in (50, 51)


def test_percentiles_empty():
    p = t.percentiles([])
    assert set(p.values()) == {0}


def test_build_arg_parser_defaults():
    args = t.build_arg_parser().parse_args([])
    assert args.max_length == 32768
    assert args.lora_r == 16
    assert args.lora_alpha == 32
    assert args.per_device_train_batch_size == 1
    assert args.gradient_accumulation_steps == 8
    assert not args.no_assistant_only_loss  # assistant-only ON by default
    assert "q_proj" in args.lora_target_modules and "down_proj" in args.lora_target_modules


def test_build_arg_parser_overrides():
    args = t.build_arg_parser().parse_args(
        ["--max-length", "8192", "--lora-r", "32", "--no-assistant-only-loss"])
    assert args.max_length == 8192
    assert args.lora_r == 32
    assert args.no_assistant_only_loss
