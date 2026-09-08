import os
import pytest

from kvcompress.config import load_config


def _clear():
    for k in list(os.environ):
        if k.startswith("AI_COMPRESS_"):
            del os.environ[k]


def test_disabled_by_default():
    _clear()
    cfg = load_config()
    assert cfg.enabled is False


def test_enable_and_window_alignment():
    _clear()
    os.environ["AI_COMPRESS_ENABLE"] = "1"
    os.environ["AI_COMPRESS_RSWA_WINDOW"] = "513"  # 非 16 对齐 → 向上对齐
    cfg = load_config()
    assert cfg.enabled is True
    assert cfg.rswa_window == 528  # ceil(513/16)*16


def test_unknown_policy_raises():
    _clear()
    os.environ["AI_COMPRESS_ENABLE"] = "1"
    os.environ["AI_COMPRESS_POLICY"] = "bogus"
    with pytest.raises(ValueError):
        load_config()


def test_sink_policy_needs_sink_len_aligned():
    _clear()
    os.environ["AI_COMPRESS_ENABLE"] = "1"
    os.environ["AI_COMPRESS_POLICY"] = "sink_window"
    os.environ["AI_COMPRESS_SINK_LEN"] = "20"
    cfg = load_config()
    assert cfg.sink_len == 32
    assert cfg.policy_obj.window == 1024
