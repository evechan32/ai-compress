from kvcompress.policy import RSWAPolicy, SinkWindowPolicy


def test_rswa_prefix_always_kept():
    p = RSWAPolicy(window=200)
    # prefix=1000, generated=100 -> 全部生成在窗口内 -> 无 gap
    assert p.retain_ranges(1000, 100) == [(0, 1100)]


def test_rswa_gap_appears_when_generation_exceeds_window():
    p = RSWAPolicy(window=64)
    # prefix=100, generated=200 -> 保留 [0,100) + 末尾 64 个生成 token [236,300)
    ranges = p.retain_ranges(100, 200)
    assert ranges == [(0, 100), (236, 300)]  # 生成段前 136 token 被驱逐
    kept = sum(e - s for s, e in ranges)
    assert kept == 100 + 64


def test_sink_window_evicts_middle_of_total():
    p = SinkWindowPolicy(window=64, sink_len=16)
    ranges = p.retain_ranges(1000, 50)  # total=1050 > 16+64
    assert ranges == [(0, 16), (1050 - 64, 1050)]


def test_sink_window_no_gap_when_short():
    p = SinkWindowPolicy(window=64, sink_len=16)
    assert p.retain_ranges(50, 10) == [(0, 60)]  # total=60 <= 80
