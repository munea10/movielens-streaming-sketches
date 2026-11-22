"""단위 테스트: 작은 입력으로 구현과 실험 설정만 확인한다 (데이터 다운로드 없음).

실행: python tests/test_sketches.py   또는   pytest tests/
"""
import random
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import movielens_streaming as ms  # noqa: E402


def test_builtin_unit_tests():
    ms.unit_tests()


def test_bloom_no_false_negative():
    rng = random.Random(1)
    keys = {(rng.randint(1, 6040), rng.randint(1, 3952)) for _ in range(5000)}
    bf = ms.BloomFilter(50_000, 7, seed=42)
    for k in keys:
        bf.add(k)
    assert all(bf.query(k) for k in keys)
    assert bf.payload() == 50_000 // 8


def test_cms_never_underestimates():
    rng = random.Random(2)
    stream = [rng.randint(1, 3952) for _ in range(20_000)]
    truth = Counter(stream)
    for w, d in ((50, 1), (50, 5), (371, 8)):
        cms = ms.CountMinSketch(w, d, seed=123)
        for x in stream:
            cms.add(x)
        assert all(cms.query(k) >= v for k, v in truth.items())
        assert cms.payload() == 4 * w * d


def test_experiment_configs():
    cfg, ref = ms.bloom_configs(1_000_209)
    assert len(cfg) == 15 and ref in cfg and ref == (9_602_008, 7)
    assert len(ms.CMS_CONFIGS) == 9 and ms.CMS_REF == (2719, 5)
    assert all(a[0] * a[1] == b[0] * b[1] for a, b in ms.EQUAL_MEMORY_PAIRS)
    assert ms.SEEDS == (42, 123, 2026)


if __name__ == '__main__':
    for name, fn in list(globals().items()):
        if name.startswith('test_'):
            fn()
            print('PASS', name)
