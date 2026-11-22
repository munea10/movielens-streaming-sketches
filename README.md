# MovieLens 1M 스트리밍 알고리즘 비교: Bloom Filter vs Count-Min Sketch

빅데이터 분석 과제 4 「스트리밍 알고리즘 2종 구현 및 정확도·메모리 트레이드오프 분석」의 소스코드와 실험 결과입니다.

- **Bloom Filter**: 원소 포함 여부 근사 판정. 키 = (사용자 ID, 영화 ID)
- **Count-Min Sketch**: 항목별 빈도 근사 추정. 키 = 영화 ID
- 두 알고리즘은 외부 라이브러리 없이 직접 구현했고, 정확한 Set·Dictionary를 Ground Truth로 써서 **정확도, 메모리, 처리시간, 초당 처리량**을 비교합니다.
- 두 알고리즘 모두 파라미터를 바꾸는 비교 실험을 수행합니다 (Bloom 15개 설정, CMS 9개 설정).

## 데이터셋

| 항목 | 내용 |
|---|---|
| 이름 | MovieLens 1M (GroupLens Research) |
| 공식 다운로드 | https://files.grouplens.org/datasets/movielens/ml-1m.zip |
| 사용 파일 | `ml-1m/ratings.dat` (`UserID::MovieID::Rating::Timestamp`) |
| 규모 | 평점 1,000,209건, 사용자 6,040명, 영화 3,706편 |

- 원본 데이터는 저장소에 포함하지 않습니다. 실행 시 위 공식 URL에서 자동으로 내려받아 `data/`에 저장합니다.
- 파일의 각 줄을 순서대로 한 줄씩 읽어 처리합니다. 원본 파일 순서대로 재생하는 스트림이며, 실시간·시간순 스트림이라고 주장하지 않습니다.
- 인용: F. Maxwell Harper and Joseph A. Konstan. 2015. *The MovieLens Datasets: History and Context.* ACM Transactions on Interactive Intelligent Systems 5, 4, Article 19. https://doi.org/10.1145/2827872

## 프로젝트 구조

```
movielens-streaming-sketches/
├── README.md
├── movielens_streaming.py          # 전체 소스: 알고리즘 구현 + 실험 + 결과 통합 + 그래프
├── requirements.txt
├── notebooks/
│   ├── run_in_colab.ipynb          # Colab 실행용 (저장소 clone → 스크립트 실행)
│   ├── MovieLens_1M_v1_executed.ipynb   # 1차 실행 기록 (Bloom 결과의 근거)
│   └── MovieLens_1M_CMS_v2_Rerun_executed.ipynb # CMS v2 실행 기록
├── results/
│   ├── v1/movielens_results.zip    # 1차 실행 결과 (Bloom 15개 설정, 포화 실험, Set 기준 등)
│   └── final/
│       ├── movielens_results_final.zip # 기존 완료된 최종 실험 결과
│       └── README.md
└── tests/
    └── test_sketches.py            # 단위 테스트
```

실행하면 다음 폴더가 생깁니다. 두 폴더 모두 Git에서 제외됩니다.
- `data/`: MovieLens 원본
- `output/`: 실행 결과 (`output/movielens_results_final.zip`)

저장소에 보관된 `results/`의 기존 결과는 덮어쓰지 않습니다. 새 런타임에서 시간까지 다시 비교할 때는 이전 `output/` 폴더를 보관 후 삭제하거나 새로 clone한 복사본에서 실행해 중간 결과가 재사용되지 않도록 합니다.

## 실행 방법

저장된 최종 결과는 `results/final/movielens_results_final.zip`에서 바로 확인할 수 있습니다. 보고서의 수치는 이 ZIP과 실행 기록 노트북을 기준으로 합니다. 새 실행은 재현용이며 처리시간은 환경에 따라 달라질 수 있습니다.

### 방법 1: Google Colab (권장)

1. 제출 전에 저장소 소유자가 `notebooks/run_in_colab.ipynb`의 `REPO_URL`을 실제 공개 GitHub 주소로 설정합니다 (또는 접근 권한이 있는 저장소를 사용합니다). 이후 해당 노트북을 Colab에서 엽니다.
2. **런타임 → 모두 실행**을 누릅니다.

노트북이 저장소를 clone하고 데이터를 자동으로 내려받아 실험을 실행합니다. 별도 파일 업로드는 필요 없습니다.

### 방법 2: 로컬 실행

```bash
git clone <이 저장소 주소>
cd movielens-streaming-sketches
pip install -r requirements.txt
python movielens_streaming.py              # reuse 모드 (기본)
```

### 실행 모드

| 모드 | 명령 | 내용 | Colab CPU 예상 시간 |
|---|---|---|---|
| reuse (기본) | `python movielens_streaming.py` | `results/v1/`의 Bloom 결과를 자동으로 불러오고, CMS v2 9개 설정과 Dictionary 기준만 측정한 뒤 최종 결과를 통합 | 약 10분 |
| full | `python movielens_streaming.py --mode full` | **기존 결과 없이 처음부터 전체 재현**: Set 기준, Bloom 15개 설정, 포화 실험, Dictionary 기준, CMS v2 9개 설정을 모두 측정 | 약 40~50분 |
| test | `python movielens_streaming.py --mode test` | 단위 테스트만 실행 (데이터 불필요) | 수 초 |

- 요구 환경: Python 3.10 이상 (`int.bit_count` 사용), matplotlib
- 처리시간은 실행 환경에 따라 달라집니다. Colab은 공유 CPU라 반복 측정 사이에도 변동이 있습니다.

## 실험 구성

### Bloom Filter (15개 설정)

| 실험 | 설정 (m/n, k) |
|---|---|
| m 변경 (k = 6 고정) | (2, 6), (4, 6), (6, 6), (8, 6), (10, 6), (12, 6), (16, 6) |
| k 변경 (m/n = 8) | (8, 1), (8, 3), (8, 6), (8, 9), (8, 12) |
| k 변경 (m/n = 12) | (12, 3), (12, 6), (12, 8), (12, 12) |
| 대표 설정 (목표 FPR 1% 설계값) | **(9.6, 7)** |

- **해시:** 64비트 혼합 함수(`mix64`) 2개로 double hashing을 하고, 비트 배열은 `bytearray`입니다.
- **질의:** 음성 질의 200,000개와 양성 질의 100,000개를 씁니다. 음성 질의는 실제 사용자 ID × 영화 ID 중 정답에 없는 조합으로 만든 평가용 합성 질의입니다.
- **정확도 지표:** FPR(95% Wilson 신뢰구간), FN=0 확인, 이론 FPR (1 − e^(−kn/m))^k와 비교
- **반복:** 대표 설정은 시간 측정 5회, 나머지는 3회입니다. 대표 설정과 (8, 1), (8, 12)는 시드 3개(42, 123, 2026)로 정확도를 확인합니다.
- **포화 실험:** 고유 조합의 절반(n/2)에서 FPR 1%가 되도록 설계한 필터에 데이터를 계속 넣어, 설계 용량의 0.25~2배 지점에서 FPR을 측정합니다.

### Count-Min Sketch v2 (9개 설정)

| 실험 | 설정 (width, depth) |
|---|---|
| 너비 효과 (d = 5) | (124, 5), (371, 5), (1235, 5), **(2719, 5) 대표**, (15000, 5) |
| 깊이 효과 (w = 371) | (371, 1), (371, 5), (371, 8) |
| 동일 메모리 비교 (카운터 수 동일) | (1855, 1) vs (371, 5), (13595, 1) vs (2719, 5) |

- **해시 (v2):** 영화 ID를 먼저 `mix64(ID XOR salt) mod (2^61 − 1)`으로 섞은 뒤, 행마다 선형 모듈러 해시 `((a·x + b) mod (2^61 − 1)) mod w`를 적용합니다. 카운터는 32비트(`array('I')`)입니다.
- **반복:** 대표 설정은 시간 측정 3회에 시드 3개로 정확도를 확인합니다. 나머지 8개는 시드 42로 1회 측정하며, 이 값은 탐색적 비교용입니다.
- **정확도 지표:** 평균 절대오차, 빈도 구간별 상대오차(1–5, 6–49, 50–99, 100+), Top-10/100 일치율, 오차 ≤ εN 보장 만족 비율, 과소추정 0건 확인
- **Top-K:** 알려진 전체 영화 ID에 대한 오프라인 평가이며, 동점은 영화 ID 오름차순으로 처리합니다.

### 측정 방법

- **처리시간:** 파일 읽기 + 파싱 + 자료구조 갱신의 전체 시간과 초당 처리량을 잽니다. 파싱만 한 시간을 따로 재서, 중앙값 차이로 연산 시간을 추정합니다. 이 추정치는 직접 측정이 아닙니다.
- **런타임 구분:** Bloom↔Set, CMS↔Dictionary는 각각 같은 런타임 안에서만 비교합니다.
- **메모리:** 세 가지를 구분합니다.
  - `sys.getsizeof` 기반 객체 크기 추정
  - 배열 저장 공간 (비트 배열·카운터 배열)
  - `tracemalloc` 실측
- **이론 참고값:** 정확한 비트맵·정수 배열은 직접 구현하지 않은 이론 참고값으로만 표시합니다.
- **판정 기준:** Bloom FPR 1%, CMS 고빈도 상대오차 5%·Top-100 95%는 이번 실험을 위한 가정이며, 산업 표준이 아닙니다.

## 결과 버전 (v1 / v2)

| 버전 | 내용 | 위치 |
|---|---|---|
| v1 (1차 실행) | Bloom 15개 설정, 포화 실험, Set·Dictionary 기준, CMS v1(선형 해시) | `results/v1/movielens_results.zip`, `notebooks/MovieLens_1M_v1_executed.ipynb` |
| v2 (CMS 재실험) | CMS v2 9개 설정, Dictionary 기준, v1 Bloom 결과와 통합한 최종 결과 | `results/final/` |

**CMS v1은 최종 결과에 사용하지 않습니다.**
- v1 CMS는 연속 정수인 영화 ID에 선형 해시를 바로 적용했습니다. 그 결과 시드에 따라 버킷 충돌이 무작위 해시보다 크게 많거나 적게 나타났습니다.
- v2는 같은 (a, b) 파라미터에 사전 혼합만 추가해 이 문제를 고쳤습니다. 이 검증은 실행 시 `cms_hash_collision_check.csv`로 저장됩니다.
- v1의 CMS 결과는 `results/v1/` ZIP 안에 실행 기록으로만 보관합니다.
- Bloom Filter는 처음부터 `mix64`를 사용해 이 문제와 무관하며, v1 결과를 그대로 씁니다.

## 테스트

```bash
python movielens_streaming.py --mode test   # 또는
python tests/test_sketches.py               # pytest 사용 시: pytest tests/
```
