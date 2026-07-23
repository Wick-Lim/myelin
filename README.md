# Myelin

**연결성 기반 동적 정밀도 할당** — 트랜스포머의 채널별 저장 비트폭을 학습 중에
연결성 신호로 실시간 재배분하는 학습 프레임워크.

- 기획 문서: [`docs/PLAN.md`](docs/PLAN.md)
- 설계 명세 (구현 1:1 대응): [`docs/DESIGN.md`](docs/DESIGN.md)
- 리서치 브리프 (신규성/관련 연구): [`docs/RESEARCH.md`](docs/RESEARCH.md)

## 핵심 아이디어 세 줄

1. 가중치를 MSB부터 쌓이는 **중첩 비트 플레인**으로 저장 — 비트 증감이 재양자화 없는 포인터 이동.
2. 채널별 **연결성**(shadow 가중치 norm × 활성 흐름 EMA)을 학습 중 측정.
3. 고정 평균 비트 예산 아래 **제로섬 waterfilling**으로 주기 재배분 — 연결성 4배 = 평면 1장.

## 설치

```bash
uv venv .venv && uv pip install -p .venv/bin/python torch numpy pytest tokenizers
uv pip install -p .venv/bin/python -e .
# Intel Mac (torch 2.2.x): numpy<2 필요 → uv pip install -p .venv/bin/python 'numpy<2'
```

## 빠른 시작

```bash
# 테스트 (비트 플레인 중첩성·제로섬 불변식 등 45+)
.venv/bin/python -m pytest tests/ -q

# 합성 데이터 스모크 (전략 1개, 수 분)
.venv/bin/python -m myelin.train --synthetic --strategy connectivity \
    --steps 1200 --out runs/smoke --config configs/demo_synthetic.json

# 한국어 위키 데이터 준비 (1회, 네트워크 필요)
.venv/bin/python scripts/prepare_data.py --out data/kowiki --vocab-size 8192

# 본 실험 매트릭스 (H2/H3 + 신호 ablation) — 예산×전략×시드
.venv/bin/python scripts/run_matrix.py --data-dir data/kowiki --steps 10000 \
    --budgets 3 3.5 4 \
    --strategies connectivity fisher random random_churn kquant uniform \
    --seeds 1337 1338 1339 --parallel 4 --total-threads 8 --out runs/matrix

# 판정
.venv/bin/python scripts/analyze.py runs/matrix          # H2/H3 페어드 비교
.venv/bin/python scripts/h1_probe.py runs/matrix/b4.0_uniform_s1337  # H1 (균일 기준)
```

## Rust 커널 (Phase 2, `kernel/`)

실제 비트 플레인 저장 + 비트 시리얼 GEMV. Phase 1의 fake quant와 달리 연산·메모리가
할당 비트에 선형이다. `pack→dequantize`는 Python `quantize_rows`와 **비트 단위 일치**
(골든 계약, 퍼즈 + 실제 체크포인트로 검증).

```bash
cd kernel && cargo test                 # 골든 계약 포함 11개 테스트
cargo run --release --example bench     # 비트 선형성 확인
maturin develop --release --features python   # PyO3 확장 빌드 (maturin 필요)
cd .. && .venv/bin/python scripts/validate_kernel.py runs/demo/b4.0_connectivity_s7
```

## 저장소 구조

```
myelin/
  bitplane.py    중첩 비트 플레인 양자화 (닫힌형 + 평면 골든 모델, STE)
  layers.py      BitplaneLinear — 채널별 비트 버퍼 + 활성 흐름 EMA
  signals.py     연결성 신호 (product/structural/flow/fisher/random/kquant/uniform)
  allocator.py   제로섬 waterfilling + 주기 재배분 (코사인 감쇠, deadband)
  model.py       MiniGPT (블록 Linear 6종만 양자화, 역할 태깅)
  train.py       학습 루프, 이중 평가(val_loss / val_loss_fp), 궤적 로깅
kernel/
  src/bitplane.rs  패킹/복원 (Python quantize_rows와 비트 단위 일치 계약)
  src/gemv.rs      비트 시리얼 GEMV (비용 ∝ 비트 수)
  src/py.rs        PyO3 바인딩 (feature "python")
  tests/golden.rs  Python 생성 골든 벡터 대조
scripts/
  prepare_data.py  kowiki → BPE → train.bin/val.bin
  run_matrix.py    실험 매트릭스 러너 (병렬, 스레드 분배, 재개 가능)
  analyze.py       H2/H3 집계
  h1_probe.py      H1 민감도-연결성 상관 프로브 (균일 기준 강등)
  gen_golden.py    Rust/RTL 골든 벡터 생성
  validate_kernel.py  Rust 커널 ↔ Python 크로스 검증 (퍼즈 + 체크포인트)
tests/             포맷 성질·불변식이 전부 테스트로 고정됨
```

## 실행 산출물

각 런 디렉토리에: `config.json`, `metrics.jsonl`(loss/lr/mean_bits/트래픽),
`alloc.jsonl`(**배분 궤적** — 이벤트마다 채널별 비트 벡터), `summary.json`, `ckpt.pt`.
