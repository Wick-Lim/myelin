# Myelin

연결성 기반 동적 정밀도 할당 연구 프로젝트. 문서는 한국어, 코드/주석은 영어.

## 명령

```bash
.venv/bin/python -m pytest tests/ -q        # 전체 테스트
.venv/bin/python -m myelin.train --synthetic --strategy connectivity --steps 100 --out runs/dev
.venv/bin/python scripts/analyze.py runs/matrix
```

## 불변 규칙 (깨면 안 됨)

- 비트 플레인 스케일은 absmax만 — 비트 수에 의존하는 스케일은 중첩성을 깬다.
- `quantize_unit`(닫힌형)과 `plane_decompose/reconstruct`(평면 루프)는 항상 일치해야
  한다. 이 등가성 테스트가 Phase 2 Rust 커널의 골든 모델 계약이다.
- 배분은 제로섬: 모든 이벤트 후 `Σ bits == round(avg_bits × N)`.
- 연결성의 구조 신호는 shadow fp32 가중치에서만 측정 (자기실현 루프 차단).
- 페어드 비교: 데이터 순서/초기 가중치/평가 배치는 시드만의 함수여야 한다.
  전략에 따라 RNG 소비가 달라지는 코드를 학습 경로에 넣지 말 것.

## 환경 주의

- 이 개발 머신은 Intel Mac → torch 2.2.2가 마지막 지원 버전이라 `numpy<2` 필요.
  실험 타깃(Ryzen/Linux)은 최신 torch + numpy 2 가능.
- CPU 학습이므로 실험 병렬화 시 `--total-threads`를 물리 코어 수에 맞출 것.
