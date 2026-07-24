# Myelin — 설계 명세 (Phase 1)

연결성 기반 동적 정밀도 할당. 기획 문서의 확장판이며, **구현과 1:1로 대응하는 명세**다.
코드 기준: `myelin/` 패키지. 모든 수식은 구현된 그대로를 기술한다.

---

## 1. 가설과 판정 기준

| 가설 | 내용 | 판정 도구 | 판정 기준 |
|---|---|---|---|
| H1 | 채널별 연결성 지표가 양자화 민감도를 예측한다 | `scripts/h1_probe.py` | 연결성 점수와 채널별 민감도의 Spearman ρ > 0, 시드 3개에서 일관. **순환성 방지**: 프로브는 전 채널을 균일 기준 비트로 리셋한 뒤 채널 하나씩 min_bits로 강등해 측정한다 — 학습된 waterfill 배분(bits ∝ log₄ score) 위에서 재면 강등 폭이 점수의 함수가 되어 귀무가설에서도 ρ > 0이 나온다. 배분이 점수와 무관했던 uniform/random 체크포인트에서 재는 것이 가장 깨끗하다 |
| H2 | 같은 평균 비트 예산에서 연결성 배분 < 무작위 배분 (loss) | `scripts/analyze.py` 페어드 비교 | 시드별 페어드 diff(connectivity − random)가 전 시드에서 음수, 또는 평균이 시드 간 std보다 크게 음수 |
| H3 | 연결성 배분 ≤ k-quant 휴리스틱 | 동일 | 평균 val loss가 kquant 이하 |

**H2 기각 시 프로젝트를 접는다**는 원칙은 유지. 단 기각 분석 시 "왜 무작위가 동등했나"를
배분 궤적(`alloc.jsonl`)에서 역추적한다 — 신호가 무정보였는지(채널 간 분산 부족),
배분이 못 따라갔는지(deadband/주기 과보수) 구분 가능하다.

## 2. 비트 플레인 포맷 명세 (`myelin/bitplane.py`)

가중치 행(출력 채널) `w`는 스케일 `s = max|w|` 와 부호 평면 `b_1..b_K ∈ {-1,+1}` 로 저장된다.
k비트 복원은

```
q_k(w) = s · Σ_{i=1..k} b_i · 2^{-i},   b_i = sign(잔차_{i-1}),  sign(0) := +1
```

**성질 (전부 테스트로 고정됨, `tests/test_bitplane.py`):**

1. **중첩성**: `q_{k+1} = q_k ± s·2^{-(k+1)}` — 평면 추가는 기존 평면을 불변으로 둔 채 잔차만 세분한다.
2. **등가성 (비트 단위)**: `q_k`는 [-1,1] 위 스텝 `2^{1-k}` 의 mid-rise 균일 양자화기와
   **비트 단위로** 일치한다. 학습에서는 이 닫힌형(`quantize_unit`)을 쓰고, 평면
   루프(`plane_decompose/reconstruct`)는 골든 모델로 유지한다.
   **이 등가성 테스트가 곧 Phase 2 Rust 커널의 검증 계약이다.**

   ⚠ **fp32 함정 (리뷰에서 발견·수정)**: 분해를 "잔차 변형" 형태(`r -= b·2^{-i}`)로
   구현하면 등가성이 깨진다 — grid 경계 아래 ~16 ulp 창에서 뺄셈이 round-half-even
   중점에 떨어져 오프셋이 grid 위로 붕괴하고, tie가 닫힌형과 반대 레벨을 고른다
   (전 f32 [-1,1]×k=1..8 전수에서 321쌍 불일치, 한 스텝 전체 오차). 올바른 형태는
   **고정된 입력 u를 실행 중 재구성 부분합 s와 비교**하는 것: s는 항상 2^{-i}의 홀수배
   (가수 ≤8비트, 정확 표현)라 모든 연산이 IEEE 정확이다. Python·Rust 모두 이 형태로
   구현되었고, 경계 ±16 ulp 래더가 테스트·골든 벡터에 고정되어 있다.
3. **오차 상한**: `|w − q_k| ≤ s·2^{-k}`.
4. **스케일 규칙**: `s`는 행 값에만 의존하고 비트 수에 의존하지 않는다(absmax).
   min-max 재적합은 비트 수에 따라 스케일이 달라져 중첩을 깨므로 금지.

비트 하한이 1인 이유: `q_1 = ±s/2` 는 부호 양자화로 의미가 있지만, 0비트는 채널 소멸이라
배분이 아니라 pruning이 된다(기획 문서 §4.3과 일치, 기본 하한은 2).

## 3. 연결성 신호 (`myelin/signals.py`)

채널 c (레이어 L의 출력 채널)에 대해:

- **구조적**: `struct_c = ‖W_L[c,:]‖₂` — **shadow fp32 가중치에서 측정** (자기실현 루프 차단).
- **동적(활성 흐름)**: `flow_c = EMA_{0.99}( mean_{batch,seq} |y_c| )` — 학습 forward의 출력에서
  매 스텝 갱신. EMA 창(~100스텝)은 재배분 주기(500스텝)보다 짧아 주기당 신호가 충분히 신선하다.
- **기본 신호(product)**: `conn_c = (struct_c / mean_L struct) × (flow_c / mean_L flow)`

**레이어별 평균 1 정규화의 이유**: 가중치/활성 스케일은 레이어마다 자릿수가 다르다. 원시 곱을
전역 비교하면 특정 레이어가 스케일 아티팩트만으로 예산을 독식한다. 평균 정규화 후에도
레이어 내 분산 차이를 통해 레이어 간 예산 이동은 일어난다(분산 큰 레이어가 상·하위 슬롯을
모두 차지). 이 선택 자체가 실험 변수이며, 원시 스케일 버전은 후속 매트릭스 항목.

**정직한 한계**: flow는 양자화된 forward의 활성에서 측정된다(shadow forward는 비용 2배라 제외).
양자화 노이즈는 근사적으로 zero-mean이라 `E|y|`에 미치는 영향은 2차적이지만, 완전한 차단은
가중치 쪽(shadow 측정)만 보장된다.

어텐션 가중치 집중도(head 단위 동적 연결성)는 이번 구현에서 제외 — 채널 단위 `|y|` 흐름으로
근사한다. head 단위 신호는 후속 변형으로 명시해 둔다.

### 3.1 강한 대조 신호: Fisher (`fisher`)

리서치 결과(RESEARCH.md §2)의 경고: **weight-norm 단독은 AWQ Table 1에서 random 수준으로
실패했고, 학습 중 배분의 기존 신호는 전부 gradient/Fisher 계열**이다. connectivity가
Fisher급 신호를 이기지 못하면 기여가 "공학적 조합"으로 축소된다. 따라서 신호 ablation을
매트릭스의 핵심 축으로 승격하고, 원리적으로 올바른 대조 신호를 구현했다:

```
fisher_c = EMA( Σ_j g_cj² ) × s_c²      (s_c = 행 absmax = 실제 양자화 스텝 스케일)
```

근거: 행 c 양자화 노이즈 δw ~ s_c·2^{-b} 가 loss에 주는 피해 ≈ Σ_j E[g_cj²]·E[δw²] —
HAWQ/FIT 계열 민감도의 채널 단위 버전이다.

**부수 발견 (설계에 중요)**: Wanda류 입력 에너지 신호 `Σ_j W_cj²·E[x_j²]` 는 출력 채널
양자화에는 부적합하다 — 같은 레이어의 모든 출력 채널이 **같은 입력 x를 공유**하므로
입력 에너지 항이 채널 간 상수가 되어 변별력이 스케일 항으로 퇴화한다. 채널을 가르는 것은
(a) 행 스케일 s_c 와 (b) 하류가 y_c를 얼마나 쓰는가이며, fisher의 g_cj = ∂L/∂y_c · x_j 가
정확히 (b)를 포착한다. connectivity의 `flow`(E|y_c|)는 (b)의 gradient-free 근사라는 것이
Myelin 가설의 실체다.

### 3.2 Weight decay와 신호의 상호작용

WD는 신호의 절반(가중치 norm)을 직접 축소한다(RESEARCH.md §4 경고). 단 WD는 레이어 내
모든 채널에 곱셈적으로 균일하게 작용하므로, **레이어별 평균 정규화가 균일 수축을 정확히
상쇄**한다 — 순위와 비율은 보존된다. 잔여 위험은 채널별 gradient 크기 차이에 의한 비균일
수축뿐이며, 이는 EMA 창(100스텝) 대비 느린 효과다.

## 4. 배분 알고리즘 (`myelin/allocator.py`)

### 4.1 Waterfilling — "몇 비트가 정당한가"의 원리적 답

채널 c를 b비트로 양자화하는 비용을 `conn_c · 4^{-b}` 로 모델링한다
(평면 하나당 제곱오차 4배 감소 × 중요도). 다음 평면의 한계 이득은 `conn_c · 4^{-bits_c}`.
총 평면 수 `T = round(avg_bits × N)` 제약 아래 한계 이득 최대 채널에 평면을 주는 그리디가
이 비용 모델의 최적해이고, 자연스러운 법칙이 나온다:

> **bits_c ≈ 상수 + log₄(conn_c)** — 연결성 4배 = 평면 1장.

k-quant가 손튜닝으로 찾은 "+1~2비트" 오프셋이 이 프레임에서는 점수비 4~16배로 표현된다.
`KQuantSignal`이 정확히 `4^offset` 점수로 이를 재현한다.

### 4.2 재배분 루프

```
warmup (기본 500스텝, 균일 min_bits) → 신호 EMA 축적
step == warmup: waterfill로 초기 배분 (예산 즉시 충족, 제로섬 시작)
이후 period(500)마다:
    target = waterfill(현재 신호)
    donors  = 현재 > target (점수 오름차순) − 쿨다운 중인 채널 제외
    recvs   = 현재 < target (점수 내림차순)
    최대 ⌈k(t)·N⌉ 회, 평면을 1장씩 donor→receiver 이동
      단, conn_recv ≥ (1+deadband)·conn_donor 일 때만 (기본 deadband=0.25)
step ≥ realloc_end_frac·total (기본 0.75, RigL 실증치): 동결
```

- `k(t) = ½·k_start·(1+cos(π·t/t_freeze))` — RigL과 동일 구조의 코사인 감쇠 (기본 k_start=0.2).
- **승격 쿨다운**: 직전 재배분에서 평면을 받은 채널은 `promote_cooldown_cycles`(기본 1)
  주기 동안 donor가 될 수 없다 — ITOP의 reliable-exploration 조건. 연결성 신호는
  자기강화적일 수 있으므로(비트 받은 채널의 norm/flow 성장), 승격 직후 강등의
  진동을 구조적으로 차단한다. 초기 배분은 승격으로 치지 않는다.
- **제로섬 불변식** `Σ bits = T` 는 모든 이벤트 후 assert로 강제된다.
- deadband 조기 종료의 정당성: donor는 오름차순, receiver는 내림차순이므로 현재 쌍에서
  검사가 실패하면 이후 모든 쌍에서도 실패한다 (단조성).
- 정적 신호(random_fixed, kquant, uniform)는 초기 배분 후 재배분을 생략한다 —
  target이 불변이므로 동작 동일, 비용만 절약.

### 4.3 축소의 안전성

학습 중 축소는 데이터 손실이 아니다: 평면은 shadow weight에서 매 forward 재생성되므로
(fake quant), 회수는 유효 정밀도의 일시적 하향일 뿐이다. Adam 모멘트도 shadow 기준이라
재배분 시 옵티마이저 상태 조작이 불필요하다. 유효 가중치 점프의 학습 충격은
EMA + 주기 제한 + deadband + k 감쇠로 완화한다.

## 5. 학습 통합 (`myelin/train.py`, `myelin/model.py`)

- **모델**: MiniGPT d192 / 4층 / 6헤드 / FFN 768 / 문맥 256. 블록 내 6개 Linear
  (q,k,v,o,up,down — **비융합**, 역할별 텐서 정체성 유지)만 양자화. 임베딩·타이드 헤드·LN은 fp32.
  기본 vocab 8192 기준 총 ~3.4M 파라미터.
- **STE**: forward = 양자값, backward = 항등. shadow fp32에 그래디언트 직행.
- **평가 이중 측정**: 매 eval마다 `val_loss`(양자) + `val_loss_fp`(shadow) 를 같이 기록 —
  양자화 갭 자체가 시계열로 남는다.
- **페어드 비교 보장**: 데이터 순서(g_data), 초기 가중치(manual_seed), 평가 배치(g_eval)가
  전략과 무관하게 시드만으로 결정된다. 같은 시드의 두 전략은 완전히 같은 배치 열을 본다.
- **측정 항목**: train/val loss, mean_bits, weight_traffic(현재 배분으로 forward 1회가
  스트리밍할 가중치 바이트 — Phase 1의 메모리 트래픽 대리 지표), 배분 궤적 전체(`alloc.jsonl`,
  이벤트마다 채널별 비트 벡터 포함).

## 6. 실험 프로토콜

### 6.0 데모 진단에서 얻은 두 가지 교훈 (2026-07-23, `runs/demo`)

합성 Markov 데이터·소형 모델·1200스텝의 메커니즘 검증 매트릭스(8런)에서:

1. **예산 4비트는 압력이 없다.** `val_loss − val_loss_fp` 갭이 ~0.0005 — 양자화가 사실상
   무해해서 전략 간 차이가 노이즈에 묻힌다. **본 실험은 예산 3, 3.5를 주력으로** 하고
   4는 보조로 둔다. 갭이 유의미한(≥ 시드 간 std) 예산에서만 H2가 검정력을 가진다.
2. **신호 대비가 waterfill 문턱(4×)에 못 미치면 배분이 uniform으로 퇴화한다.** 데모에서
   정규화된 product 점수의 채널 간 비율이 대부분 4× 미만 → 초기 배분이 전 채널 4비트.
   그런데도 connectivity가 random을 두 시드 모두 이겼다(−0.0016): random은 강제로
   퍼뜨린 배분이고, 그것이 uniform보다 나빴다 — "**정보 없는 spread는 해롭다**"는 것 자체가
   데이터 포인트다. 신호의 순위 정보를 실제로 시험하려면 spread가 있어야 하므로
   `score_gamma`(비트 오프셋 = γ·log₄ score, 기본 1.0)를 ablation 축에 추가한다.
   γ 상향은 "신호를 더 신뢰한다"는 베팅이며, random 대조군에도 같은 γ가 적용되므로 공정하다.

**단계적 실행 (실측 기반)**: d192/L4/vocab8192, batch 32×block 256에서 스텝당
1.6초(2스레드, i9-10910). 10k스텝 풀 매트릭스는 이 머신에서 수일이 걸리므로:
- **1단계**: 예산 3 × 전략 7종(fp32 앵커 포함) × 시드 3 = 21런, **5000스텝**(41M 토큰,
  ~12 tokens/param) → 4병렬로 약 13시간. H2/H3 1차 판정.
- **2단계**: 1단계가 유망할 때만 예산 3.5/4 확장, 10k스텝, score_gamma ablation.

```bash
# 데이터 (1회)
python scripts/prepare_data.py --out data/kowiki --vocab-size 8192 --max-docs 250000

# 본 매트릭스: 예산 3종 × 전략 6종 × 시드 3개 = 54회 (+ fp32 앵커 3회)
python scripts/run_matrix.py --data-dir data/kowiki --steps 10000 \
    --budgets 3 3.5 4 \
    --strategies connectivity fisher random random_churn kquant uniform \
    --seeds 1337 1338 1339 --parallel 4 --total-threads 8 --out runs/matrix
python scripts/run_matrix.py --data-dir data/kowiki --steps 10000 \
    --budgets 8 --strategies fp32 --seeds 1337 1338 1339 \
    --parallel 3 --total-threads 8 --out runs/matrix

# 판정
python scripts/analyze.py runs/matrix
python scripts/h1_probe.py runs/matrix/b4.0_connectivity_s1337
```

확장 항목(2차): **신호 ablation — product vs fisher vs structural vs flow**
(미결정 1번의 실증 해소이자 신규성 방어의 핵심, §7 참조),
random_churn (churn 비용 분리), deadband/k_start 민감도.

### 6.1 1단계 결과 (2026-07-24, kowiki 41M tokens × 21런, 예산 3비트, `runs/stage1`)

| 판정 | 결과 | 수치 |
|---|---|---|
| **H2** (연결성 > 무작위) | **통과** | 페어드 diff −0.0179, 전 시드 음수 (−0.0173/−0.0172/−0.0191), 시드 간 diff 산포 ~0.001 |
| **H3** (연결성 ≥ k-quant) | **통과 (동등)** | +0.0008, 시드별 부호 혼재 — 통계적 동등 |
| **H1** (연결성 → 민감도 예측) | **약한 지지** | uniform 체크포인트 3개에서 Spearman ρ = 0.18 / 0.14 / 0.08 (전부 양수, n=160, 개별로는 1~2.3σ) |

**정직한 기계론적 해석 — 형식적 통과 뒤의 실체:**

1. **uniform이 공동 우승이다.** connectivity(4.7397) vs uniform(4.7373): 페어드 +0.0023으로
   uniform이 근소 우위. fisher(−0.0011)만 uniform을 명목상 이겼고 그마저 유의하지 않다.
   즉 1단계가 실증한 것은 "**정보 없는 spread는 해롭다**"(random +0.0202 vs uniform)이지,
   "정보 있는 spread가 이롭다"가 아직 아니다.
2. **원인은 신호 대비 부족.** 학습된 product 신호의 p95/p5 비율이 4.18× — waterfill
   ±1비트 문턱(4×)에 걸려 γ=1에서 채널의 11%만 차등화됐다 (역할/깊이 평균 전부 ~3.0,
   k-quant 패턴 재발견 실패). connectivity ≈ uniform은 이 때문이다.
3. **churn 비용 실증**: random_churn(4.7663)이 random(4.7575)보다 +0.009 나쁘다 —
   주기 재배분 자체에 비용이 있으며 EMA/deadband/cooldown/freeze의 정당화 근거.
4. **QAT 역전 갭**: val_loss(양자) < val_loss_fp(shadow) — STE 학습이 양자화된 forward에
   최적화되므로 shadow fp 평가가 오히려 나쁘다 (예상 가능한 QAT 성질, 시계열로 기록됨).
5. **3비트의 비용**: fp32 앵커 4.666 vs 최선 양자 4.736 — Δloss ≈ +0.07 (ppl ~+7%).

**2단계 (γ ablation, `runs/stage2_*`)**: 순위 정보의 실체를 시험하려면 spread를 강제해야
한다. γ=2에서 37%, γ=3에서 54% 채널이 차등화됨(학습된 신호 분포로 시뮬레이션).
connectivity-γ2/γ3, fisher-γ2 × 시드 3 = 9런을 1단계와 동일 시드·데이터로 실행,
1단계 uniform/random과 완전 페어드 비교. **γ2가 uniform을 이기면 순위 정보가 실재하는
것이고, random 쪽으로 퇴화하면 연결성 순위가 약하다는 뜻이다.**

### 6.2 2단계 결과 (γ ablation, `runs/stage2_*`) — spread 가설 기각, 가치의 재위치

| 비교 (페어드, 3시드) | diff | 판정 |
|---|---|---|
| conn-γ2 − uniform | **+0.0058 [전 시드 패배]** | 정보 있는 spread도 학습 중엔 손해 |
| conn-γ3 − uniform | +0.0082 [전 시드 패배] | spread가 클수록 더 나쁨 (단조) |
| **fisher-γ2 − uniform** | **+0.0059 [전 시드 패배]** | **Fisher 순위조차 spread로 이득 없음** — 신호 결함이 아니라 구조적 현상 |
| conn-γ2 − random | −0.0143 [전 시드 승리] | 순위 정보는 여전히 실재 |

**기계론적 결론 — 학습 중 배분에서 uniform이 이기는 이유:**
QAT-from-scratch에서 채널 민감도는 **내생적**이다. 2비트를 받은 채널은 2비트짜리
채널이 되도록 학습되고(가중치 재배치·이웃 채널 보상), 배분이 제공하려던 적응을
학습 그 자체가 수행한다. 제로섬 아래에서 한 채널의 3→4비트 이득(4배 체감)이 다른
채널의 3→2비트 손해를 못 이기는 비대칭까지 겹쳐, **적응이 가능한 한 균일이 최적**이
된다. 부호 있는 증거: QAT 갭이 음수(양자 forward가 shadow fp보다 낫다), 그리고
배포 예산을 학습 예산보다 **올려도** loss가 오히려 상승(아래 §6.3).

### 6.3 사후 배분 스윕 (`scripts/posthoc_sweep.py`) — 배분의 가치가 사는 곳

uniform-3비트 학습 체크포인트에서 **재학습 없이** 비트만 재배분(중첩 포맷이라 순수
eval)한 예산별 loss (3시드 평균):

| 배포 예산 | connectivity 순위 | uniform 순위 | random 순위 |
|---|---|---|---|
| 2.0 | 5.006 | 5.006 | 5.006 |
| 2.5 | **4.856** | 4.870 | 4.911 |
| **2.75** | **4.784** | 4.807 | 4.883 |
| 3.0 (=학습 예산) | 4.751 | **4.743** | 4.861 |
| 3.5 | 4.765 | 4.762 | 4.824 |

**판정: 적응이 얼어붙은 배포 시점, 학습 예산 아래로 줄일 때 연결성 순위가 전 시드
승리한다** (2.75에서 uniform 순위 대비 −0.023, random 대비 −0.104). 학습 예산
그대로면 건드리지 않는 게 최선이고, 예산을 올리는 것은 오히려 미세 손해
(모델이 학습된 정밀도에 적응해 있음 — §6.2의 내생성 증거).

**프로젝트 가치의 재정식화:** "학습 중 동적 재배분"이 아니라 —
1. **한 체크포인트 = 임의 배포 예산** (중첩 평면 + 사후 waterfill, 재양자화 없음)
2. **축소 시 누구를 희생할지는 연결성이 답한다** (배포 다이얼의 순위 함수)
3. 학습 중에는 균일이 최적이라는 것 자체가 실증 결과 (QAT 내생성)

### 6.4 진행 중: 강제 triage 실험 (`runs/stage3_b25`)

예산 2.5비트에서는 waterfill이 반드시 절반을 2비트/절반을 3비트로 나눠야 하므로
균일 배분이 존재하지 않는다 — 학습 중에도 순위가 강제로 작동하는 조건.
connectivity/fisher/random/uniform(=균형 무작위 반반) × 시드 3 실행 중.

## 7. 베이스라인 명세

| 전략 | 신호 | 동적? | 검증 대상 |
|---|---|---|---|
| `connectivity` | struct × flow (정규화 곱) | ✓ | 본안 |
| `fisher` | EMA(Σg²) × scale² (FIT 계열) | ✓ | **신호 상한 대조** — connectivity가 이것에 근접해야 gradient-free 주장이 산다 |
| `random` | 고정 무작위 순위 (log-uniform 4^±2 → 예산 4에서 2~6비트, 의도적으로 온건한 spread) | ✗ | H2 — 순서 무정보 대조 |
| `random_churn` | 매 주기 재추첨 | ✓ | H2 교란 통제 — `random`은 정적이라 connectivity와 **순서 정보 + 중간 적응** 두 가지가 동시에 다르다. churn 팔이 "적응만 있고 정보는 없는" 사분면을 채워 둘을 분리한다 |
| `kquant` | Q4_K_M 충실 재현: use_more_bits 레이어 게이팅 + attn_v/mlp_down **+2비트** (llama-quant.cpp@4310aa4) | ✗ | H3 |
| `uniform` | 상수 | ✗ | 참고 앵커 |
| `fp32` | — (양자화 끔) | — | 상한 앵커 |

추가 판정 기준 (RESEARCH.md §3): k-quant의 승격 패턴(residual stream에 되쓰는 attn_v,
mlp_down + 초반/후반 레이어)을 connectivity 배분이 **자발적으로 재발견하는지**를
`alloc.jsonl` 역할별 궤적에서 확인한다. 재발견 실패는 신호 결함의 강한 징후다.

`random`의 점수를 log-uniform으로 뽑는 이유: waterfill이 log₄로 비트를 매기므로,
점수 분포가 좁으면 무작위 배분이 사실상 uniform으로 퇴화해 대조군이 죽는다.
4^±2 범위는 예산 평균 ±2비트의 spread를 만든다 (예산 4 → 실측 2~6비트).
전 범위(2~8)를 강제하지 않는 것은 의도다 — 과도한 무작위 spread는 대조군을
허수아비로 만든다.

**waterfill 동률 처리 주의**: 동률(정확히 같은 점수·한계이득)의 잔여 평면은 고정
시드 순열로 나눈다. 채널 평탄 인덱스 순서는 곧 깊이 순서라, 인덱스 동률 처리는
uniform/kquant 같은 대량-동률 신호에서 예산을 앞쪽 레이어에 몰아주는 깊이 편향을
만들며 시드 평균으로도 씻기지 않는다 (리뷰에서 실증 후 수정됨).

## 8. Phase 2/3 인터페이스 — **Phase 2 크레이트 가동 중** (`kernel/`)

`myelin-kernel` (Rust) 스캐폴드가 구현·검증되었다. Phase 1 학습을 건드리지 않는
독립 크레이트이며, 골든 계약이 이미 실증된 상태다.

**저장 포맷 (`kernel/src/bitplane.rs`):**
```
행(출력 채널)별: scale f32 (absmax, eps 1e-12) + bits u8 + 부호 평면 bits[r]장
평면: u64 워드 배열, 열 j → 워드 j/64 의 비트 j%64 (LSB-first)
      비트 1 = +1, 비트 0 = −1 (sign(0):=+1), cols 초과 패딩은 0 (무해 증명됨)
배치: 행별 평면 연속 저장 → 행 읽기 = bits[r]·words·8 바이트만 스트리밍
```

**검증된 계약:**
- `pack → dequantize` == Python `quantize_rows` **비트 단위 일치** (유한 입력 전체) —
  모든 경로가 IEEE 정확 연산(2의 거듭제곱 나눗셈, 실행 부분합 비교, dyadic 유리수 합)
  이라 보장되며, 골든 벡터 10케이스(경계 ±16 ulp 래더 포함, `scripts/gen_golden.py` →
  `kernel/tests/golden.json`) + 무작위 퍼즈 300행렬 + **실제 학습 체크포인트
  12레이어**에서 실증 (`scripts/validate_kernel.py`).
- **입력 계약**: 비유한(NaN/inf) 가중치는 Rust가 명시적 에러로 거부한다 — torch는 NaN을
  전파하고 `f32::max`는 무시하므로, 패킹을 허용하면 조용한 발산만 가능하기 때문.
  차원 오버플로/OOM급 요청도 프로세스 abort 대신 에러를 낸다 (checked 산술,
  내부 할당이 입력 크기에 유계임을 증명).
- GEMV: `y_r = s_r·Σ_i 2^-i·(2S_i − T)` 비트 시리얼 경로가 f64 기준치와 1e-5 상대오차 내.

**실측 (i9-10910 1스레드, 768×768):** 연산 시간이 비트 수에 선형이고, AVX2
sign-mask LUT 커널(바이트당 256-엔트리 부호마스크 + XOR + 8레인 누적)이 스칼라 대비
4~6배 — 2비트 0.152ms → 8비트 0.583ms, 유효 대역폭 ~1 GB/s. 다음 최적화(다중 x
벡터 블로킹, 멀티스레드, prefetch)도 포맷 변경 없이 내부 루프만 교체된다.

- **PyO3 바인딩**: `cd kernel && maturin develop --release --features python` →
  `import myelin_kernel; pack_matrix(...).matvec(x)`.
- **교체 지점**: `BitplaneLinear.forward`의 `fake_quantize_rows` 호출 하나. backward는
  표준 fp32 경로 그대로 (활성 그래디언트는 full precision).
- **RTL(Phase 3)**: 같은 골든 벡터가 테스트벤치 입력이 되고, Rust 커널 출력이
  사이클 단위 비교 기준이 된다.

## 9. 결정 로그 (기획 문서 §9의 해소)

| # | 항목 | 결정 | 근거 |
|---|---|---|---|
| 1 | 연결성 정의 | **product (norm × flow)** 를 기본값으로 구현하되, structural/flow/fisher도 같은 인터페이스로 구현 → 매트릭스에서 실증 비교 | "확정"이 아니라 실험 변수로 승격하는 것이 H1 검증과 정합. weight-norm 단독의 알려진 실패(AWQ) 때문에 fisher 대조가 필수 |
| 2 | 토크나이저 | **한국어 byte-level BPE 직접 학습** (HF tokenizers) | 학습=배포 일치 철학과 정합. 리서치 실측: 8k에서 1.80 chars/token, 전 고빈도 음절 atomic (RESEARCH.md §5) |
| 3 | 어휘 크기 | **기본 8192** (설정으로 조정) | d192에서 임베딩 1.57M — 총 3.4M로 예산 상단. ko-wiki 전체 ≈ 330M tokens @ 8k — Chinchilla 예산(~60M)의 5배라 서브샘플로 충분 |

**신규성 포지셔닝 (RESEARCH.md §2의 결론 수용)**: "학습 중 채널별 비트 재배분" 범주
자체는 FracBits·QBitOpt가 선행한다. Myelin의 방어 가능한 기여는 3가지 조합 —
(a) gradient-free connectivity 신호로 학습 중 배분 구동, (b) per-output-channel 단위
hard zero-sum swap, (c) 학습→CPU bit-serial 추론까지 nested bit-plane 포맷 일관 유지
(재배분 = re-quantization이 아니라 plane add/drop). 논문화 시 FracBits/QBitOpt를
반드시 인용·비교할 것.

## 10. 리스크 대응 현황

| 리스크 | 구현된 대응 |
|---|---|
| 잔차 방식 손실 | `val_loss` vs `val_loss_fp` 시계열로 갭 상시 정량화 |
| 재배분 불안정 | EMA(0.99) + 주기 500 + deadband 0.25 + k 코사인 감쇠 + 0.8 이후 동결, 전부 설정값 |
| 신호 무정보 (H2 기각 경로) | `alloc.jsonl`의 궤적으로 신호 분산·이동량 사후 분석 가능 |
| 대조군 퇴화 | random 점수 log-uniform 설계 (§7) |

## 11. 관련 연구

리서치 브리프(신규성 평가, k-quant 정확 규칙, QAT 안정화 권고 포함)는
`docs/RESEARCH.md`에 별도 정리한다.
