# Myelin 설계 문서용 연구 브리프: Nested Bit-Plane + 학습 중 Per-Channel 비트 재할당

작성일: 2026-07-23. 본 브리프는 6개 리서치 트랙(MatQuant/Any-Precision, llama.cpp k-quants, RigL/ITOP/Top-KAST, Hessian-aware MP, 신규성 검증, QAT 안정화 + 데이터/하드웨어)의 조사 결과를 종합한 것이다. 확신도가 낮은 항목은 본문에 **(확신도: medium)** 으로 명시했다.

---

## 1. 관련 연구 정밀 지도

Myelin의 4개 축 — **비트 결정 시점 / granularity / 방향(증가·감소) / 할당 signal** — 기준 비교표.

| 연구 | 비트 결정 시점 | Granularity | 방향 | Signal |
|---|---|---|---|---|
| **Myelin (목표)** | 학습 중, 주기적 (매 ΔT step) | per-output-channel | 양방향, hard zero-sum global budget | connectivity (weight-norm × activation-flow, gradient-free) |
| MatQuant [1] | 학습 시 multi-width co-training, 배포 시 MSB slicing | 모델 전체 uniform (Mix'n'Match는 per-layer, post-hoc static) | 정적 (재할당 없음) | multi-bit-width weighted loss (λ=0.1/0.1/1) |
| Any-Precision LLM [2] | post-training (incremental upscaling), 서빙 시 per-request | 모델 전체 uniform | 정적 | weighted K-means cluster split (SqueezeLLM 계열) |
| AnyBCQ [3] | post-training, 서빙 시 dynamic per-request | bit-plane 단위 (plane별 scale) | 런타임 전환만 | BCQ residual |
| llama.cpp k-quants [4][6] | 변환 시 1회 static | per-tensor (tensor role × layer depth) | 정적 | 수작업 heuristic (perplexity 기반 hand-tuning) |
| FracBits [20] | **학습 중, 매 step 연속** | per-layer 또는 **per-kernel (=output channel)** | **양방향** | task-loss + resource penalty gradient (**soft** budget) |
| QBitOpt [21] | **학습 중, 주기적 (τ=250 step)** | per-tensor (실험 기준; per-channel은 언급만) | **양방향, hard avg-bit budget** | FIT (Fisher Information Trace, fp latent weights 기반) |
| Bayesian Bits [22] | 학습 중 (stochastic hard-concrete gates) | per-tensor (2/4/8/16/32-bit nested residual) | 원칙상 양방향 | gate 학습 + BOP-aware prior (soft) |
| BSQ [23] / CSQ [24] | 학습 중 주기적 재양자화 (50–100 epoch) / 연속 | per-layer | BSQ 주로 감소, CSQ 양방향 | bit-level group Lasso / budget-aware regularizer (soft) |
| MixLLM [25] | post-training 1회 | **per-output-channel** (4/8-bit 2단계) | 정적 | global loss-distance salience, global bit fraction |
| SliM-LLM [26] / CMPQ [27] | post-training | per-group / per-channel | 정적 | salience/activation 분포, **strict avg-bit budget** |
| BitStack [28] | training-free, 런타임 load/unload | ~1-bit residual block, 전 layer global sort | 런타임 양방향 | SVD 기반 importance (post-hoc) |
| AutoQ [29] | 학습 외부 RL search (outer loop) | per-kernel | 탐색 결과 static | DRL reward |
| DQ [30] | 학습 중 연속 | per-tensor | 양방향 | gradient (step size + range 파라미터화) |
| CPT [31] | 학습 중 cyclic | **global (전 네트워크)** | 시간축 양방향 | schedule (학습 효율 목적, 최종 모델은 비-mixed) |
| Bit-by-Bit [32] | 학습 중 stage-wise progressive | per-block, nested grids | **감소만** | schedule |
| HAWQ-V3 [13] | 학습 전/후 ILP 1회 (<1 s) | per-layer (scale만 per-channel) | 정적 | Hessian trace × quantization error, hard budget |
| (구조 참고) RigL [7] / Top-KAST [10] | 학습 중 주기적 (ΔT=100) | per-weight (sparsity) | **zero-sum swap** | magnitude drop + gradient grow / TopK magnitude |

---

## 2. 신규성 평가 (정직한 결론)

**결론 먼저: "학습 중 per-channel 비트 재할당, 양방향, 고정 평균 비트 budget"이라는 범주 자체는 신규가 아니다.** FracBits(2020/21)는 per-output-channel 양방향 비트 학습을 (soft budget으로) 이미 했고 [20], QBitOpt(2023)는 논문 제목부터 "Fast and Accurate **Bitwidth Reallocation during Training**"이며 hard avg-bit budget 하의 주기적 양방향 재할당을 구현했다 [21]. Nested MSB-first 학습 포맷은 Bayesian Bits [22]와 MatQuant [1]이, bit-plane 저장 + CPU bit-serial 서빙은 Any-Precision LLM [2]과 T-MAC [47]이 선점했다.

**실제로 살아남는 신규성 (3가지 조합 요소):**

1. **Gradient-free connectivity signal (weight-norm × activation-flow)로 학습 중 비트 할당을 구동** — 학습 중 할당 기준으로 이 signal을 쓴 논문은 ~15회의 adversarial search에서 발견되지 않았다. 기존 학습 중 기준은 전부 gradient/Fisher/Hessian 또는 bit-sparsity regularizer이고, weight×activation salience는 PTQ(AWQ 계열)에만 존재한다 **(확신도: medium — 부재 증명)**.
2. **Per-output-channel 단위의 hard zero-sum swap** — QBitOpt는 실험이 per-tensor, FracBits는 soft budget. 단, QBitOpt는 per-channel 확장을, BSQ는 filter-wise 그룹을 언어적으로 예약해 두었다.
3. **학습→CPU bit-serial 추론까지 nested sign-magnitude bit-plane 포맷 일관 유지** — 재할당이 re-quantization이 아니라 plane add/drop이라는 metadata 변경이 되는 구조는 어느 논문에도 없다.

**최근접 이웃 + 한 줄 차이:**

| 이웃 | Myelin과의 차이 (한 줄) |
|---|---|
| FracBits [20] | per-channel·양방향·학습 중이지만 soft L1 budget + gradient signal, nesting 없음 |
| QBitOpt [21] | hard budget·주기적·양방향이지만 per-tensor 실험, Fisher signal, nesting 없음 |
| Bayesian Bits [22] | nested residual 분해를 학습하지만 per-tensor gate, soft prior, 2의 거듭제곱 비트만 |
| MatQuant [1] | nested int8⊃int4⊃int2를 학습하지만 uniform precision, budget/재할당 없음 |
| Any-Precision LLM [2] + T-MAC [47] | bit-plane 저장/커널은 완비했지만 비트 깊이 결정은 post-training/서빙 시 |
| MixLLM [25] | per-output-channel + global budget salience이지만 one-shot PTQ, 4/8 2단계뿐 |
| BitStack [28] | global sorted residual-plane budget dial이지만 training-free(SVD), per-matrix |
| Bit-by-Bit [32] | nested grid QAT지만 감소 전용 progressive, per-channel budget 없음 |

**권장 포지셔닝:** "QBitOpt-style hard-budget reallocation을 per-output-channel로 내리고, MatQuant/Bayesian-Bits-style nested format으로 re-quantization-free하게 만들고, Fisher sensitivity를 저비용 connectivity signal로 대체한 것." **필수 baseline/ablation:** FracBits(kernel-wise), QBitOpt(per-channel 확장), BSQ/CSQ, MatQuant, 그리고 동일 평균 비트에서 signal ablation(connectivity vs FIT/squared-gradient vs weight-norm-only). **connectivity signal이 FIT를 이기지 못하면 기여는 engineering 조합으로 축소된다.** 또한 Hessian 계열 문헌의 교훈: 할당 규칙은 raw sensitivity 순위가 아니라 **sensitivity × bit별 error delta** (Ω_i(b) = S_i·err(b), err는 비트당 ~4배 감소) 형태여야 하며 [12][13], weight-norm 단독은 AWQ Table 1에서 random 수준으로 실패했으므로 [16] input-activation 2차 모멘트 가중(row energy = Σ_j W_ij²·E[x_j²], Wanda가 diagonal-Hessian OBS와 동치임을 증명 [17])을 반드시 포함해야 한다.

---

## 3. k-quant 규칙 (baseline 구현 사양)

출처: llama.cpp `src/llama-quant.cpp`, **commit 4310aa4 (2026-07-22) 기준** — category refactor로 구버전 해설과 다르므로 재현 시 commit을 명시할 것 [4].

**Layer gate (`use_more_bits`)** — 전체 depth의 첫 1/8 + 마지막 1/8 + 중간 3개마다 1개 (~50%):

```
use_more_bits(i, n) = (i < n/8) || (i >= 7n/8) || ((i - n/8) % 3 == 2)
```

**Q4_K_M 정확 규칙** (base = Q4_K 4.5 bpw, 모든 tensor):

| Tensor role | 할당 |
|---|---|
| attn_v | use_more_bits layer에서 **Q6_K** (6.5625 bpw), 아니면 Q4_K |
| ffn_down | use_more_bits layer에서 **Q6_K**, 아니면 Q4_K |
| output.weight | 항상 **Q6_K** (n_embd % 256 ≠ 0이면 Q8_0 fallback) |
| token_embd | Q4_K |
| attn_q, attn_k, attn_output, ffn_gate, ffn_up | Q4_K (**승격 없음** — "Q4_K_M이 attn_output을 올린다"는 통설은 현 master에서 거짓; attn_output→Q5_K는 n_expert==8 MoE 한정) |

**Q4_K_S**: attn_v → Q5_K (첫 4 layer만), ffn_down → Q5_K (첫 n/8 layer, 비-Falcon), output.weight → Q6_K, 나머지 전부 Q4_K. 아키텍처 조건부: 70B GQA는 attn_v→Q5_K 전 layer, 8-expert MoE는 attn_v/attn_k→Q8_0 (비용 대비 효과 인식 할당: GQA에서 attn_v는 attn_q의 1/8 크기) [4].

**bpw 참조** (256-weight super-block) [5]: Q2_K 2.625 / Q3_K 3.4375 / Q4_K 4.5 / Q5_K 5.5 / Q6_K 6.5625 / Q8_0 8.5. Q5_K는 5번째 비트를 별도 `qh[]` bit-plane으로 저장 — Myelin nested format의 shipped CPU 선례. **벤치마크 anchor**: LLaMA-7B에서 Q4_K_S→Q4_K_M (+0.30 bpw)가 perplexity −0.061 (6.0215→5.9601) **(bpw 환산은 파생치, 확신도: medium)** [6]. Myelin의 학습된 할당은 이 ppl/bpw 교환비를 이겨야 한다.

**Baseline용 role → relative bit offset** (Q4_K_M 준거; offset은 `use_more_bits` layer에서만 적용, 나머지 layer는 0; layer-uniform 단순화가 필요하면 duty ~50%이므로 평균 +1로 근사):

```json
{"attn_q": 0, "attn_k": 0, "attn_v": 2, "attn_o": 0, "mlp_up": 0, "mlp_down": 2}
```

**해석**: 추가 비트는 residual stream에 되쓰는 두 행렬(attn_v, mlp_down)과 output head, 그리고 초반/후반 layer에 집중된다. Myelin의 connectivity signal이 이 패턴을 재발견하지 못하면 signal 결함을 의심할 것.

---

## 4. QAT 안정화 권고 (Myelin trainer 직접 적용 항목)

**Scale 추정:**
- **LSQ류 learned scale 금지, 통계적(absmax/absmean) per-output-channel symmetric scale을 매번 재계산.** 근거 3중: (a) learned step size는 Q_P 의존적이라 비트 재할당 시마다 무효화 [34], (b) OFQ가 learned scale이 transformer oscillation을 증폭함을 보임 [35], (c) LLM-QAT가 LLM 스케일에서 clipping류가 MinMax를 이기지 못함을 보임 (outlier가 load-bearing) [38].
- Activation은 per-token absmax int8, RMSNorm을 quantizer 직전에 배치 (BitNet SubLN 트릭) [39]. Activation을 8-bit 미만으로 내릴 경우 FFN down-projection 입력은 예외 처리 (outlier 집중 지점) [39].
- Slicing 시 truncation이 아닌 **round-to-nearest**: `clamp(round(q/2^(c-r)), 0, 2^r−1)` (MatQuant 명시적 설계 선택) [1].

**Warmup / 스케줄:**
- fp(또는 최대 비트) 선행 학습 후 LR-cooldown 경계에서 공격적 양자화/재할당 개시 [36][40]. QAT 비율은 평균 비트가 낮을수록 확대 — 1-bit급이면 학습의 40–50% **(확신도: medium)** [40]. sub-3-bit 채널은 ~3배의 적응 토큰이 필요하고 표현이 재조직됨(ParetoQ의 2↔3-bit 'learning transition') — 저비트 채널의 잦은 재할당은 토큰 비용이 크다 [41].
- BitNet b1.58 optimizer 패턴 (저비트 유효폭 시): fp baseline 대비 ~5–6배 높은 peak LR, 2단계 decay, **weight decay 0.1 → 0 (후반부)**, warmup 375 step, Adam (0.9, 0.95) [39]. **주의 — WD는 Myelin connectivity signal의 절반(weight-norm)을 직접 축소하고 flip churn을 늘린다**: norm은 decay 적용 전 값을 읽거나, WD를 끄는 시점 이후 재할당을 동결할 것.

**재할당 스케줄 (RigL/ITOP/Top-KAST 이식):**
- Gate: `t mod ΔT == 0 && t < T_end`, 재할당 비율은 cosine anneal `f(t) = (α/2)(1+cos(πt/T_end))`, α≈0.3, **T_end ≈ 학습의 75%에서 할당 동결** (조기 동결이 정확도에 유리 + 마지막 1/4 동안 고정 포맷 확보) [7].
- **신규 plane은 zero-init + optimizer state도 zero** — nested format에서는 LSB plane 추가가 자동으로 output-preserving이므로 공짜 (RigL이 공들여 만든 성질) [7][8].
- **ΔT는 batch size에 반비례 스케일** (RigL ΔT=100 @ batch 4096 ≈ ITOP ΔT=4000 @ batch 64; 갱신 간 ~40만 samples 기준) [7][9]. Connectivity signal은 자기강화적(비트를 받은 채널의 norm/flow가 커짐)이므로, **승격된 채널은 최소 ΔT 동안 강등 금지** (ITOP의 reliable-exploration 조건) [9].
- Health metric: ΔT 이상 고비트를 보유해 본 채널의 누적 비율(Rs-analogue)을 추적 — 탐색인지 고착인지 판별 [9].
- Promotion 경계 안정화 (Top-KAST): 임계 직하 채널의 shadow band에 fp32 master-weight 업데이트 유지(Myelin은 어차피 fp32 shadow 사용) + 비대칭 penalty(incumbent에 약한 decay, exploration band에 ×1/D 강한 decay)로 경계 churn 억제 [10].

**Oscillation 대응:**
- Per-weight integer flip을 EMA(m≈0.01–0.1)로 추적, f_th≈0.01–0.02(cosine anneal 0.04→0.01) 초과 시 **integer domain에서 freeze** (scale이 계속 움직여도 유효) [33]. Dampening 대안: bin-center 방향 regularizer `||ŵ − clip(w)||²` [33].
- **Myelin 고유 리스크**: 재할당은 채널 grid를 이동시켜 새 decision threshold에 재노출시키고, nested format의 MSB(residual-sign) flip이 최악 사례 — **채널별 oscillation rate가 낮을 때만 해당 채널 재할당을 허용**하는 gating 권장.
- 다중 비트폭 loss 사용 시 최저 비트 항을 강하게 가중 (MatQuant 기본 λ = 10:1 for int2) [1]; ~0.05 bits/weight의 outlier side-channel이 2-bit에서 +6% **(확신도: medium)** [1]. Bit-by-Bit의 outlier channel splitting은 zero-sum budget에 굶주린 채널의 escape valve 후보 [32].
- **경고**: scale 기반 uniform quantizer에 nesting을 후속 부착하면 붕괴 (AWQ 4-bit ppl 5.59→22.50) [2] — Myelin의 per-channel scale은 plane 수와 무관하게 안정해야 하며, residual-sign 분해가 이를 자연 제공.

---

## 5. 데이터 / 토크나이저 결정 근거

**데이터**: HF `wikimedia/wikipedia` config **`20231101.ko`** — 647,897 문서, parquet ~782 MB, 텍스트 ~1.29 GB, CC-BY-SA-3.0+GFDL [42]. 이 repo는 2023-11 dump에서 동결 상태이므로, 더 신선한 데이터가 필요하면 `omarkamali/wikipedia-monthly` `latest.ko` (754,861 rows, 월간 갱신, 1k/5k/10k sub-split 제공 — CPU toy run에 적합) [43], 또는 `lcw99/wikipedia-korean-20240501` [44].

**Vocab: byte-level BPE 8,192 권장.** 근거 (로컬 실험, ~2,700 문서 학습 / 300 문서 held-out [56]):

| vocab | chars/token | 비고 |
|---|---|---|
| 2,048 | 1.43 | budget 대부분을 음절 재조립에 소모 |
| 4,096 | 1.62 | 고빈도 음절 ~1,094개 atomic |
| **8,192** | **1.80** | 전 고빈도 음절 atomic + 2음절 형태소 4,860개 |
| 16,384 | 1.97 | fertility +9%에 embedding table 2배 |

핵심 논리: (a) 한글 음절은 UTF-8 3바이트라 byte-level BPE는 ~1,000 merge를 음절 복원에 먼저 소비; 실사용 음절 ~1,000개가 텍스트의 99.75%를 커버하므로 8k에서 음절 + 형태소까지 도달. (b) 3M-param budget에서 tied embedding 8k×d128 ≈ 1.05M params (~35%)가 이미 상한 — 16k는 불가. (c) 토큰 총량: 전체 ko-wiki @ 8k ≈ **~330M tokens (320–355M)** **(외삽치, 확신도: medium)** — 3M params의 Chinchilla budget(~60M)의 ~5배 이상이므로 1 epoch 미만 subsample로 충분. 영어 중심 tokenizer 재사용 금지, 한국어 텍스트로 직접 학습할 것. NFC(완성형) vs NFD(자모) 선택은 후일 bit-serial RTL 경로와의 정합성 관점에서 검토.

---

## 6. 하드웨어 전망

**Bit-serial의 선례와 비용 모델:**
- Stripes (MICRO 2016): 실행 시간이 precision에 ~선형, bit-parallel 대비 평균 1.92× (1.30–4.51×), 에너지 57% 개선, 면적 +32% — "비용이 비트에 선형" 가정의 hardware 원전 [45]. BitFusion(ISCA 2018)은 spatial 대안이지만 2의 거듭제곱 폭만 지원 — **임의 폭(3-bit 등) + MSB-first early termination을 자연 지원하는 것은 bit-serial(temporal) 쪽이며 Myelin nested format과 정합** [46].
- **T-MAC이 Myelin의 CPU 실행 모델**: weight를 1-bit plane으로 분해, g=4 grouping + 2^4 partial-sum LUT (NEON `tbl` / AVX2 `pshufb`) — n-bit 채널 = n개의 1-bit GEMM 스택, 비용이 할당 비트에 정확히 선형, 3-bit가 4-bit의 3/4 비용 [47]. Single-thread kernel vs llama.cpp: **11.2× (1-bit) / 5.8× (2-bit) / 4.7× (3-bit) / 3.1× (4-bit)** — dequant 커널은 4→2-bit에서 이득이 없는 반면 LUT는 선형 [47]. llama.cpp Q3_K는 unpacking 오버헤드로 오히려 4-bit보다 느림 **(확신도: medium)** [47]. **zero-sum budget의 경제성 검증**: bit-plane LUT 커널에서 채널 간 비트 이동은 throughput-neutral, 총 budget 삭감은 ~선형 이득.
- **Decode는 bandwidth-bound**: multi-thread GEMV는 DRAM 대역폭에 수렴 (2-bit에서 M2-Ultra 2.5×, RPi5 4.0×, Orin 5.31×) [47]. 단 **prefill/batch는 LUT의 약점** — BiQGEMM은 batch >128에서 MKL에 역전 [49], Vec-LUT(2025-12)는 scalar LUT가 multi-token에서 대역폭을 못 채움을 지적 (SOTA LUT 대비 최대 4.2×) [51] — prefill용 vectorized-lookup 또는 MAD fallback 계획 필요. ULPPACK류 packing은 activation도 저비트여야 해서 Myelin(고정밀 activation)에 부적용 [50].

**int8 대비 crossover:**
- 직접적 T-MAC vs VNNI 벤치마크는 부재. 최선의 증거는 bitnet.cpp가 MAD 커널(I2_S)과 LUT 커널(TL1/TL2)을 **둘 다** 출하한다는 사실 — crossover는 플랫폼/스레딩 의존 (LUT: compute-bound/소수 코어 우세, MAD int8: bandwidth-bound + 강한 int8 유닛에서 경합) **(확신도: medium)** [48]. Apple AMX 같은 강한 GEMM 하드웨어는 compute-bound 구간에서 LUT 우위를 침식 (M2-Ultra prefill 2.0×에 그침) [47].

**Zen 3 (예: Ryzen 7 5800X) 사실 확정:**
- AVX2 + FMA3 있음; **AVX-512 없음 (전 variant); VNNI 없음** (AVX512-VNNI는 Zen 4, AVX-VNNI는 Zen 5부터). int8 dot은 `vpmaddubsw`+`vpmaddwd` 에뮬레이션 → fp32 대비 ~2× (VNNI의 ~4×가 아님) — **Zen 3에서는 crossover가 LUT 커널 쪽으로 유리하게 이동** [52].
- fp32 peak: 2×256-bit FMA pipe = 32 FLOP/cycle/core; 8-core 이론치 ~1.15–1.20 TFLOPS (all-core AVX2 ~4.5–4.7 GHz); 잘 튜닝된 SGEMM 실측 ~0.85–1.05 TFLOPS **(추정치, 공인 multi-core 실측 부재, 확신도: medium)** [53]. dual-channel DDR4 ~50 GB/s는 bandwidth-bound 논거를 T-MAC 시험 기기들보다 오히려 강화.
- End-to-end 참고치: BitNet-3B 30 tok/s (M2-Ultra 1-core), Llama-2-7B W2 CPU 0.66 J/token vs GPU 1.54 J/token (Orin) [47]; bitnet.cpp는 x86에서 llama.cpp 대비 2.37–6.17× [48].

---

## 7. 참고문헌

1. MatQuant (Matryoshka Quantization) — https://arxiv.org/abs/2502.06786
2. Any-Precision LLM — https://arxiv.org/abs/2402.10517 (code: https://github.com/SNU-ARC/any-precision-llm)
3. AnyBCQ — https://arxiv.org/abs/2510.10467
4. llama.cpp `src/llama-quant.cpp` @ 4310aa4 — https://github.com/ggml-org/llama.cpp/blob/4310aa4f871c104698f6a6614a362bdec87c247a/src/llama-quant.cpp
5. ggml-common.h (block layouts) — https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-common.h
6. llama.cpp k-quants PR #1684 — https://github.com/ggml-org/llama.cpp/pull/1684
7. RigL — https://arxiv.org/abs/1911.11134 / http://proceedings.mlr.press/v119/evci20a/evci20a.pdf
8. RigL 공식 코드 — https://github.com/google-research/rigl
9. ITOP — https://arxiv.org/abs/2102.02887
10. Top-KAST — https://proceedings.neurips.cc/paper/2020/file/ee76626ee11ada502d5dbf1fb5aae4d2-Paper.pdf
11. HAWQ — https://arxiv.org/abs/1905.03696
12. HAWQ-V2 — https://arxiv.org/abs/1911.03852
13. HAWQ-V3 — https://arxiv.org/abs/2011.10680
14. Optimal Brain Surgeon — https://proceedings.neurips.cc/paper/1992/file/303ed4c69846ab36c2904d3ba8573050-Paper.pdf
15. GPTQ — https://arxiv.org/abs/2210.17323
16. AWQ — https://arxiv.org/abs/2306.00978
17. Wanda — https://arxiv.org/abs/2306.11695
18. SqueezeLLM — https://arxiv.org/abs/2306.07629
19. Molchanov, Importance Estimation for NN Pruning — https://arxiv.org/abs/1906.10771
20. FracBits — https://arxiv.org/abs/2007.02017
21. QBitOpt — https://arxiv.org/abs/2307.04535
22. Bayesian Bits — https://arxiv.org/abs/2005.07093
23. BSQ — https://arxiv.org/abs/2102.10462
24. CSQ — https://arxiv.org/abs/2212.02770
25. MixLLM — https://arxiv.org/abs/2412.14590
26. SliM-LLM — https://arxiv.org/abs/2405.14917
27. CMPQ — https://arxiv.org/abs/2410.13056
28. BitStack — https://arxiv.org/abs/2410.23918
29. AutoQ — https://arxiv.org/abs/1902.05690
30. Differentiable Quantization (Uhlich) — https://arxiv.org/abs/1905.11452
31. CPT — https://arxiv.org/abs/2101.09868
32. Bit-by-Bit — https://arxiv.org/abs/2604.07888
33. Nagel, Overcoming Oscillations in QAT — https://arxiv.org/abs/2203.11086
34. LSQ — https://arxiv.org/abs/1902.08153
35. OFQ — https://arxiv.org/abs/2302.02210
36. Jacob, Integer-Arithmetic-Only Inference — https://arxiv.org/abs/1712.05877
37. Krishnamoorthi whitepaper — https://arxiv.org/abs/1806.08342
38. LLM-QAT — https://arxiv.org/abs/2305.17888
39. BitNet b1.58 Training Tips/Code/FAQ PDF — https://github.com/microsoft/unilm (bitnet; 로컬 추출본: docs/refs/bitnet_tips.txt (저장소 보존본))
40. Compute-Optimal QAT — https://arxiv.org/abs/2509.22935
41. ParetoQ — https://arxiv.org/abs/2502.02631
42. wikimedia/wikipedia (20231101.ko) — https://huggingface.co/datasets/wikimedia/wikipedia
43. omarkamali/wikipedia-monthly — https://huggingface.co/datasets/omarkamali/wikipedia-monthly
44. lcw99/wikipedia-korean-20240501 — https://huggingface.co/datasets/lcw99/wikipedia-korean-20240501
45. Stripes (MICRO 2016) — https://ieeexplore.ieee.org/document/7783722/
46. BitFusion (ISCA 2018) — https://arxiv.org/abs/1712.01507
47. T-MAC — https://arxiv.org/abs/2407.00088 (code: https://github.com/microsoft/T-MAC)
48. bitnet.cpp — https://arxiv.org/abs/2410.16144 / https://arxiv.org/abs/2502.11880
49. BiQGEMM — https://arxiv.org/abs/2005.09904
50. ULPPACK (MLSys 2022) — https://proceedings.mlsys.org/paper_files/paper/2022/hash/e09d45e14e9ece7142217550ddd3c4d0-Abstract.html
51. Vec-LUT — https://arxiv.org/abs/2512.06443
52. Zen 4 AVX-512 분석 (Phoronix) — https://www.phoronix.com/review/amd-zen4-avx512 ; TechInsights — https://www.techinsights.com/blog/amd-zen-4-adds-avx-512
53. CPU GEMM peak 산식 — https://salykova.github.io/gemm-cpu
54. MSQ — https://arxiv.org/abs/2507.22349
55. Bit-Mixer — https://arxiv.org/abs/2103.17267 ; AdaBits — https://arxiv.org/abs/1912.09666
56. 로컬 BPE 실험 스크립트 — docs/refs/bpe_experiment.py (저장소 보존본) (tokenizers 0.23.1, ByteLevel BPE, 20231101.ko 샘플)
