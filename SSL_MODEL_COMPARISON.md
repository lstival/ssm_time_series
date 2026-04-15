# SSL Model Comparison — Probe Results

> Avg MSE over horizons H=96/192/336/720. **Bold** = best per row.
> Dataset: Lotsa (new). Evaluated with linear probe.

---

## Table 1 — Bimodal / Full Models

| Model | Description |
|-------|-------------|
| SimCLR-Full | SimCLR bimodal, full lotsa training (~20h) |
| Gram-Full | Gram matrix SSL, full lotsa training (~10h) |
| SimCLR-Best | SimCLR bimodal, best HPO config, old dataset |
| HPO-Best | Best HPO run, old dataset |
| Best | Best overall checkpoint, old dataset |
| CLIP-MFT | CLIP micro fine-tune on lotsa (~12min) |

| Dataset | SimCLR-Full | Gram-Full | SimCLR-Best | HPO-Best | Best | CLIP-MFT |
| --- | --- | --- | --- | --- | --- | --- |
| ETTm1 | **0.1633** | 0.3027 | 0.2968 | 0.3740 | 0.2200 | 0.2533 |
| ETTm2 | **0.1540** | 0.2372 | 0.2506 | 0.2919 | 0.2051 | 0.2156 |
| ETTh1 | **0.2620** | 0.4408 | 0.2803 | 0.3270 | 0.3324 | 0.2824 |
| ETTh2 | 0.2414 | 0.3394 | 0.2625 | 0.2656 | 0.2716 | **0.2291** |
| weather | 0.0184 | 0.1543 | 0.0278 | 0.0206 | 0.0423 | **0.0127** |
| traffic | 0.3174 | 1.3948 | 0.3628 | 0.3545 | **0.2916** | 0.4663 |
| electricity | **0.4335** | 0.8981 | 0.4634 | 0.4761 | 0.4956 | 0.4595 |
| exchange_rate | **0.8803** | 2.3152 | 1.0709 | 1.8533 | 1.7726 | 1.7202 |
| **Overall Avg** | **0.3088** | 0.7603 | 0.3769 | 0.4954 | 0.4539 | 0.4549 |

---

## Table 2 — Unimodal Models

| Model | Description |
|-------|-------------|
| BYOL-T | BYOL temporal-only, nano, new lotsa dataset |
| SimCLR-T-nano | SimCLR temporal-only, nano, new lotsa dataset |
| SimCLR-V-nano | SimCLR visual-only, nano, new lotsa dataset |
| SimCLR-T-old | SimCLR temporal-only, old dataset |

| Dataset | BYOL-T | SimCLR-T-nano | SimCLR-V-nano | SimCLR-T-old |
| --- | --- | --- | --- | --- |
| ETTm1 | **0.1438** | 0.3130 | 0.2685 | 0.2890 |
| ETTm2 | **0.1616** | 0.2301 | 0.2199 | 0.2224 |
| ETTh1 | **0.1605** | 0.2886 | 0.2741 | 0.2515 |
| ETTh2 | **0.2179** | 0.2452 | 0.2285 | 0.2607 |
| weather | 0.0324 | 0.0332 | **0.0219** | 0.0867 |
| traffic | **0.3048** | 0.3685 | 0.3976 | 0.7497 |
| electricity | 0.4278 | 0.4801 | **0.4175** | 0.6989 |
| exchange_rate | 1.3156 | **1.0431** | 1.8002 | 1.2910 |
| **Overall Avg** | **0.3456** | 0.3752 | 0.4535 | 0.4812 |

---

## Notes

- **BYOL-T** is surprisingly competitive — 2nd best overall, wins on all ETT datasets and traffic
- **SimCLR-Full** leads overall but BYOL-T beats it on 4/8 datasets
- No CLIP full training on new dataset yet (only micro fine-tune)
- Gram-Full does not scale well — collapses on traffic and exchange_rate
