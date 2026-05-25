# Background-noise model: deeper analysis

Follow-up to the v2 analysis, focused exclusively on the background-noise model. Anomaly detection is ignored.

## 1. AR order selection (AIC / BIC)

We fit AR(p) for p in 1, 2, 3, 5, 10 on the event-masked clean residual of each month. Lower AIC/BIC = better fit (BIC penalises higher orders more heavily).

### 24/02/2015 11:20

| p | AIC | BIC | innovation std | innov skew | innov exc kurt |
|---:|---:|---:|---:|---:|---:|
| 1 | 92290 | 92315 | 1.095 | +0.10 | +0.02 |
| 2 | 92285 | 92318 | 1.095 | +0.10 | +0.02 |
| 3 | 92284 | 92326 | 1.095 | +0.10 | +0.01 |
| 5 | 92277 | 92336 | 1.095 | +0.10 | +0.02 |
| 10 | 92250 | 92350 | 1.095 | +0.10 | +0.02 |

Best by AIC: **AR(10)**. Best by BIC: **AR(1)**.

### 30/04/2015 13:10

| p | AIC | BIC | innovation std | innov skew | innov exc kurt |
|---:|---:|---:|---:|---:|---:|
| 1 | 32563 | 32585 | 1.106 | +0.07 | +0.09 |
| 2 | 32562 | 32591 | 1.106 | +0.07 | +0.09 |
| 3 | 32558 | 32595 | 1.106 | +0.07 | +0.09 |
| 5 | 32554 | 32605 | 1.105 | +0.07 | +0.09 |
| 10 | 32547 | 32634 | 1.105 | +0.07 | +0.09 |

Best by AIC: **AR(10)**. Best by BIC: **AR(1)**.

### 22/06/2015 07:31

| p | AIC | BIC | innovation std | innov skew | innov exc kurt |
|---:|---:|---:|---:|---:|---:|
| 1 | 35441 | 35463 | 1.144 | +0.09 | -0.06 |
| 2 | 35434 | 35463 | 1.144 | +0.09 | -0.06 |
| 3 | 35431 | 35468 | 1.144 | +0.09 | -0.06 |
| 5 | 35424 | 35475 | 1.144 | +0.09 | -0.06 |
| 10 | 35412 | 35500 | 1.143 | +0.09 | -0.05 |

Best by AIC: **AR(10)**. Best by BIC: **AR(1)**.

### 20/10/2015 09:18

| p | AIC | BIC | innovation std | innov skew | innov exc kurt |
|---:|---:|---:|---:|---:|---:|
| 1 | 67300 | 67324 | 1.130 | +0.06 | +0.00 |
| 2 | 67286 | 67318 | 1.130 | +0.06 | +0.00 |
| 3 | 67282 | 67322 | 1.130 | +0.06 | -0.00 |
| 5 | 67275 | 67331 | 1.130 | +0.06 | -0.00 |
| 10 | 67263 | 67359 | 1.129 | +0.06 | -0.00 |

Best by AIC: **AR(10)**. Best by BIC: **AR(2)**.

## 2. Whitening quality (Ljung-Box on innovations)

Null hypothesis: innovations have zero autocorrelation up to the test lag (= the model captures all serial structure). p > 0.05 means we fail to reject -> the model is well specified.

| Month | order p | LB(10) p-value | LB(20) p-value | LB(50) p-value |
|---|---:|---:|---:|---:|
| 24/02/2015 11:20 | 1 | 0.00256 | 0.0104 | 0.129 |
| 24/02/2015 11:20 | 2 | 0.0243 | 0.0396 | 0.255 |
| 24/02/2015 11:20 | 3 | 0.023 | 0.0413 | 0.262 |
| 24/02/2015 11:20 | 5 | 0.0201 | 0.0449 | 0.28 |
| 24/02/2015 11:20 | 10 | 0.888 | 0.927 | 0.952 |
| 30/04/2015 13:10 | 1 | 0.588 | 0.715 | 0.748 |
| 30/04/2015 13:10 | 2 | 0.626 | 0.757 | 0.769 |
| 30/04/2015 13:10 | 3 | 0.635 | 0.764 | 0.777 |
| 30/04/2015 13:10 | 5 | 0.672 | 0.762 | 0.778 |
| 30/04/2015 13:10 | 10 | 0.999 | 0.987 | 0.943 |
| 22/06/2015 07:31 | 1 | 0.025 | 0.0462 | 0.018 |
| 22/06/2015 07:31 | 2 | 0.0765 | 0.144 | 0.0485 |
| 22/06/2015 07:31 | 3 | 0.167 | 0.206 | 0.0695 |
| 22/06/2015 07:31 | 5 | 0.709 | 0.526 | 0.155 |
| 22/06/2015 07:31 | 10 | 0.993 | 0.992 | 0.51 |
| 20/10/2015 09:18 | 1 | 0.0141 | 0.0181 | 0.16 |
| 20/10/2015 09:18 | 2 | 0.363 | 0.163 | 0.429 |
| 20/10/2015 09:18 | 3 | 0.47 | 0.272 | 0.555 |
| 20/10/2015 09:18 | 5 | 0.68 | 0.559 | 0.75 |
| 20/10/2015 09:18 | 10 | 0.952 | 0.954 | 0.95 |

## 3. AR(1) innovation distribution

Is the innovation eps_t really Gaussian, or is it heavy-tailed? A Student-t with finite degrees of freedom would indicate heavy tails.

| Month | innov skew | innov exc kurt | t df | t scale |
|---|---:|---:|---:|---:|
| 24/02/2015 11:20 | +0.10 | +0.02 | 398.0 | 1.093 |
| 30/04/2015 13:10 | +0.07 | +0.09 | 69.8 | 1.090 |
| 22/06/2015 07:31 | +0.09 | -0.06 | 259335297426.6 | 1.144 |
| 20/10/2015 09:18 | +0.06 | +0.00 | 6896.7 | 1.130 |

*Excess kurtosis* near zero = Gaussian. *t df* near infinity = Gaussian; t df near 4-6 = heavy-tailed.

## 4. Parameter stability across the month

Each month is split in 4 equal chunks; AR(1) is re-fitted on each. If phi and sigma_innov drift across the month, the noise is non-stationary and a single AR(1) won't reproduce its full behaviour.

### 24/02/2015 11:20

| chunk | n clean | phi | innov_std | marginal_std |
|---:|---:|---:|---:|---:|
| 0 | 7928 | 0.891 | 1.132 | 2.498 |
| 1 | 9829 | 0.875 | 1.084 | 2.237 |
| 2 | 9829 | 0.864 | 1.082 | 2.152 |
| 3 | 9832 | 0.878 | 1.124 | 2.349 |

Range: phi in [0.864, 0.891] (spread 0.027); sigma_innov in [1.082, 1.132] (spread 0.049).

### 30/04/2015 13:10

| chunk | n clean | phi | innov_std | marginal_std |
|---:|---:|---:|---:|---:|
| 0 | 6883 | 0.945 | 1.184 | 3.619 |
| 1 | 7147 | 0.865 | 1.093 | 2.182 |
| 2 | 8078 | 0.866 | 1.100 | 2.201 |
| 3 | 7043 | 0.866 | 1.132 | 2.262 |

Range: phi in [0.865, 0.945] (spread 0.079); sigma_innov in [1.093, 1.184] (spread 0.091).

### 22/06/2015 07:31

| chunk | n clean | phi | innov_std | marginal_std |
|---:|---:|---:|---:|---:|
| 0 | 3336 | 0.877 | 1.114 | 2.316 |
| 1 | 9829 | 0.859 | 1.143 | 2.230 |
| 2 | 7159 | 0.851 | 1.143 | 2.174 |
| 3 | 3819 | 0.909 | 1.177 | 2.820 |

Range: phi in [0.851, 0.909] (spread 0.058); sigma_innov in [1.114, 1.177] (spread 0.063).

### 20/10/2015 09:18

| chunk | n clean | phi | innov_std | marginal_std |
|---:|---:|---:|---:|---:|
| 0 | 5825 | 0.924 | 1.160 | 3.031 |
| 1 | 2598 | 0.847 | 1.109 | 2.087 |
| 2 | 9829 | 0.861 | 1.118 | 2.198 |
| 3 | 9832 | 0.867 | 1.130 | 2.270 |

Range: phi in [0.847, 0.924] (spread 0.077); sigma_innov in [1.109, 1.160] (spread 0.051).

## 5. Cross-month consistency

If the same physical noise process produces all four months, AR(1) phi and sigma_innov should be similar across months.

| Month | AR(1) phi | innov std | marginal std |
|---|---:|---:|---:|
| 24/02/2015 11:20 | 0.872 | 1.095 | 2.235 |
| 30/04/2015 13:10 | 0.867 | 1.106 | 2.222 |
| 22/06/2015 07:31 | 0.856 | 1.144 | 2.216 |
| 20/10/2015 09:18 | 0.864 | 1.130 | 2.243 |

## 6. Baseline drift

Linear fit on the ultra-wide MA-17280 baseline; ADF tests the null hypothesis that the baseline is non-stationary (p<0.05 => stationary, p>0.05 => unit root / drift).

| Month | slope/sample | total drift | wide-base range | ADF p |
|---|---:|---:|---:|---:|
| 24/02/2015 11:20 | +7.51e-05 | +2.95 | 2.83 | 0.998 |
| 30/04/2015 13:10 | +2.14e-05 | +0.84 | 1.53 | 0.944 |
| 22/06/2015 07:31 | +9.85e-05 | +3.87 | 3.04 | 0.0781 |
| 20/10/2015 09:18 | -1.02e-04 | -3.99 | 3.44 | 0.938 |

## 7. Conclusions

1. **Best AR order.** See section 1. If AR(1) is consistently the AIC/BIC minimum, the v2 conclusion holds and the generator should use AR(1). If a higher order is preferred, the generator needs to match it.
2. **Innovation distribution.** See section 3. If excess kurt is ~0 and t df is high, Gaussian innovations are fine.
3. **Stationarity.** See section 4. If phi/sigma drift across the month, the generator should sample them per-recording from the empirical distribution rather than using point estimates.
