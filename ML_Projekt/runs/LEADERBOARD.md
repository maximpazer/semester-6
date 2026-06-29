# Leaderboard

Baseline F1: 0.6938

| Run | Hypothesis | Dev Δ | Dev SE | Conf Δ | Conf SE | Decision | Time |
|-----|-----------|-------|--------|--------|---------|---------|------|
| 20260629_161541_extratrees_13feat | ExtraTrees (500 trees, balanced) auf 13  | +0.0002 | 0.0030 | — | — | REJECT | 2s |
| 20260629_161837_meta_logit | Logit-Transform S6 → LogReg(C=1) meta | +0.0000 | 0.0000 | — | — | REJECT |
| 20260629_161838_meta_C01 | LogReg(C=0.1) meta auf S6 (stärkere Regu | -0.0045 | 0.0049 | — | — | REJECT |
| 20260629_161839_meta_C001 | LogReg(C=0.01) meta auf S6 | -0.0134 | 0.0049 | — | — | REJECT |
| 20260629_161840_meta_rank | Rank-normierte S6-Inputs → LogReg(C=1) m | +0.0000 | 0.0000 | — | — | REJECT |
| 20260629_161841_meta_logit_C01 | Logit-Transform + LogReg(C=0.1) meta | -0.0104 | 0.0061 | — | — | REJECT |
| 20260629_161939_meta_logit | Logit-Transform S6 → LogReg(C=1) meta | -0.0009 | 0.0056 | — | — | REJECT |
| 20260629_161940_meta_C01 | LogReg(C=0.1) meta auf S6 (stärkere Regu | -0.0045 | 0.0049 | — | — | REJECT |
| 20260629_161941_meta_C001 | LogReg(C=0.01) meta auf S6 | -0.0134 | 0.0049 | — | — | REJECT |
| 20260629_161943_meta_rank | Rank-normierte S6-Inputs → LogReg(C=1) m | -0.0165 | 0.0050 | — | — | REJECT |
| 20260629_161944_meta_logit_C01 | Logit-Transform + LogReg(C=0.1) meta | -0.0113 | 0.0086 | — | — | REJECT |
| 20260629_162030_lgbm_native_nan | lgbm_native_nan | +0.0091 | 0.0034 | +0.0057 | 0.0008 | KEEP |
| 20260629_162158_catboost_native_nan | catboost_native_nan | +0.0042 | 0.0087 | — | — | REJECT |
| 20260629_162308_xgb_native_nan | xgb_native_nan | +0.0008 | 0.0010 | — | — | REJECT |
| 20260629_162409_nan_lgbm_cat_combined | S6 + LGBM_NaN + CatBoost_NaN combined | +0.0025 | 0.0070 | — | — | REJECT |
| 20260629_162409_nan_lgbm_cat_avg | S6 + avg(LGBM_NaN, CatBoost_NaN) | +0.0040 | 0.0048 | — | — | REJECT |
| 20260629_162602_lgbm_nan_seedbag3 | lgbm_nan_seedbag3 | +0.0046 | 0.0060 | — | — | REJECT |
