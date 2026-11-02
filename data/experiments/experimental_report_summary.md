# Automated Experimental & Ablation Report

## 1. E1 - E4 Comparative Evaluation (ms_conflict_bench)

| Experiment             | Dataset           |   Accuracy |   Precision |   Recall |   F1_Score |   AUROC |    FPR |    ECE |   Brier_Score |   Conflict_Acc |   Avg_LLM_Calls |   Avg_Latency_ms |   Avg_Tokens |
|:-----------------------|:------------------|-----------:|------------:|---------:|-----------:|--------:|-------:|-------:|--------------:|---------------:|----------------:|-----------------:|-------------:|
| E1_ORIGINAL_BASELINE   | ms_conflict_bench |     0.5    |      0      |   0      |     0      |  0.8272 | 0      | 0.3566 |        0.3264 |         0.6667 |           0     |            1.714 |         0    |
| E2_CLAIM_LEVEL_NLI     | ms_conflict_bench |     0.6111 |      0.5714 |   0.8889 |     0.6957 |  0.7531 | 0.6667 | 0.1755 |        0.2293 |         0.6667 |           0     |            1.305 |         0    |
| E3_HYBRID_VERIFICATION | ms_conflict_bench |     0.6111 |      0.5714 |   0.8889 |     0.6957 |  0.7531 | 0.6667 | 0.2553 |        0.2675 |         0.6667 |           1.278 |            1.056 |       194.22 |
| E4_PROPOSED_ACVE       | ms_conflict_bench |     0.8333 |      0.75   |   1      |     0.8571 |  0.8642 | 0.3333 | 0.1822 |        0.1565 |         0.7222 |           0.611 |            1.87  |       122.78 |

## 2. 7-Variant Component Ablation Study

| Ablation_Variant                 |   Accuracy |   Precision |   Recall |   F1_Score |   Delta_F1_vs_Full |   AUROC |    ECE |   Conflict_Acc |   Avg_LLM_Calls |   Avg_Latency_ms |
|:---------------------------------|-----------:|------------:|---------:|-----------:|-------------------:|--------:|-------:|---------------:|----------------:|-----------------:|
| 1_without_claim_decomposition    |     0.9444 |      0.9    |   1      |     0.9474 |             0.0903 |  0.963  | 0.0873 |         0.7222 |           0.389 |            1.308 |
| 2_without_nli_verification       |     0.6667 |      0.6    |   1      |     0.75   |            -0.1071 |  0.7778 | 0.242  |         0.6667 |           1.278 |            1.159 |
| 3_without_conflict_detection     |     0.6111 |      0.5714 |   0.8889 |     0.6957 |            -0.1614 |  0.7531 | 0.2607 |         0.6667 |           0.556 |            0.868 |
| 4_without_adaptive_llm_judging   |     0.6667 |      0.6    |   1      |     0.75   |            -0.1071 |  0.8272 | 0.2847 |         0.7222 |           0     |            1.294 |
| 5_without_confidence_aggregation |     0.8333 |      0.75   |   1      |     0.8571 |             0      |  0.8889 | 0.1578 |         0.7222 |           0.611 |            1.371 |
| 6_without_source_reliability     |     0.8333 |      0.75   |   1      |     0.8571 |             0      |  0.7901 | 0.1956 |         0.7222 |           0.611 |            1.182 |
| 7_complete_proposed_acve         |     0.8333 |      0.75   |   1      |     0.8571 |             0      |  0.8642 | 0.1941 |         0.7222 |           0.611 |            1.117 |
