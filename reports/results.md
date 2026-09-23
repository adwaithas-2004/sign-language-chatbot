Official INCLUDE-50 test split: 192 videos (trained on 689, validated on 77).

| Model | Top-1 | Top-3 | Macro-F1 |
|---|---|---|---|
| Baseline: logistic regression on summary features (deterministic) | 94.3% | 97.9% | 0.934 |
| BiGRU on landmark sequences, mean ± std over 5 training runs | 95.5% ± 0.8 | 99.0% ± 0.5 | 0.953 ± 0.008 |
| **BiGRU, saved model** (best validation accuracy of the 5 runs) | **96.4%** | 98.4% | 0.962 |
| INCLUDE paper, best model on INCLUDE-50 | 94.5% | – | – |

Averaged over 5 runs, the BiGRU scores +1.3 points top-1 against the baseline (run-to-run standard deviation 0.8 points).

BiGRU inference: 4.7 ms per sign on CPU. Hardest words for the saved model: girl (50%), court (67%), fall (67%), boy (75%), hot (83%).

INCLUDE's 7 signers appear in every split, so these are *seen-signer* results. Accuracy for a new signer and camera will be lower.
