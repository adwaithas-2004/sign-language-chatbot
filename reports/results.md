Official INCLUDE-50 test split: 192 videos (trained on 689, validated on 77).

| Model | Top-1 | Top-3 | Macro-F1 |
|---|---|---|---|
| Baseline: logistic regression on summary features (deterministic) | 94.3% | 97.9% | 0.934 |
| BiGRU on landmark sequences, mean ± std over 5 training runs | 96.1% ± 1.1 | 98.8% ± 0.3 | 0.958 ± 0.011 |
| **BiGRU, saved model** (best validation accuracy of the 5 runs) | **97.4%** | 99.0% | 0.972 |
| INCLUDE paper, best model on INCLUDE-50 | 94.5% | – | – |

Averaged over 5 runs, the BiGRU scores +1.9 points top-1 against the baseline (run-to-run standard deviation 1.1 points).

BiGRU inference: 5.1 ms per sign on CPU. Hardest words for the saved model: court (67%), fall (67%), storeorshop (75%), hot (83%), trainticket (88%).

INCLUDE's 7 signers appear in every split, so these are *seen-signer* results. Accuracy for a new signer and camera will be lower.
