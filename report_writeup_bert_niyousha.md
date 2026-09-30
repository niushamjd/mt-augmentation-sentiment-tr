## Classifier: BERT (dbmdz/bert-base-turkish-cased)

**Architecture.** Full fine-tuning (no frozen layers) of `dbmdz/bert-base-turkish-cased`
with a linear classification head over the pooled `[CLS]` representation, 2 output
classes (negative/positive). Max sequence length 128, linear learning-rate warmup over
10% of total steps, early stopping on dev macro-F1 with patience 3 evaluations
(one per epoch) and a 5-epoch cap. Every run restores the best-dev-F1 checkpoint before
scoring on the frozen test set, per the shared protocol.

**Hyperparameters tried and rejected.** Tuned on C1 only, seed 42, dev macro-F1 only
(never test), per Section 1.5. Grid: lr ∈ {2e-5, 3e-5, 5e-5} at batch size 16, plus
3e-5 at batch size 32.

| lr | batch size | best dev macro-F1 | best epoch | train time (s) |
|---|---|---|---|---|
| 2e-5 | 16 | 0.8920 | 4 | 382.1 |
| 3e-5 | 16 | 0.8940 | 5 | 374.8 |
| 5e-5 | 16 | 0.8850 | 4 | 373.4 |
| **3e-5** | **32** | **0.8940** | 4 | 341.6 |

3e-5/batch-32 tied 3e-5/batch-16 on dev F1 but converged one epoch earlier and trained
faster, so it was selected and used unchanged for every subsequent condition (C2, C3,
C2b, RQ3, RQ4), per the "tune once on C1" rule.

**Results (test set, tuned lr=3e-5, batch=32).**

| condition | mt_system | synth_ratio | seed | test acc | test macro-F1 | F1 neg / pos |
|---|---|---|---|---|---|---|
| C1 (real only) | – | 0 | 42 | 0.877 | 0.8770 | 0.876 / 0.878 |
| C1 (real only) | – | 0 | 1337 | 0.869 | 0.8690 | 0.869 / 0.869 |
| C2 (real+all synth) | final | 10.0 | 42 | 0.871 | 0.8710 | 0.870 / 0.872 |
| C2 (real+all synth) | final | 10.0 | 1337 | 0.8735 | 0.8735 | 0.871 / 0.876 |
| C3 (real+filtered synth) | final | 6.54 | 42 | 0.875 | 0.8750 | 0.876 / 0.874 |
| C3 (real+filtered synth) | final | 6.54 | 1337 | 0.8705 | 0.8705 | 0.873 / 0.868 |
| C2b (real+size-matched unfiltered) | final | 6.54 | 42 | 0.871 | 0.8710 | 0.870 / 0.872 |
| C2b (real+size-matched unfiltered) | final | 6.54 | 1337 | 0.8665 | 0.8665 | 0.865 / 0.868 |
| RQ4: C2 with early-checkpoint MT | early | 10.0 | 42 | 0.8585 | 0.8584 | 0.855 / 0.862 |
| RQ4: C2 with pretrained MT (bonus tier) | pretrained | 10.0 | 42 | 0.8745 | 0.8744 | 0.871 / 0.878 |
| RQ3: synth at 1x real size | final | 1.0 | 42 | 0.8715 | 0.8715 | 0.870 / 0.873 |
| RQ3: synth at 5x real size | final | 5.0 | 42 | 0.871 | 0.8710 | 0.872 / 0.870 |

**What surprised me.** Two things. First, the C1 real-only baseline (0.877 acc) is
already so close to every synthetic-augmented condition (0.866–0.875) that none of
C2/C3/C2b beats it by more than about half a point on the good seed, and C2b actually
*loses* to it on seed 1337 — for this classifier, translation-augmented data barely
moves the needle over just fine-tuning BERT on the 2,000 real examples directly, which
runs against the premise that low-resource Turkish sentiment would benefit substantially
from MT augmentation. Second, MT quality mattered far less than I expected in one
direction and far more in the other: the early-checkpoint (undertrained) MT system
dropped test accuracy by ~1.9 points (0.877→0.8585, and notably hurt f1_neg the most),
confirming garbage-in/garbage-out for augmentation data — but going the other way, the
much stronger pretrained OPUS-MT system (RQ4 bonus tier) only recovered to 0.8745,
still *not* beating the plain real-only baseline. So MT quality has a floor effect
(bad MT clearly hurts) but not much of a ceiling effect (great MT doesn't clearly help)
for a model as strong as BERT starting from only 2,000 labeled real examples — the
classifier itself may simply be strong enough that the synthetic data's main value is
in a lower-resource regime than the one we're testing, which fits the RQ3 curve staying
essentially flat from 1x to 10x (2,000 → 20,000 synthetic examples): 0.8715 / 0.8710 /
0.871, no clear scaling benefit at all.
