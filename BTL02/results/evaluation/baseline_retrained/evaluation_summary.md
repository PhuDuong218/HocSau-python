# Baseline evaluation: test

Chi xem la test doc lap neu tap nay chua tung duoc dung de chon model/hyperparameters.

- Samples: 7178
- Accuracy: 48.41%
- Macro-F1: 0.3793
- Weighted-F1: 0.4613
- Cross-entropy: 1.3401

## Most frequent errors

- sad -> neutral: 460
- angry -> neutral: 281
- fear -> neutral: 269
- angry -> sad: 215
- fear -> sad: 201

Filter visualization shows mean signed conv3 weights across input channels.
It does not show activation maps or establish which facial regions the model uses.
Use predictions.csv to inspect examples before explaining causes of errors.
