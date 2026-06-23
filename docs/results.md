# ASR Experiment Results

# 2026-06-23 — Baseline BiLSTM CTC Evaluation

Setup

- Model: Character-level BiLSTM acoustic model
- Input Features:
  - Framing (25 ms)
  - Hop Length (10 ms)
  - Hamming Window
  - FFT
  - 80-bin Log-Mel Spectrogram
- Decoder: Greedy CTC
- Loss: CTC Loss
- Optimizer: AdamW
- Learning Rate: 0.001
- Weight Decay: 1e-4
- Batch Size: 1
- Gradient Clipping: max_norm=5.0
- Epochs: 300
- Train Set: 40 clips
- Validation Set: 10 clips
- Feature Cache: Enabled

 Results

| Metric               | Value  |
|---|---:|
| Train CER            | 0.0129 |
| Validation CER       | 0.7832 |
| Random Baseline CER. | 0.9472 |
| Skill Score vs Random| 0.1731 |

### Interpretation

Training achieved near-perfect reconstruction on the training set but generalized poorly to the validation set.

Validation performance remained approximately 17.3% better than random character generation, indicating the model learned non-random acoustic-text relationships despite severe overfitting.

Next planned experiments:
- Expand dataset
- Bootstrap CER confidence intervals
- Compare decoding strategies
- Evaluate regularization methods