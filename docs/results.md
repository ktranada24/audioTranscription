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

# Interpretation

Training achieved near-perfect reconstruction on the training set but generalized poorly to the validation set.

Validation performance remained approximately 17.3% better than random character generation, indicating the model learned non-random acoustic-text relationships despite severe overfitting.

Next planned experiments:
- Expand dataset
- Bootstrap CER confidence intervals
- Compare decoding strategies
- Evaluate regularization methods


#2026-06-24 — LibriSpeech Pilot Experiment (Real Dataset Transition)

Setup

- Model: Character-level BiLSTM acoustic model
- Training Split:
  - Samples 0-99
- Validation Split
  - Samples 100-119
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
- Epochs: 40
- Train Set: 100 clips
- Validation Set: 20 clips
- Feature Cache: Enabled

 Results

| Metric                    | Value  |
|---|---:|
| Train CER                 | 0.2431 |
| Validation CER            | 0.5754 |
| Random Baseline CER.      | 0.9233 |
| Train Skill Score         | 0.7365 |
| Validation Skill Score    | 0.3768 |

# Interpretation

The model substantially outperformed random prediction and learned meaningful acoustic-to-text mappings on unseen speech.

Compared to the original 40-clip experiment, the transition to LibriSpeech improved generalization while maintaining strong training performance.

Validation CER decreased from 0.783 to 0.575 despite substantially longer utterances and larger transcript lengths.

Next Planned Experiments:
- Train longer (80 epochs)
- Scale to 500+ training samples
- Bootstrap CER confidence intervals
- Compare greedy vs beam-search decoding
- Track train/validation CER curves
- Evaluate regularization and augmentation

#2026-06-24 — LibriSpeech Experiment 02 500+ training samples

Setup

- Model: Character-level BiLSTM acoustic model
- Training Split:
  - Samples 0-499
- Validation Split
  - Samples 500-599
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
- Epochs: 40
- Train Set: 500 clips
- Validation Set: 100 clips
- Feature Cache: Enabled

 Results

| Metric                    | Value  |
|---|---:|
| Train CER                 | 0.0774 |
| Validation CER            | 0.4819 |
| Random Baseline CER.      | 0.9212 |
| Train Skill Score         | 0.9160 |
| Validation Skill Score    | 0.4768 |

# Interpretation 

Scaling the training dataset from 100 examples to 500 examples produced a substantial improvement in generalization performance.

The improvement from increasing dataset size exceeded the gains previously observed from increasing training duration alone, suggesting model performance remains data-limited rather than optimization-limited.

Next Planned Experiments

- Train 500-sample split for 80 epochs
- Track train/validation learning curves
- Compare greedy vs beam-search decoding
- Bootstrap CER confidence intervals
- Scale to 1000+ training samples
- Evaluate augmentation and regularization
