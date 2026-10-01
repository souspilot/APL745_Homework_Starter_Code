# HW2 — One question: CNNs for anonymous acoustic events (100 marks)

You are given 360 five-second, mono, 16 kHz audio recordings in `dataset/audio/` and a `dataset/manifest.csv` with columns `id`, `split`, and `label`. The ten labels `C00`–`C09` are arbitrary codes. Your task is to classify the recordings. Use the named `train` split (240 recordings) for fitting, `validation` (80) for model selection, and `test` (40) **once** for final evaluation. The splits are balanced by class. The instructor has a separate, unlabeled-to-you evaluation set. Do not infer a split from the order or names of files.

The question has six connected parts. Submit one notebook or a small, reproducible code project, a short report (at most four pages), trained model checkpoints/configurations/seeds, and an inference entry point as described below. Report all choices that materially affect the results. You may use PyTorch or another deep-learning framework, but train the classifiers from scratch; no pretrained audio or vision model. The companion `audio_lab.ipynb` is for exploration and is not a required submission.

## (a) Represent the same sounds in two ways (15 marks)

1. Load the waveform, check sample rate and duration, scale it to floating point, and show one waveform and one **log-magnitude mel spectrogram** from the same recording. Explain what is retained and lost in each representation, including the role of phase.
2. State all spectrogram choices: window, hop, FFT length, number of mel bands, frequency range, log compression, and any resizing or normalization. Calculate any normalization statistics on training data only.
3. Explain why a waveform is suited to `Conv1d` and a time-frequency image to `Conv2d`. State the input tensor shapes.

## (b) Train two CNN baselines (20 marks)

Train a small 1D CNN on waveforms and a small 2D CNN on log-mel spectrograms. Each must have at least two convolutional layers, a nonlinearity, downsampling, and a classifier head. Keep the models small enough to train on the supplied data, and document layer-by-layer tensor shapes and parameter counts. Use cross-entropy loss. Use the **same splits and evaluation metric** for both models, and state what makes their comparison imperfect (for example, different input sizes or parameter counts).

## (c) Modern architecture and regularization (20 marks)

Improve the 2D model using a compact **Inception-style block**: parallel branches with different kernel sizes, at least one `1×1` projection, concatenation along channels, and a global-average-pooling classifier. Show that the branch outputs have compatible spatial sizes and compare its parameter count with a plain convolutional alternative. Use batch normalization after convolution and before the activation, and use dropout in the classifier head. Explain the train/evaluation behavior of batch normalization and dropout, and why augmentation acts differently from either one. You may use a small residual block instead of the Inception-style block if you clearly explain the skip path and dimension matching.

## (d) Augmentation and controlled comparison (20 marks)

Apply at least **two appropriate audio augmentations** to training samples only, such as a short time shift, mild gain change, or low-level additive noise. If you augment spectrograms, explain why the transformation still represents a plausible sound. Do not augment validation or test examples. Keep one fixed training recipe and compare the improved 2D model under these four conditions:

| Run | Batch normalization | Dropout | Augmentation |
| --- | --- | --- | --- |
| Full | On | On | On |
| No BN | Off | On | On |
| No dropout | On | Off | On |
| No augmentation | On | On | Off |

Use identical splits, preprocessing, epoch budget, seed policy, and checkpoint criterion. Show training and validation loss curves and a compact results table. Discuss overfitting and at least one interaction or limitation of these single-factor ablations. The “Full” run can be the model from part (c).

## (e) Backpropagation through a convolution (15 marks)

Frameworks implement CNN backpropagation, but you must work through one case by hand. A single valid 2D **cross-correlation** layer (the convention used by most deep-learning libraries) has input

```text
X = [[1, 2, 0],       K = [[ 1, 0],       b = 0
     [0, 1, 3],            [-1, 1]]
     [2, 1, 1]]
```

Let `Z = X ⋆ K + b`, `A = ReLU(Z)`, target `Y = [[1, 0], [1, 2]]`, and `L = 1/2 Σᵢⱼ(Aᵢⱼ − Yᵢⱼ)²`. Compute `Z`, `A`, `L`, `∂L/∂Z`, `∂L/∂K`, and `∂L/∂b`, showing how the same kernel weights accumulate gradients from all output positions. Verify one kernel-gradient entry with an automatic differentiation or finite-difference check. Use the ReLU derivative zero for negative preactivations.

## (f) Evaluation and interpretation (10 marks)

Choose hyperparameters using `validation` only. For the selected waveform and spectrogram models, report `test` accuracy, macro-F1, a confusion matrix, and at least two inspected errors. Include runtime or hardware and one limitation of conclusions from this small test set. Do not tune on `test`. Describe whether the learned result supports any claim about waveform versus spectrogram input; distinguish observation from general conclusion.

### Reproducibility and integrity

Include package versions, random seeds, exact train/validation/test counts, preprocessing, optimizer, learning rate, batch size, epoch budget, checkpoint rule, and a command or notebook order that reproduces the results. Files sharing a source recording may be similar; preserve the supplied splits to avoid leakage. Keep class codes in the report. The dataset attribution and license are in `ATTRIBUTION.txt`.

For instructor evaluation, submit a `predict.py` entry point that loads your chosen **spectrogram** model checkpoint and accepts `--audio-dir`, `--manifest`, and `--output`. The input CSV will have `id` and `split` columns but **no labels**; audio files are named by `id`. Write a CSV with exactly `id,predicted_label` (one row per input, using `C00`–`C09`). The instructor will run this on the separate private set. Your code must not retrain or adjust preprocessing statistics during inference. Include a one-line command showing how to run it on the public test set.
