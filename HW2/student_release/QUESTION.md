# HW2 Q2: CNNs for anonymous acoustic events (100 marks)

You are given 360 five-second, mono, 16 kHz WAV recordings in `dataset/audio/`. The CSV `dataset/manifest.csv` has columns `id,split,label`. Its `train`, `validation`, and `test` splits contain 240, 80, and 40 recordings; each has the same ten classes `C00`–`C09`. The class codes are arbitrary. Use `train` to fit models, `validation` to choose settings and checkpoints, and `test` once for the results in your report. A separate 40-recording set is reserved for instructor grading.

## What to submit

Submit **one ZIP file** whose top level contains these required files; you may add a `src/` directory for helper modules:

```text
submission.zip
├── submission.py        # the single executable entry point
├── requirements.txt     # exact package versions used, one per line
├── report.pdf           # at most four pages, including figures and tables
└── src/                 # optional Python modules imported by submission.py
```

Do not put audio data, model weights, caches, or a virtual environment in the ZIP. Use Python 3.11 and PyTorch. Grading may run without network access and with only a CPU. Your code must not download models or data, use pretrained weights, read the private labels, or depend on the working directory being the data directory. All paths in the command below may be absolute.

The instructor will unpack your ZIP and run this **single command** from its top level in a fresh environment with your `requirements.txt` installed:

```bash
python submission.py \
  --data-root /path/to/HW2_student/dataset \
  --predict-manifest /path/to/hidden/input.csv \
  --predict-audio-dir /path/to/hidden/audio \
  --output /path/to/predictions.csv \
  --seed 745
```

`--data-root` contains `manifest.csv` and `audio/`. Your script must **start with randomly initialized weights**, train your selected final 2D CNN on `train` only, use `validation` for checkpoint selection, reload that checkpoint, and predict every row in `--predict-manifest`. The hidden input CSV has exactly `id,split` and no labels; the corresponding WAV file is `--predict-audio-dir/<id>`. The split value may be `instructor_test`; your predictor must not require a known split name or a `label` column. Do not fit normalization statistics or tune choices using `test` or the prediction set. The public `test` rows in `--data-root/manifest.csv` must be ignored by this grading command.

On success, create the parent of `--output` if needed, write a UTF-8 CSV with **exactly** the header `id,predicted_label`, and exit with code 0. Write one row per input in the same order, preserving each `id` byte-for-byte. Every prediction must be one of `C00`–`C09`; do not include probabilities, an index column, or extra rows. Use the supplied seed for Python, NumPy, and PyTorch random generators. A run may use at most **30 training epochs** for the final model and has a **15-minute wall-time limit** on the course grading machine. Print the selected epoch and validation accuracy to standard output. The instructor scores hidden accuracy and macro-F1 from the CSV; the report is graded separately. The command above runs the `full` condition by default. The same entry point must also accept `--experiment` with one of `waveform`, `spectrogram`, `full`, `no_bn`, `no_dropout`, or `no_augmentation` to retrain the corresponding report run from scratch using the same path arguments.

Exercise the command locally with an unlabeled CSV made from the 40 public `test` IDs. For example, make `public_input.csv` with columns `id,split` and run the same command with `--predict-manifest public_input.csv` and `--predict-audio-dir dataset/audio`. This checks the label-free interface before submission.

## The question

### (a) Represent the sounds (15 marks)

Load and listen to at least one `train` clip. Show its waveform and log-magnitude mel spectrogram. State your FFT length, window, hop, mel-band count, frequency range, log transform, tensor shapes, and any normalization. Compute any fitted statistics on `train` only. Explain what information each representation retains or loses, including phase, and why `Conv1d` and `Conv2d` fit the respective inputs. The optional `audio_lab.ipynb` demonstrates visual exploration.

### (b) Two CNN baselines (20 marks)

Train one waveform `Conv1d` model and one log-mel `Conv2d` model from scratch. Each needs at least two convolutional layers, nonlinearities, downsampling, and a classification head. Report layer-by-layer output shapes and trainable parameter counts. Use cross-entropy loss, the same `train`/`validation` split, and a checkpoint chosen by validation accuracy (break ties using lower validation loss). Explain at least one limitation of comparing their accuracies directly.

### (c) Modern architecture and regularization (20 marks)

Make a compact improved 2D model with either an Inception-style multi-kernel block (including a `1×1` projection and channel concatenation) or a residual block (including a dimension-matched skip path), followed by global average pooling. Use batch normalization after convolution and before activation, plus dropout in the classifier head. Show how the block preserves compatible tensor dimensions, compare its parameter count with a plain convolutional block, and explain training versus evaluation behavior for batch normalization and dropout. The improved model with both regularizers and augmentation is the final model used by `submission.py`.

### (d) Augmentation and ablation (20 marks)

Use at least two plausible audio augmentations on `train` only, for example short shifts, mild gain changes, or low-level noise. State their exact parameter ranges. If an augmentation acts on the spectrogram, explain why the result is acoustically plausible. Run the improved 2D model four times with the same split, preprocessing, epoch limit, checkpoint rule, and seed policy:

| Run | Batch normalization | Dropout | Augmentation |
| --- | --- | --- | --- |
| Full | On | On | On |
| No BN | Off | On | On |
| No dropout | On | Off | On |
| No augmentation | On | On | Off |

Show training and validation loss curves, a table of best validation accuracy and selected epoch for all four runs, and a short explanation of overfitting or an inconclusive result. The grading command runs only **Full**, but the `--experiment` option must reproduce the other five runs through the same entry point.

### (e) Backpropagation through a convolution (15 marks)

Work through a single valid 2D **cross-correlation** layer by hand:

```text
X = [[1, 2, 0],       K = [[ 1, 0],       b = 0
     [0, 1, 3],            [-1, 1]]
     [2, 1, 1]]
```

Let `Z = X ⋆ K + b`, `A = ReLU(Z)`, target `Y = [[1, 0], [1, 2]]`, and `L = 1/2 Σᵢⱼ(Aᵢⱼ − Yᵢⱼ)²`. Compute `Z`, `A`, `L`, `∂L/∂Z`, `∂L/∂K`, and `∂L/∂b`. Show how kernel gradients accumulate from all output positions. Verify one `∂L/∂K` entry using PyTorch autograd or a centered finite difference. Use ReLU derivative zero for negative preactivations.

### (f) Evaluation and interpretation (10 marks)

For the selected waveform baseline and Full 2D model, report public `test` accuracy, macro-F1, and confusion matrices. Inspect two errors. State hardware, package versions, seed, optimizer, learning rate, batch size, epoch limit, checkpoint rule, and total training time. Explain what the small public test set can and cannot establish about waveform versus spectrogram models. Your `report.pdf` must contain the results and derivation for parts (a)–(f); `submission.py` must satisfy the grading command above.

The dataset attribution and license are in `ATTRIBUTION.txt`. Keep the anonymous class codes in your report.
