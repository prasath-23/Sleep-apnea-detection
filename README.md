# Sleep Apnea Detection from a Single-Lead ECG

Detecting sleep apnea, minute by minute, from **one** ECG wire — using a convolutional neural network that teaches itself what to look for.

<img src="architecture.png" alt="Model architecture" width="480" />

---

## Table of contents

1. [What problem is this solving?](#1-what-problem-is-this-solving)
2. [The core idea](#2-the-core-idea)
3. [How the data is prepared](#3-how-the-data-is-prepared)
4. [The architecture](#4-the-architecture)
5. [Results](#5-results)
6. [The other models we compared against](#6-the-other-models-we-compared-against)
7. [The serving API](#7-the-serving-api)
8. [Repository map](#8-repository-map)
9. [Getting it running](#9-getting-it-running)
10. [Glossary](#10-glossary)
11. [Limitations](#11-limitations)
12. [Credit and licence](#12-credit-and-licence)

---

## 1. What problem is this solving?

**Sleep apnea** is when a sleeping person repeatedly stops breathing for seconds at a time. It fragments sleep and, untreated, raises the risk of high blood pressure, heart disease and stroke. It is also very common — and mostly undiagnosed.

The standard diagnostic test, **polysomnography (PSG)**, means sleeping overnight in a lab wearing roughly twenty sensors while a trained technician watches. It is accurate, but expensive, uncomfortable and scarce.

**This project asks:** can a single ECG electrode — the sort of thing a chest patch or smartwatch already records — do the job instead?

```
   😴  person asleep, one ECG electrode
            │
            ▼
   ~∿~∿~∿~∿~∿~   raw heart signal, all night
            │
            ▼
        🤖 CNN
            │
            ▼
   minute 1  normal  ✅
   minute 2  normal  ✅
   minute 3  APNEA   ⚠️
   minute 4  APNEA   ⚠️
   minute 5  normal  ✅
```

---

## 2. The core idea

### Why the heart knows about the lungs

When breathing stops, the body reacts. Two of those reactions show up clearly in an ECG:

| Clue | What it means |
|---|---|
| **Heart-rate rhythm changes** | Oxygen drops, the nervous system reacts, and the heart slows then surges in a characteristic pattern. |
| **Heartbeat spike height changes** | The tall R-peak of each beat rises and falls as the chest and diaphragm move — so it tracks breathing effort. |

Together these two derived signals carry enough information to separate an apnea minute from a normal one.

### Why a neural network

Classical approaches required an ECG expert to hand-design features — heart-rate variability statistics, frequency-band powers, nonlinear measures — and pick which ones mattered. That expertise is rare and the choices are subjective.

A **convolutional neural network (CNN)** skips that step. Show it enough labelled examples and it discovers its own features directly from the signal.

### The key modification: look at the neighbours

This is what separates the method from a plain CNN classifier.

To label minute *N*, the model is not shown minute *N* alone. It is shown a **five-minute window**: the two minutes before, the target minute, and the two minutes after.

```
        ┌────┬────┬══════┬────┬────┐
window: │ N-2│ N-1│  N   │ N+1│ N+2│   ← what the network sees
        └────┴────┴══════┴────┴────┘
                     ↑
              the minute being labelled
```

Apnea events do not politely begin and end on minute boundaries, and the physiological response — the heart-rate surge, the recovery breath — often lands in the *following* minute. The surrounding context carries real evidence, and including it is what lifts the accuracy.

---

## 3. How the data is prepared

### The dataset

The public **[Apnea-ECG database](https://physionet.org/content/apnea-ecg/1.0.0/)** from PhysioNet:

| Split | Recordings | Notes |
|---|---|---|
| Train | 35 — `a01`–`a20`, `b01`–`b05`, `c01`–`c10` | Full nights, each **minute** labelled `A` (apnea) or `N` (normal) by an expert |
| Test | 35 — `x01`–`x35` | Held out; labels come from [`dataset/event-2-answers`](dataset/event-2-answers) |

ECG is sampled at **100 Hz** — 6,000 numbers per minute, 30,000 per five-minute window. Far too raw to feed a small network, hence the pipeline below.

### The preprocessing pipeline (`Preprocessing.py`)

For each five-minute window:

**1. Band-pass filter, 3–45 Hz.** An FIR filter that strips slow baseline drift and high-frequency noise, keeping the sharp QRS complexes.

**2. Detect R-peaks.** The Hamilton segmenter from `biosppy` finds the tall spike of each heartbeat, then a correction pass snaps each detection onto the true local maximum.

**3. Reject bad windows.** Fewer than 40 or more than 200 beats per minute means the detection failed or the signal is corrupt. Windows with physiologically impossible heart rates (outside 20–300 bpm) are **discarded entirely** rather than fed to the model.

**4. Derive the two signals.**

- **RRI** — the R-to-R interval, i.e. the gap in seconds between consecutive beats. A three-tap median filter removes single-beat glitches.
- **Amplitude** — the height of each R-peak.

```
raw ECG      ─┴─╮──┴─╮───┴─╮──┴─╮────┴─╮
                  ↓ detect R-peaks
peaks         │     │      │     │      │
                  ↓ measure
RRI (s)       0.81   0.94   0.78   1.02      ← heart rhythm
Amplitude(mV) 1.12   0.98   1.20   0.91      ← breathing effort
```

The whole processed dataset is pickled to `dataset/apnea-ecg-database-1.0.0/apnea-ecg.pkl`. Run this **once**; every training script just loads the pickle. The script uses a process pool, one record per worker.

### Making every sample the same shape

Heartbeats are irregular, so windows yield differing numbers of RRI values — but a network needs fixed-size input. The fix is **cubic-spline interpolation** onto a uniform 3 Hz grid, plus min–max normalisation into 0–1.

```
5 min × 60 s × 3 samples/s  =  900 time steps
                            ×    2 channels (RRI, amplitude)
                            →  input shape (900, 2)
```

---

## 4. The architecture

A **modified LeNet-5**. LeNet-5 (LeCun et al., 1998) was the classic digit-recognition network operating on 2D images; this is the same skeleton rebuilt in **1D**, sliding along time instead of across a picture.

```
   input (900, 2)
        │
   ┌────▼─────────────────────────────────────┐
   │ Conv1D · 32 filters · kernel 5 · stride 2│  learn short beat-to-beat patterns
   │ ReLU · He init · L2(1e-3)                │
   └────┬─────────────────────────────────────┘
   ┌────▼──────────────┐
   │ MaxPooling1D · 3  │                         keep the strongest response, shrink 3×
   └────┬──────────────┘
   ┌────▼─────────────────────────────────────┐
   │ Conv1D · 64 filters · kernel 5 · stride 2│  compose them into longer-range patterns
   │ ReLU · He init · L2(1e-3)                │
   └────┬─────────────────────────────────────┘
   ┌────▼──────────────┐
   │ MaxPooling1D · 3  │
   └────┬──────────────┘
   ┌────▼──────────────┐
   │ Dropout · 0.8     │                         hard regularisation
   └────┬──────────────┘
   ┌────▼──────────────┐
   │ Flatten           │
   │ Dense 32 · ReLU   │
   │ Dense 2 · softmax │                       → [P(normal), P(apnea)]
   └───────────────────┘
```

**Each piece, in plain English:**

- **Conv1D** — a small window (5 samples wide) slides along the signal hunting for one specific shape. Thirty-two filters means thirty-two different shapes learned in parallel. `stride 2` makes it hop two samples at a time, halving the length as it goes.
- **MaxPooling1D(3)** — of every three neighbouring values, keep only the largest. Shortens the sequence and makes the model tolerant of small timing shifts.
- **Dropout(0.8)** — during training, randomly silence most of the units. Counter-intuitive but effective: it stops the network leaning on any single pathway, forcing redundant, generalisable representations.
- **L2 regularisation** — a penalty on large weights, same goal: generalise instead of memorise. Applied to both kernels and biases.
- **He initialisation** — the weight-initialisation scheme matched to ReLU activations, so gradients neither vanish nor explode at the start.
- **Softmax** — converts the two final numbers into probabilities summing to 1.

**Training setup:** Adam optimiser · categorical cross-entropy · batch size 32 · 20 epochs · a `LearningRateScheduler` that decays the rate ×0.1 in the late epochs so the model settles rather than bounces.

---

## 5. Results

Evaluated on the held-out test nights (`x01`–`x35`), which the model never sees during training. Figures below are recomputed from the committed `output/LeNet.csv` — **16,945 test minutes**, 35 subjects, decision threshold 0.5:

| Metric | Score | Reading it in plain English |
|---|---|---|
| **Accuracy** | **83.4 %** | Of every 100 minutes, about 83 are labelled correctly. |
| **Sensitivity** (recall) | **88.6 %** | Of every 100 genuine apnea minutes, about 89 are caught. |
| **Specificity** | **80.2 %** | Of every 100 genuine normal minutes, about 80 are correctly left alone. |

Sensitivity is deliberately the higher of the two — in screening, **missing a sick patient costs far more than a false alarm**, since a flagged patient simply goes on to a proper sleep study.

Comparable to classical machine-learning results on this dataset, with **zero hand-engineered features**.

`LeNet.py` also prints a confusion matrix, a full classification report and the AUC-ROC, and writes:

- `output/LeNet.csv` — per-minute `y_true`, `y_score`, `subject`
- `performance_curvesle.png` — training/validation loss and accuracy curves
- `models/model.final.h5` — the trained weights

Curves for the other models: `performance_curvesGLN.png` (GoogLeNet), `performance_curveshyb.png` (hybrid).

---

## 6. The other models we compared against

Same inputs, same splits, different architectures — so the comparison is fair.

| Script | Model | The idea |
|---|---|---|
| **`LeNet.py`** | **Modified LeNet-5, 1D** | The main method. Small, fast, ~4 layers deep. |
| `Pre-training.py` | Classic LeNet-5, 2D | The literal 1998 architecture — Conv2D + average pooling, dense 120 → 84 → 2 — as a baseline to beat. |
| `GoogleLeNet.py` | 1D GoogLeNet (Inception) | Much deeper. Each **inception module** examines the signal with 1-, 3- and 5-wide filters *simultaneously* plus a pooled branch, then concatenates them — so short and long patterns are captured at once. Five modules, then global average pooling. |
| `hybrid.py` | Inception + LSTM | The inception stack extracts features; two **LSTM** layers then read them as a sequence, modelling how the signal evolves over time. Adds batch normalisation, early stopping on validation accuracy, and best-model checkpointing. |

**Why add an LSTM?** A CNN recognises shapes; an LSTM remembers **order**. Apnea recurs in cycles through the night, so a memory of what came before can help — at the cost of a much slower model.

---

## 7. The serving API

`app.py` is a Flask service (CORS enabled, so a web front-end can call it directly).

| Endpoint | Method | What it does |
|---|---|---|
| `/predict` | POST | One five-minute window in → `Apnea` / `Non-Apnea` plus the probability. |
| `/detect_points` | POST | Slides a 30-second window with a 10-second step across a longer signal and returns **when** apnea was detected, with start time, end time and probability for each hit. |
| `/metrics` | GET | Test-set loss, accuracy, confusion matrix, classification report, and the ROC curve rendered as a base64 PNG. |

Request body for both POST endpoints — four arrays, **at least 4 points each** (cubic splines need four knots):

```json
{
  "rri_tm":      [0.0, 0.81, 1.75, 2.53],
  "rri_signal":  [0.81, 0.94, 0.78, 1.02],
  "ampl_tm":     [0.0, 0.81, 1.75, 2.53],
  "ampl_signal": [1.12, 0.98, 1.20, 0.91]
}
```

Response:

```json
{ "prediction": "Apnea", "apnea_probability": 0.87 }
```

Note that `app.py` currently **trains its model on startup** before serving, so the first launch is slow.

---

## 8. Repository map

### Scripts

| File | Purpose |
|---|---|
| `Preprocessing.py` | **Run first.** Raw PhysioNet records → filtered signal → R-peaks → RRI + amplitude → `apnea-ecg.pkl`. Parallelised across CPU cores. |
| `LeNet.py` | Train and evaluate the modified LeNet-5. The main experiment. |
| `Pre-training.py` | Classic 2D LeNet-5 baseline → `models/custom_lenet5_model.h5`. |
| `GoogleLeNet.py` | Train and evaluate the 1D Inception network. |
| `hybrid.py` | Train and evaluate the Inception + LSTM network. |
| `app.py` | Flask REST API (see above). |
| `patientdata.py` | Whole-patient report: counts events, computes the **AHI** and prints a positive/negative verdict for one subject. |
| `Report.py` | Regenerates the figures used in the write-up — signals, feature maps, confusion matrix, ROC. |
| `extract.py` | Dumps the pickled dataset back out to CSV for inspection. |
| `analyse.py` | Dataset statistics over `output/sleep_data.csv`. Mostly commented-out exploratory experiments; treat as a scratchpad. |

### Data and outputs

| Path | Contents |
|---|---|
| `dataset/apnea-ecg-database-1.0.0/` | The PhysioNet records (download separately) and the generated `apnea-ecg.pkl`. |
| `dataset/event-1-answers`, `event-2-answers` | Ground-truth labels for the challenge test set. |
| `models/`, `*.h5` | Saved trained weights. |
| `output/LeNet.csv` | Per-minute test predictions from the main model. |
| `output/sleep_data.csv` | Per-recording summary: minutes, AI, HI, AHI, age, sex, height, weight. |
| `evaluation_results.csv` | Per-sample true label, predicted label and probability. |
| `utils/` | The official per-recording scoring code from the PhysioNet challenge. |
| `architecture.png`, `performance_curves*.png`, `training_results.png` | Figures. |

---

## 9. Getting it running

### Install dependencies

```bash
pip install numpy scipy scikit-learn pandas matplotlib seaborn \
            tensorflow keras wfdb biosppy tqdm netron flask flask-cors
```

### Get the data

Download the Apnea-ECG records from [PhysioNet](https://physionet.org/content/apnea-ecg/1.0.0/) into:

```
dataset/apnea-ecg-database-1.0.0/
```

All paths in the scripts resolve **relative to the repository**, so no editing is needed and you can run them from any working directory.

### Run

```bash
# 1. Build the processed dataset — once, and it takes a while
python Preprocessing.py

# 2. Train and evaluate the main model
python LeNet.py

# 3. Or try the alternatives
python Pre-training.py
python GoogleLeNet.py
python hybrid.py

# 4. Per-patient AHI report
python patientdata.py

# 5. Serve predictions over HTTP
python app.py            # http://127.0.0.1:5000
```

Example call:

```bash
curl -X POST http://127.0.0.1:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"rri_tm":[0,0.81,1.75,2.53],"rri_signal":[0.81,0.94,0.78,1.02],
       "ampl_tm":[0,0.81,1.75,2.53],"ampl_signal":[1.12,0.98,1.20,0.91]}'
```

### A note on versions

`LeNet.py` and `GoogleLeNet.py`/`hybrid.py` open a **[Netron](https://netron.app/)** viewer on the saved model, which blocks the terminal — comment out the `netron.start(...)` line for unattended runs.

Keras and TensorFlow change optimiser internals between releases and both carry inherent randomness, so your numbers will move a little. The original paper used **Keras 2.3.1 / TensorFlow 1.15.0**; these scripts target the TensorFlow 2.x `tensorflow.keras` API.

---

## 10. Glossary

| Term | Meaning |
|---|---|
| **ECG / EKG** | Electrocardiogram — the electrical trace of the heartbeat. |
| **R-peak** | The tall spike of each heartbeat in an ECG. |
| **RRI** | R-to-R interval — seconds between consecutive heartbeats. |
| **PSG** | Polysomnography — the full overnight sleep study with ~20 sensors. |
| **AI / HI / AHI** | Apnea Index, Hypopnea Index, and the two combined, counted **per hour of sleep**. AHI ≥ 5 mild, ≥ 15 moderate, ≥ 30 severe. |
| **CNN** | Convolutional neural network — learns local patterns with sliding filters. |
| **LSTM** | Long short-term memory — a recurrent layer that remembers sequence order. |
| **Sensitivity** | Fraction of genuine positives correctly caught. |
| **Specificity** | Fraction of genuine negatives correctly left alone. |
| **AUC-ROC** | One number for how well the model *ranks* positives above negatives, independent of threshold. 0.5 = coin flip, 1.0 = perfect. |
| **Epoch** | One full pass over the training data. |

---

## 11. Limitations

- **Small, dated data.** 35 training nights from a 1990s database — limited in age range, ethnicity, comorbidity and recording hardware. Generalisation to modern wearables is untested here.
- **Minute-level labels only.** The model cannot genuinely localise a 20-second event. `/detect_points` approximates it with a shorter sliding window, but the model was never trained at that resolution — treat its output as indicative, not precise.
- **Noisy windows are dropped, not classified.** A real device has to decide what to do with signal it cannot trust; this pipeline simply discards it.
- **Apnea vs. normal only.** No distinction between obstructive, central and mixed apnea, and hypopnea is not modelled separately.
- **Not a medical device.** This is a research project. It diagnoses no one. Real diagnosis requires a clinician.

---

## 12. Credit and licence

Method and dataset preparation follow:

> Wang T, Lu C, Shen G, et al. *Sleep apnea detection from a single-lead ECG signal with automatic feature-extraction through a modified LeNet-5 convolutional neural network.* **PeerJ**, 2019, 7: e7731. https://doi.org/10.7717/peerj.7731

Dataset: **Apnea-ECG Database**, PhysioNet — https://physionet.org/content/apnea-ecg/1.0.0/

Released under the terms in [LICENSE](LICENSE).
