#  AI-Based Automatic Modulation Classifier (AMC)

![Status](https://img.shields.io/badge/Status-Under%20Active%20Development-orange)
![Accuracy](https://img.shields.io/badge/CNN%20Accuracy-97.1%25-brightgreen)
![Python](https://img.shields.io/badge/Python-3.10-blue)
![MATLAB](https://img.shields.io/badge/MATLAB-R2023-red)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)

A research-grade deep learning system that automatically identifies 
the modulation type of a received radio signal using a Convolutional 
Neural Network (CNN) trained on realistic simulated I/Q samples.

> This project simulates a complete Software Defined Radio (SDR) 
> receiver pipeline entirely in software — no hardware required.

---

## Project Goal

Design an AI system capable of classifying 6 radio modulation types:

| Modulation | Type | Use Case |
|------------|------|----------|
| AM | Analog | Broadcast Radio |
| FM | Analog | Music Radio |
| PM | Analog | Phase Modulation |
| BPSK | Digital | Satellite / GPS |
| QPSK | Digital | 4G / Wi-Fi |
| BFSK | Digital | IoT / Bluetooth |

The system must remain accurate under **real-world channel conditions** 
including noise, fading, and hardware imperfections.

---

## System Architecture
```
┌─────────────────────────────────────────────────────────┐
│                  COMPLETE AMC PIPELINE                   │
├──────────────┬──────────────────┬───────────────────────┤
│   PHASE 1    │     PHASE 2      │       PHASE 3         │
│  Data Gen    │   AI Training    │    Live Inference     │
│  (MATLAB)    │   (Python/CNN)   │  (GNU Radio)        │
├──────────────┼──────────────────┼───────────────────────┤
│ • 6 mod types│ • Z-score norm   │ • ZeroMQ stream       │
│ • Rician     │ • Conv1D CNN     │ • Real-time predict   │
│   fading     │ • BatchNorm      │ • Constellation plot  │
│ • Pulse      │ • GlobalAvgPool  │ • Live confidence     │
│   shaping    │ • 97.1% accuracy │   display             │
│ • AWGN noise │ • Confusion      │                       │
│ • Freq/Phase │   matrix eval    │                       │
│   offsets    │ •offline testing │                       │
│ • Clock drift│                  │                       │
└──────────────┴──────────────────┴───────────────────────┘
```

---

##  Development Progress

| Phase | Task | Status |
|-------|------|--------|
| 1 | MATLAB V3 signal generator |  Complete |
| 1 | Rician fading + multipath |  Complete |
| 1 | Raised cosine pulse shaping | Complete |
| 1 | Random freq/phase/clock offsets per frame | Complete |
| 1 | AWGN across SNR range (-10 to 30 dB) | Complete |
| 2 | Python data pipeline (12,000 I/Q frames) | Complete |
| 2 | Z-score normalization |  Complete |
| 2 | CNN training (30 epochs) |  Complete |
| 2 | Confusion matrix evaluation |  Complete |
| 2 | **97.1% test accuracy** |  Complete |
| 3 | GNU Radio real-time stream | 🔄 In Progress |
| 3 | Live inference + visualization | 🔄 In Progress |

---

##  Results

### Confusion Matrix
![Confusion Matrix](results/confusion_matrix.png)


### Per-Class Accuracy
| Modulation | Accuracy |
|------------|----------|
| AM | 99% |
| FM | 98% |
| PM | 99% |
| BPSK | 95% |
| QPSK | 93% |
| BFSK | 98% |
| **Overall** | **97.1%** |

> BPSK and QPSK show slight mutual confusion — expected behavior 
> since both are phase-shift keying variants. This matches findings 
> in published AMC research literature.

---

##  CNN Architecture
```
Input: (1024, 2)  ← I and Q channels
    │
    ▼
Conv1D(128, kernel=7) → BatchNorm → MaxPool(2)
    │
    ▼
Conv1D(64, kernel=5) → BatchNorm → MaxPool(2)
    │
    ▼
Conv1D(32, kernel=3) → BatchNorm
    │
    ▼
GlobalAveragePooling1D  ← Position-independent features
    │
    ▼
Dense(64) → Dropout(0.5)
    │
    ▼
Softmax(6)  ← Final classification
```

**Key design decisions:**
- `GlobalAveragePooling` instead of `Flatten` — makes model robust 
  to signal position within the frame
- `Z-score normalization` — removes amplitude bias from fading
- `Dropout(0.5)` — prevents overfitting on noisy data

---

##  Realistic Channel Effects (MATLAB V3)

This is what separates this dataset from toy examples:
```
Raw Signal
    │
    ▼
Rician Fading (K=4)     ← Direct path + scattered multipath
    │
    ▼
Frequency Offset        ← Simulates SDR oscillator drift
    │
    ▼
Phase Offset            ← Random per frame (0 to 2π)
    │
    ▼
Clock Drift (±0.05%)    ← Hardware timing inaccuracy
    │
    ▼
Power Normalization     ← Unit energy per frame
    │
    ▼
AWGN (-10 to 30 dB)    ← Thermal noise floor
    │
    ▼
Saved I/Q Frame (1024 samples)
```

---

##  Project Structure
```
ai-modulation-classifier/
│
├── generate_modulation_dataset_realistic.m  
│       MATLAB V3: Generates research-grade I/Q dataset
│       with Rician fading, pulse shaping, clock drift
│
├── train_and_evaluate.py                    
│       Python: Loads dataset, trains CNN, evaluates
│       with confusion matrix and classification report
│
├── results/
│       confusion_matrix.png   ← Saved evaluation output
│
├── .gitignore                 ← Excludes .mat, .h5 files
└── README.md
```

---

##  How to Run

### Step 1 — Generate Dataset (MATLAB)
```matlab
% Open MATLAB and run:
generate_modulation_dataset_realistic.m
% Output: mod_data.mat (not tracked by Git — too large)
```

### Step 2 — Train & Evaluate (Python)
```bash
# Install dependencies
pip install tensorflow scipy numpy matplotlib seaborn scikit-learn

# Run training
python train_and_evaluate.py

# Output:
# - mod_classifier_model.h5
# - results/confusion_matrix.png
# - Classification report in terminal
```

---

## 🛠️ Tech Stack

| Tool | Purpose |
|------|---------|
| MATLAB R2023 | Signal generation & channel simulation |
| Python 3.10 | Data pipeline & model training |
| TensorFlow / Keras | CNN architecture |
| NumPy / SciPy | Signal processing & .mat loading |
| Scikit-learn | Train/test split & metrics |
| Seaborn / Matplotlib | Confusion matrix visualization |
| GNU Radio + ZeroMQ | Live SDR simulation *(planned)* |

---

##  Background & Motivation

Automatic Modulation Classification (AMC) is a critical capability 
in modern cognitive radio and spectrum monitoring systems. Traditional 
approaches rely on expert-crafted features (cyclostationary analysis, 
higher-order statistics). This project demonstrates that a CNN trained 
directly on raw I/Q samples can match or exceed those methods — a 
finding consistent with the DeepSig / RadioML research direction.

This project was built without SDR hardware by accurately simulating 
the full RF channel in MATLAB, making it fully reproducible and 
accessible for research purposes.

---

##  Author

**Thabiso Mdaka**  
BSc Electronic Engineering — University of KwaZulu-Natal  
Interests: DSP · Embedded AI · Telecommunications · SDR  

[![GitHub](https://img.shields.io/badge/GitHub-ThabisoMdaka-black)](https://github.com/ThabisoMdaka)