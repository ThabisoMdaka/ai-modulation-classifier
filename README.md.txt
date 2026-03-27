# AI-Based Automatic Modulation Classifier (AMC)

A deep learning system that automatically identifies the modulation 
type of a received radio signal using a CNN trained on simulated I/Q samples.

>  **Project Status: Under Active Development**

---

##  Project Goal

Detect modulation types (AM, FM, PM, BPSK, QPSK, BFSK) from raw I/Q 
signal samples — mimicking a real SDR receiver, without needing hardware.

---

##  Progress So Far

| Phase | Task | Status |
|-------|------|--------|
| 1 | MATLAB signal generator (6 modulations, 1000 frames each) |  Done |
| 1 | Realistic channel effects (AWGN, Rayleigh fading, freq offset) |  Done |
| 2 | Python data pipeline (6000 samples, stratified split) |  Done |
| 2 | CNN model training |  Done — **99.08% accuracy** |
| 3 | GNU Radio real-time signal stream | Coming soon |
| 3 | Live inference + constellation visualization |  Coming soon |

---

## 📁 Project Structure
```
ai-modulation-classifier/
├── generate_modulation_dataset.m   # MATLAB: generates I/Q signal dataset
├── prepare_dataset.py              # Python: loads .mat file, formats for CNN
├── train_model.py                  # Python: CNN architecture + training
└── README.md
```

---

##  CNN Architecture

- Input: I/Q samples shaped (1024, 2)
- Conv1D (128 filters) → BatchNorm → MaxPool
- Conv1D (64 filters) → BatchNorm → MaxPool
- Conv1D (32 filters) → Dropout(0.3)
- Dense(128) → Softmax(6 classes)

---

##  Results

- **Test Accuracy: 99.08%**
- Dataset: 6000 frames (1000 per modulation type)
- Split: 80% train / 10% val / 10% test

---

##  Tech Stack

- **MATLAB** — Signal generation & channel simulation
- **Python** — Data pipeline & deep learning
- **TensorFlow / Keras** — CNN model
- **NumPy / SciPy** — Signal processing
- **GNU Radio + ZeroMQ** — Live SDR simulation (planned)

---

##  Author

**Thabiso Mdaka** — Electronic Engineering Student
Building an SDR signal classifier from scratch 