import scipy.io
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

# 1. LOAD AND PREPROCESS DATA
print("--- Loading MATLAB Dataset ---")
try:
    mat_data = scipy.io.loadmat('mod_data.mat')
except FileNotFoundError:
    print("Error: mod_data.mat not found! Run the MATLAB script first.")
    exit()

mod_types = [str(m.item()).strip() for m in mat_data['modTypes'].flatten()]
raw_signals = mat_data['allSignals'] # Shape: (Frames, 1024, Mods)

X, Y = [], []
for m_idx in range(len(mod_types)):
    for f_idx in range(raw_signals.shape[0]):
        sig = raw_signals[f_idx, :, m_idx]
        
        # Extract I and Q channels
        iq_frame = np.stack((np.real(sig), np.imag(sig)), axis=1) # (1024, 2)
        
        # --- CRITICAL: Z-Score Normalization ---
        # Ensures all signals have mean=0 and std=1 regardless of MATLAB fading/scaling
        iq_frame = (iq_frame - np.mean(iq_frame)) / (np.std(iq_frame) + 1e-6)
        
        X.append(iq_frame)
        Y.append(m_idx)

X = np.array(X, dtype=np.float32)
Y = np.array(Y, dtype=np.int64)

# Split: 80% Train, 20% Test (Stratified to keep class balance)
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=42)

# 2. BUILD THE ROBUST CNN MODEL
def build_robust_model(input_shape, num_classes):
    model = models.Sequential([
        layers.Input(shape=input_shape),
        
        # Layer 1: Feature Extraction
        layers.Conv1D(128, 7, activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling1D(2),
        
        # Layer 2: Deep Pattern Recognition
        layers.Conv1D(64, 5, activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling1D(2),
        
        # Layer 3: Final Spatial Features
        layers.Conv1D(32, 3, activation='relu', padding='same'),
        layers.BatchNormalization(),
        
        # Global Pooling makes the model "Position Independent" (Industry Standard)
        layers.GlobalAveragePooling1D(), 
        
        # Dense Layers with high Dropout to prevent Overfitting
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.5), 
        layers.Dense(num_classes, activation='softmax')
    ])
    return model

# 3. TRAINING
model = build_robust_model((1024, 2), len(mod_types))
model.compile(optimizer='adam', 
              loss='sparse_categorical_crossentropy', 
              metrics=['accuracy'])

print("\n--- Starting Training (30 Epochs) ---")
history = model.fit(X_train, y_train, 
                    epochs=30, 
                    batch_size=64, 
                    validation_split=0.1,
                    verbose=1)

# 4. OFFLINE EVALUATION (The Report Part)
print("\n--- Evaluating Model ---")
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
print(f"Final Test Accuracy: {test_acc*100:.2f}%")

# Generate Predictions
y_pred_probs = model.predict(X_test)
y_pred = np.argmax(y_pred_probs, axis=1)

# Generate Text Report (Precision, Recall, F1)
print("\nDetailed Classification Report (Copy this into your report):")
print(classification_report(y_test, y_pred, target_names=mod_types))

# Generate Confusion Matrix Visual
cm = confusion_matrix(y_test, y_pred)
cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] # Normalized for %

plt.figure(figsize=(10, 8))
sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues', 
            xticklabels=mod_types, yticklabels=mod_types)

plt.title(f'Offline Evaluation: Confusion Matrix\n(Accuracy: {test_acc*100:.1f}%)')
plt.ylabel('True Label (MATLAB)')
plt.xlabel('Predicted Label (CNN)')
plt.tight_layout()

# Save for your project report
plt.savefig('offline_test_results.png')
print("\nSuccess! Plot saved as 'offline_test_results.png'")

# 5. SAVE MODEL
model.save('mod_classifier_model.h5')
print("Model saved as 'mod_classifier_model.h5'")
plt.show()
