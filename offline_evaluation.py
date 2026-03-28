import scipy.io
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import os

# ===============================
# SETUP
# ===============================
os.makedirs('results', exist_ok=True)

print("--- Loading Dataset & Model ---")
mat_data = scipy.io.loadmat('mod_data.mat')
mod_types = [str(m.item()).strip() for m in mat_data['modTypes'].flatten()]
raw_signals = mat_data['allSignals']
model = tf.keras.models.load_model('mod_classifier_model.h5')

# ===============================
# PREPARE DATA
# ===============================
X, Y = [], []
for m_idx in range(len(mod_types)):
    for f_idx in range(raw_signals.shape[0]):
        sig = raw_signals[f_idx, :, m_idx]
        iq_frame = np.stack((np.real(sig), np.imag(sig)), axis=1)
        iq_frame = (iq_frame - np.mean(iq_frame)) / (np.std(iq_frame) + 1e-6)
        X.append(iq_frame)
        Y.append(m_idx)

X = np.array(X, dtype=np.float32)
Y = np.array(Y, dtype=np.int64)

_, X_test, _, y_test = train_test_split(
    X, Y, test_size=0.2, stratify=Y, random_state=42
)

y_pred_probs = model.predict(X_test)
y_pred = np.argmax(y_pred_probs, axis=1)

# ===============================
# GRAPH 1: CONFUSION MATRIX
# ===============================
print("\n--- Generating Confusion Matrix ---")
cm = confusion_matrix(y_test, y_pred)
cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

plt.figure(figsize=(10, 8))
sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues',
            xticklabels=mod_types, yticklabels=mod_types)
plt.title('Confusion Matrix — Normalized\n(Overall Accuracy: {:.1f}%)'.format(
    np.mean(y_pred == y_test) * 100))
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.savefig('results/confusion_matrix.png', dpi=150)
plt.close()
print(" Saved: results/confusion_matrix.png")

# ===============================
# GRAPH 2: TRAINING CURVES
# ===============================
print("\n--- Generating Training Curves ---")

# Re-train briefly just to capture history
# (Only if you don't have history saved — we'll train fresh)
X_train, X_val, y_train, y_val = train_test_split(
    X, Y, test_size=0.1, stratify=Y, random_state=42
)

def build_model(input_shape, num_classes):
    model = tf.keras.models.Sequential([
        tf.keras.layers.Input(shape=input_shape),
        tf.keras.layers.Conv1D(128, 7, activation='relu', padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling1D(2),
        tf.keras.layers.Conv1D(64, 5, activation='relu', padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling1D(2),
        tf.keras.layers.Conv1D(32, 3, activation='relu', padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.GlobalAveragePooling1D(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    return model

train_model = build_model((1024, 2), len(mod_types))
train_model.compile(optimizer='adam',
                    loss='sparse_categorical_crossentropy',
                    metrics=['accuracy'])

history = train_model.fit(
    X_train, y_train,
    epochs=30,
    batch_size=64,
    validation_data=(X_val, y_val),
    verbose=1
)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Accuracy
ax1.plot(history.history['accuracy'], label='Train', linewidth=2, color='blue')
ax1.plot(history.history['val_accuracy'], label='Validation',
         linewidth=2, color='orange', linestyle='--')
ax1.set_title('Model Accuracy over Epochs')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Accuracy')
ax1.legend()
ax1.grid(True, alpha=0.3)
ax1.set_ylim([0, 1])

# Loss
ax2.plot(history.history['loss'], label='Train', linewidth=2, color='blue')
ax2.plot(history.history['val_loss'], label='Validation',
         linewidth=2, color='orange', linestyle='--')
ax2.set_title('Model Loss over Epochs')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Loss')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.suptitle('CNN Training Curves — AMC Project', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('results/training_curves.png', dpi=150)
plt.close()
print("Saved: results/training_curves.png")

# ===============================
# GRAPH 3: ACCURACY VS SNR
# ===============================
print("\n--- Generating Accuracy vs SNR Curve ---")
mat_data2 = scipy.io.loadmat('mod_data.mat')
raw_signals2 = mat_data2['allSignals']

snr_values = range(-10, 31, 5)
snr_accuracies = []

for snr_target in snr_values:
    X_snr, Y_snr = [], []
    for m_idx in range(len(mod_types)):
        # Take 50 frames per class at this SNR level
        for f_idx in range(min(50, raw_signals2.shape[0])):
            sig = raw_signals2[f_idx, :, m_idx]
            # Add specific SNR noise
            sig_power = np.mean(np.abs(sig)**2)
            noise_power = sig_power / (10**(snr_target/10))
            noise = np.sqrt(noise_power/2) * (
                np.random.randn(len(sig)) + 1j*np.random.randn(len(sig))
            )
            sig_noisy = sig + noise
            iq_frame = np.stack((np.real(sig_noisy),
                                  np.imag(sig_noisy)), axis=1)
            iq_frame = (iq_frame - np.mean(iq_frame)) / (
                np.std(iq_frame) + 1e-6)
            X_snr.append(iq_frame)
            Y_snr.append(m_idx)

    X_snr = np.array(X_snr, dtype=np.float32)
    Y_snr = np.array(Y_snr, dtype=np.int64)
    preds = np.argmax(train_model.predict(X_snr, verbose=0), axis=1)
    acc = np.mean(preds == Y_snr) * 100
    snr_accuracies.append(acc)
    print(f"  SNR {snr_target:+3d} dB → Accuracy: {acc:.1f}%")

plt.figure(figsize=(10, 6))
plt.plot(list(snr_values), snr_accuracies,
         marker='o', linewidth=2.5, color='darkblue',
         markersize=8, markerfacecolor='orange')
plt.axhline(y=90, color='red', linestyle='--',
            alpha=0.7, label='90% threshold')
plt.title('CNN Classification Accuracy vs SNR\n(AMC Project)', fontsize=14)
plt.xlabel('SNR (dB)', fontsize=12)
plt.ylabel('Accuracy (%)', fontsize=12)
plt.xticks(list(snr_values))
plt.ylim([0, 105])
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig('results/accuracy_vs_snr.png', dpi=150)
plt.close()
print(" Saved: results/accuracy_vs_snr.png")

# ===============================
# GRAPH 4: PER-CLASS BAR CHART
# ===============================
print("\n--- Generating Per-Class Bar Chart ---")
per_class_acc = cm_norm.diagonal() * 100

colors = ['#2196F3', '#4CAF50', '#FF9800', '#E91E63', '#9C27B0', '#00BCD4']
plt.figure(figsize=(10, 6))
bars = plt.bar(mod_types, per_class_acc, color=colors,
               edgecolor='black', linewidth=0.8)

# Add value labels on top of bars
for bar, acc in zip(bars, per_class_acc):
    plt.text(bar.get_x() + bar.get_width()/2.,
             bar.get_height() + 0.5,
             f'{acc:.1f}%', ha='center', va='bottom',
             fontweight='bold', fontsize=11)

plt.axhline(y=90, color='red', linestyle='--',
            alpha=0.7, label='90% threshold')
plt.title('Per-Class Classification Accuracy\n(AMC CNN Model)',
          fontsize=14, fontweight='bold')
plt.xlabel('Modulation Type', fontsize=12)
plt.ylabel('Accuracy (%)', fontsize=12)
plt.ylim([0, 110])
plt.legend()
plt.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig('results/per_class_accuracy.png', dpi=150)
plt.close()
print("Saved: results/per_class_accuracy.png")

# ===============================
# TERMINAL SUMMARY
# ===============================
print("\n" + "="*50)
print("   OFFLINE EVALUATION COMPLETE")
print("="*50)
print(f"\n Overall Test Accuracy: {np.mean(y_pred == y_test)*100:.2f}%")
print("\n Detailed Report:")
print(classification_report(y_test, y_pred, target_names=mod_types))
print("\n Saved Results:")
print("   results/confusion_matrix.png")
print("   results/training_curves.png")
print("   results/accuracy_vs_snr.png")
print("   results/per_class_accuracy.png")
print("="*50)