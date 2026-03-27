import scipy.io
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split

# 1. Load & Normalize Data
mat_data = scipy.io.loadmat('mod_data.mat')
mod_types = [str(m.item()).strip() for m in mat_data['modTypes'].flatten()]
raw_signals = mat_data['allSignals'] 

X, Y = [], []
for m_idx in range(len(mod_types)):
    for f_idx in range(raw_signals.shape[0]):
        sig = raw_signals[f_idx, :, m_idx]
        iq_frame = np.stack((np.real(sig), np.imag(sig)), axis=1)
        
        # --- THE FIX: Standardize each frame (Mean 0, Std 1) ---
        iq_frame = (iq_frame - np.mean(iq_frame)) / (np.std(iq_frame) + 1e-6)
        
        X.append(iq_frame)
        Y.append(m_idx)

X = np.array(X, dtype=np.float32)
Y = np.array(Y, dtype=np.int64)

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, stratify=Y)

# 2. Build the "Robust" CNN
def build_model(input_shape, num_classes):
    model = models.Sequential([
        layers.Input(shape=input_shape),
        layers.Conv1D(128, 7, activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling1D(2),
        
        layers.Conv1D(64, 5, activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling1D(2),
        
        layers.Conv1D(32, 3, activation='relu', padding='same'),
        layers.BatchNormalization(),
        
        # This replaces Flatten() to prevent the 38% accuracy issue
        layers.GlobalAveragePooling1D(), 
        
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.5), # High dropout to force the model to be smart
        layers.Dense(num_classes, activation='softmax')
    ])
    return model

# 3. Train
model = build_model((1024,2), len(mod_types))
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

print("\n--- Training on Improved Realistic Data ---")
model.fit(X_train, y_train, epochs=30, batch_size=64, validation_split=0.1)

# 4. Final Verdict
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
print(f'\nNew Test Accuracy: {test_acc*100:.2f}%')
model.save('mod_classifier_model.h5')