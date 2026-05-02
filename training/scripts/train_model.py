"""
train_model.py — IndicSignAI Landmark-Based Model Training
===========================================================
Key changes for pipeline alignment:

1.  Uses extract_and_normalize_landmarks() from src/utils/landmark_utils.py
    so that features written to keypoint.csv are IDENTICAL to the vectors
    computed at inference time in src/app.py.

2.  Provides build_keypoint_csv() to re-generate keypoint.csv from the raw
    image dataset with the aligned normalisation.

3.  AugmentedLandmarkGenerator applies Gaussian coordinate noise + random
    scale during model.fit() so the model generalises to real-world variation.
"""

import sys
import os

# Allow imports from src/ when running from training/scripts/
_repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(_repo_root, 'src'))

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (accuracy_score, precision_score,
                             recall_score, f1_score, classification_report)
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import json

# ── Universal landmark pipeline (must match src/app.py exactly) ──────────────
from utils.landmark_utils import extract_and_normalize_landmarks, LANDMARK_FEATURE_DIM


# =============================================================================
# COORDINATE AUGMENTATION GENERATOR
# =============================================================================

class AugmentedLandmarkGenerator(keras.utils.Sequence):
    """Keras data generator that applies coordinate-level augmentation.

    Instead of pixel-space image augmentation this operates on the normalised
    (63,) landmark vectors.  Two augmentations are applied per batch:

    1.  Gaussian noise   — adds ±noise_std random jitter to every coordinate.
        Simulates natural hand tremor and imprecise signing.
        Default ±0.02 (2 % of the bounding-box range).

    2.  Random scale     — multiplies all coordinates by a factor drawn from
        U[1-scale_range, 1+scale_range], then clips to [0, 1].
        Simulates the user signing slightly larger or smaller than in training.
        Default ±5 % (scale_range=0.05).

    Usage
    -----
        gen = AugmentedLandmarkGenerator(X_train, y_train, batch_size=128,
                                         noise_std=0.02, scale_range=0.05)
        model.fit(gen, epochs=100, validation_data=(X_val, y_val), ...)
    """

    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        batch_size: int = 128,
        noise_std: float = 0.02,
        scale_range: float = 0.05,
        augment: bool = True,
        shuffle: bool = True,
    ):
        self.X          = X.astype(np.float32)
        self.y          = y
        self.batch_size = batch_size
        self.noise_std  = noise_std
        self.scale_range = scale_range
        self.augment    = augment
        self.shuffle    = shuffle
        self.indices    = np.arange(len(self.X))
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.X) / self.batch_size))

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)

    def __getitem__(self, idx):
        batch_idx = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]
        X_batch = self.X[batch_idx].copy()
        y_batch = self.y[batch_idx]

        if self.augment:
            # 1. Gaussian coordinate noise — simulates hand tremor
            noise = np.random.normal(0.0, self.noise_std, X_batch.shape).astype(np.float32)
            X_batch = X_batch + noise

            # 2. Random scale per sample — simulates distance variation
            scales = np.random.uniform(
                1.0 - self.scale_range,
                1.0 + self.scale_range,
                size=(len(X_batch), 1)
            ).astype(np.float32)
            X_batch = X_batch * scales

            # Clamp back to valid normalised range
            X_batch = np.clip(X_batch, 0.0, 1.0)

        return X_batch, y_batch


# =============================================================================
# KEYPOINT CSV BUILDER
# =============================================================================

def build_keypoint_csv(
    image_dataset_dir: str,
    output_csv: str = 'keypoint.csv',
    mp_confidence_gate: float = 0.80,
):
    """Scan a labelled image dataset and write keypoint.csv.

    Expected directory layout:
        <image_dataset_dir>/
            <class_name_1>/
                img001.jpg
                img002.jpg
            <class_name_2>/
                ...

    Each row written to keypoint.csv:
        label, x0, y0, z0, x1, y1, z1, ..., x20, y20, z20
        (1 + 63 = 64 columns)

    The landmark extraction uses extract_and_normalize_landmarks() — the same
    function used in src/app.py — guaranteeing pipeline symmetry.
    """
    rows = []
    class_dirs = sorted([
        d for d in os.listdir(image_dataset_dir)
        if os.path.isdir(os.path.join(image_dataset_dir, d))
    ])

    print(f"Building keypoint.csv from {len(class_dirs)} classes in {image_dataset_dir}")

    for class_name in class_dirs:
        class_path = os.path.join(image_dataset_dir, class_name)
        images = [f for f in os.listdir(class_path)
                  if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]

        accepted = 0
        for img_file in images:
            img_path = os.path.join(class_path, img_file)
            frame = cv2.imread(img_path)
            if frame is None:
                continue

            # ── USE THE UNIVERSAL FUNCTION ──
            landmark_vec, _ = extract_and_normalize_landmarks(
                frame, mp_confidence_gate=mp_confidence_gate
            )
            if landmark_vec is None:
                continue   # no hand detected or below confidence gate

            row = [class_name] + landmark_vec.tolist()
            rows.append(row)
            accepted += 1

        print(f"  {class_name:30s}: {accepted}/{len(images)} frames accepted")

    df = pd.DataFrame(rows)
    df.to_csv(output_csv, index=False, header=False)
    print(f"\nSaved {len(rows)} rows → {output_csv}")
    return output_csv


# =============================================================================
# TRAINER
# =============================================================================

class SignLanguageTrainer:
    def __init__(self, data_path):
        self.data_path     = data_path
        self.model         = None
        self.history       = None
        self.label_encoder = LabelEncoder()

    def load_and_preprocess_data(self):
        """Load keypoint.csv and split into train / test."""
        print("Loading data...")
        data = pd.read_csv(self.data_path, header=None)
        data[0] = data[0].astype(str)

        print(f"Dataset shape: {data.shape}")
        print(f"Classes: {data[0].unique()}")
        print(f"Class distribution:\n{data[0].value_counts()}")

        X = data.iloc[:, 1:].values.astype(np.float32)
        y = data.iloc[:, 0].values

        # Verify feature dimension matches the universal extractor
        expected = LANDMARK_FEATURE_DIM   # 63
        if X.shape[1] != expected:
            raise ValueError(
                f"keypoint.csv has {X.shape[1]} features but "
                f"extract_and_normalize_landmarks returns {expected}. "
                f"Rebuild keypoint.csv with build_keypoint_csv()."
            )

        y_encoded = self.label_encoder.fit_transform(y)

        print(f"Features: {X.shape[1]}  |  Samples: {X.shape[0]}")
        print(f"Classes:  {list(self.label_encoder.classes_)}")

        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
        )
        return X_train, X_test, y_train, y_test

    def create_model(self, input_shape, num_classes):
        """Dense classifier for 63-dimensional landmark vectors."""
        model = keras.Sequential([
            layers.Dense(1470, activation='relu', input_shape=(input_shape,)),
            layers.BatchNormalization(),
            layers.Dropout(0.5),

            layers.Dense(832, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),

            layers.Dense(428, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),

            layers.Dense(264, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),

            layers.Dense(128, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),

            layers.Dense(num_classes, activation='softmax')
        ])

        model.compile(
            loss='sparse_categorical_crossentropy',
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            metrics=['accuracy']
        )
        return model

    def train_model(self, X_train, X_test, y_train, y_test,
                    epochs=100, batch_size=128):
        """Train with coordinate-augmentation generator."""
        os.makedirs('models', exist_ok=True)

        # ── Augmented generator for training data ──────────────────────────
        train_gen = AugmentedLandmarkGenerator(
            X_train, y_train,
            batch_size=batch_size,
            noise_std=0.02,     # ±2 % jitter on every coordinate
            scale_range=0.05,   # ±5 % random scale per sample
            augment=True,
            shuffle=True,
        )
        # Validation data is NOT augmented — we want clean eval metrics
        val_gen = AugmentedLandmarkGenerator(
            X_test, y_test,
            batch_size=batch_size,
            augment=False,
            shuffle=False,
        )

        callbacks = [
            EarlyStopping(monitor='val_loss', patience=10,
                          restore_best_weights=True, verbose=1),
            ModelCheckpoint('models/best_model.h5', monitor='val_accuracy',
                            save_best_only=True, verbose=1),
            ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5,
                              min_lr=1e-5, verbose=1),
        ]

        print("Starting training with coordinate augmentation...")
        self.history = self.model.fit(
            train_gen,
            epochs=epochs,
            validation_data=val_gen,
            callbacks=callbacks,
            verbose=1,
        )
        return self.history

    def evaluate_model(self, X_test, y_test):
        test_loss, test_accuracy = self.model.evaluate(X_test, y_test, verbose=0)
        y_pred         = self.model.predict(X_test, verbose=0)
        y_pred_classes = np.argmax(y_pred, axis=1)

        metrics = {
            'accuracy':  accuracy_score(y_test, y_pred_classes),
            'precision': precision_score(y_test, y_pred_classes, average='weighted'),
            'recall':    recall_score(y_test, y_pred_classes, average='weighted'),
            'f1_score':  f1_score(y_test, y_pred_classes, average='weighted'),
        }
        class_report = classification_report(
            y_test, y_pred_classes,
            target_names=self.label_encoder.classes_
        )
        return metrics, class_report, y_pred_classes

    def plot_training_history(self):
        if self.history is None:
            print("No training history available.")
            return

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

        ax1.plot(self.history.history['accuracy'],     label='Train Accuracy', linewidth=2)
        ax1.plot(self.history.history['val_accuracy'], label='Val Accuracy',   linewidth=2)
        ax1.set_title('Model Accuracy', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Epoch'); ax1.set_ylabel('Accuracy')
        ax1.legend(); ax1.grid(True, alpha=0.3)

        ax2.plot(self.history.history['loss'],     label='Train Loss', linewidth=2)
        ax2.plot(self.history.history['val_loss'], label='Val Loss',   linewidth=2)
        ax2.set_title('Model Loss', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Epoch'); ax2.set_ylabel('Loss')
        ax2.legend(); ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('models/training_history.png', dpi=300, bbox_inches='tight')
        plt.show()

    def save_model_and_metadata(self):
        self.model.save('models/sign_language_model.h5')
        np.save('models/label_encoder.npy', self.label_encoder.classes_)

        history_dict = {k: [float(v) for v in vals]
                        for k, vals in self.history.history.items()}
        with open('models/training_history.json', 'w') as f:
            json.dump(history_dict, f, indent=4)

        print("Model and metadata saved successfully!")

    def run_training(self):
        print("=== Sign Language Model Training ===")
        X_train, X_test, y_train, y_test = self.load_and_preprocess_data()

        self.model = self.create_model(X_train.shape[1],
                                       len(self.label_encoder.classes_))
        print("\nModel architecture:")
        self.model.summary()

        self.train_model(X_train, X_test, y_train, y_test)

        metrics, class_report, y_pred = self.evaluate_model(X_test, y_test)
        print("\n" + "="*50)
        print("EVALUATION RESULTS")
        print("="*50)
        for metric, value in metrics.items():
            print(f"{metric.capitalize():<12}: {value:.4f}")
        print("\nClassification Report:")
        print(class_report)

        self.plot_training_history()
        self.save_model_and_metadata()
        print("\nTraining completed successfully!")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--build-csv', metavar='IMAGE_DATASET_DIR',
                        help='Re-build keypoint.csv from raw image dataset')
    parser.add_argument('--csv', default='keypoint.csv',
                        help='Path to keypoint.csv (default: keypoint.csv)')
    args = parser.parse_args()

    if args.build_csv:
        build_keypoint_csv(args.build_csv, output_csv=args.csv)
    else:
        trainer = SignLanguageTrainer(args.csv)
        trainer.run_training()
