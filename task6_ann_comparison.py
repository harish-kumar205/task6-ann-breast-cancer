"""Task 6: Keras ANN and ML baselines on the breast cancer dataset."""
import random
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             f1_score, precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.keras.utils.set_random_seed(SEED)
OUT = Path('.')

# Load data: target 0 = malignant, 1 = benign.
data = load_breast_cancer()
X, y = data.data, data.target
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=SEED, stratify=y)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ANN: 16 and 8 ReLU hidden units; sigmoid predicts P(benign).
ann = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(X_train.shape[1],)),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(8, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid'),
])
ann.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
t0 = time.perf_counter()
history = ann.fit(X_train, y_train, epochs=50, batch_size=16,
                  validation_split=0.20, verbose=1)
ann_time = time.perf_counter() - t0

# Plot training/validation accuracy and loss.
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
for key, label in [('accuracy', 'Training'), ('val_accuracy', 'Validation')]:
    ax[0].plot(history.history[key], label=label)
for key, label in [('loss', 'Training'), ('val_loss', 'Validation')]:
    ax[1].plot(history.history[key], label=label)
ax[0].set(title='ANN accuracy', xlabel='Epoch', ylabel='Accuracy')
ax[1].set(title='ANN loss', xlabel='Epoch', ylabel='Binary cross-entropy')
for a in ax: a.legend(); a.grid(alpha=.25)
fig.tight_layout(); fig.savefig(OUT / 'task6_training_curves.png', dpi=160); plt.show()

# Compare models on the same held-out test set. Positive class for metrics = malignant (0).
models = {
    'Logistic Regression (Task 4)': LogisticRegression(max_iter=1000, random_state=SEED),
    'Random Forest (Task 5)': RandomForestClassifier(n_estimators=300, random_state=SEED, n_jobs=-1),
}
rows, predictions = [], {}
for name, model in models.items():
    t0 = time.perf_counter(); model.fit(X_train, y_train); fit_time = time.perf_counter() - t0
    pred = model.predict(X_test); p_malignant = model.predict_proba(X_test)[:, 0]
    predictions[name] = (pred, p_malignant, fit_time)
ann_prob = 1 - ann.predict(X_test, verbose=0).ravel()
ann_pred = (ann_prob < 0.5).astype(int)
predictions['Neural Network (Task 6)'] = (ann_pred, ann_prob, ann_time)
for name, (pred, prob, fit_time) in predictions.items():
    rows.append({'Model': name, 'Accuracy': accuracy_score(y_test, pred),
        'Malignant precision': precision_score(y_test, pred, pos_label=0, zero_division=0),
        'Malignant recall': recall_score(y_test, pred, pos_label=0, zero_division=0),
        'Malignant F1': f1_score(y_test, pred, pos_label=0, zero_division=0),
        'Malignant ROC-AUC': roc_auc_score(y_test == 0, prob), 'Fit time (s)': fit_time})
comparison = pd.DataFrame(rows).set_index('Model')
print(comparison.round(4))
comparison.to_csv(OUT / 'task6_model_comparison.csv')
pd.DataFrame(classification_report(y_test, ann_pred, labels=[0, 1],
    target_names=['malignant', 'benign'], output_dict=True, zero_division=0)).T.to_csv(
    OUT / 'task6_ann_classification_report.csv')

# Confusion matrices make malignant false negatives visible.
fig, ax = plt.subplots(1, 3, figsize=(12, 3.8))
for a, (name, (pred, _, _)) in zip(ax, predictions.items()):
    cm = confusion_matrix(y_test, pred, labels=[0, 1]); a.imshow(cm, cmap='Blues')
    a.set(title=name, xlabel='Predicted', ylabel='Actual',
          xticks=[0, 1], yticks=[0, 1], xticklabels=['Malignant', 'Benign'],
          yticklabels=['Malignant', 'Benign'])
    for (i, j), value in np.ndenumerate(cm): a.text(j, i, str(value), ha='center', va='center')
fig.tight_layout(); fig.savefig(OUT / 'task6_confusion_matrices.png', dpi=160); plt.show()
print('ANN classification report (test set):')
print(classification_report(y_test, ann_pred, target_names=['malignant', 'benign'], zero_division=0))
