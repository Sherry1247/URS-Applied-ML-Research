"""
Artificial Neural Network (ANN) Model for Mountaineering Survival Prediction
Analyzing how technology, time period, and socioeconomic factors affect survival probability
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, roc_curve
from sklearn.inspection import permutation_importance
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks
import os
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Set style
sns.set_style("whitegrid")
plt.rcParams['font.size'] = 10

# Define paths
DATA_PATH = '/Users/daisiqi/Machine-Learning-for-Thermodynamic-Property-dataset-URS-/survival_prob_project/Himalayan/data/exped.csv'
OUTPUT_DIR = '/Users/daisiqi/Machine-Learning-for-Thermodynamic-Property-dataset-URS-/survival_prob_project/Himalayan/visualization'
MODEL_DIR = '/Users/daisiqi/Machine-Learning-for-Thermodynamic-Property-dataset-URS-/survival_prob_project/Himalayan/models'

# Create output directories if they don't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

print("="*60)
print("MOUNTAINEERING SURVIVAL PREDICTION - ANN MODEL")
print("="*60)

# Load data
print("\nLoading expedition data...")
exped = pd.read_csv(DATA_PATH, low_memory=False)
print(f"Loaded {len(exped)} expeditions")

# Extract and clean features
print("\nPreparing dataset...")
exped['year'] = pd.to_numeric(exped['year'], errors='coerce')

# Handle season - may be full names or abbreviations
exped['season'] = exped['season'].fillna('Unknown')
season_map = {'Spr': 'Spring', 'Sum': 'Summer', 'Aut': 'Autumn', 'Win': 'Winter'}
exped['season'] = exped['season'].replace(season_map)

# Time period encoding
exped['time_period_code'] = pd.cut(exped['year'], 
                                    bins=[1900, 1960, 1980, 2000, 2025],
                                    labels=[0, 1, 2, 3]).astype(float)

# Decade
exped['decade'] = (exped['year'] // 10) * 10

# Season encoding
season_code_map = {'Spring': 0, 'Summer': 1, 'Autumn': 2, 'Winter': 3, 'Unknown': 4}
exped['season_code'] = exped['season'].map(season_code_map)

# Oxygen usage - o2used column
exped['oxygen_used'] = (exped['o2used'] == 'Y').astype(int)

# Calculate total participants and deaths
exped['mdeaths'] = pd.to_numeric(exped['mdeaths'], errors='coerce').fillna(0)
exped['hdeaths'] = pd.to_numeric(exped['hdeaths'], errors='coerce').fillna(0)
exped['totmembers'] = pd.to_numeric(exped['totmembers'], errors='coerce').fillna(0)
exped['tothired'] = pd.to_numeric(exped['tothired'], errors='coerce').fillna(0)

exped['total_participants'] = exped['totmembers'] + exped['tothired']
exped['total_deaths'] = exped['mdeaths'] + exped['hdeaths']

# Size category
exped['size_category'] = pd.cut(exped['total_participants'], 
                                bins=[0, 5, 10, 20, 50, 1000],
                                labels=[0, 1, 2, 3, 4]).astype(float)

# Remove rows with missing critical data
exped = exped.dropna(subset=['year', 'time_period_code', 'season_code', 'size_category'])
exped = exped[exped['total_participants'] > 0]

print(f"After cleaning: {len(exped)} expeditions")

# Create individual participant records
print("Creating individual participant records...")
all_records = []

for idx, row in exped.iterrows():
    # Members
    for _ in range(int(row['totmembers'])):
        all_records.append({
            'year': row['year'],
            'time_period_code': row['time_period_code'],
            'decade': row['decade'],
            'season_code': row['season_code'],
            'oxygen_used': row['oxygen_used'],
            'total_participants': row['total_participants'],
            'size_category': row['size_category'],
            'is_hired': 0,
            'survived': 1  # Default to survived
        })
    
    # Hired staff
    for _ in range(int(row['tothired'])):
        all_records.append({
            'year': row['year'],
            'time_period_code': row['time_period_code'],
            'decade': row['decade'],
            'season_code': row['season_code'],
            'oxygen_used': row['oxygen_used'],
            'total_participants': row['total_participants'],
            'size_category': row['size_category'],
            'is_hired': 1,
            'survived': 1  # Default to survived
        })

df = pd.DataFrame(all_records)

# Add deaths - simplified approach
death_idx = 0
for idx, row in exped.iterrows():
    # Member deaths
    member_records = df[(df['year'] == row['year']) & (df['is_hired'] == 0)]
    if len(member_records) > 0 and row['mdeaths'] > 0:
        deaths_to_mark = min(int(row['mdeaths']), len(member_records))
        df.loc[member_records.index[:deaths_to_mark], 'survived'] = 0
    
    # Hired staff deaths
    hired_records = df[(df['year'] == row['year']) & (df['is_hired'] == 1)]
    if len(hired_records) > 0 and row['hdeaths'] > 0:
        deaths_to_mark = min(int(row['hdeaths']), len(hired_records))
        df.loc[hired_records.index[:deaths_to_mark], 'survived'] = 0

print(f"Total participant records: {len(df):,}")
print(f"Survivors: {df['survived'].sum():,} ({(df['survived'].sum()/len(df)*100):.2f}%)")
print(f"Deaths: {(df['survived']==0).sum():,} ({((df['survived']==0).sum()/len(df)*100):.2f}%)")

# Select features
feature_cols = ['year', 'time_period_code', 'decade', 'season_code', 
                'oxygen_used', 'total_participants', 'size_category', 'is_hired']
X = df[feature_cols].values
y = df['survived'].values

# Sample data for efficiency (using stratified sampling)
print("\nSampling data for training...")
sample_size = min(50000, len(df))
X_sample, _, y_sample, _ = train_test_split(X, y, train_size=sample_size, 
                                             stratify=y, random_state=42)
print(f"Using {len(X_sample):,} samples for training")

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_sample, y_sample, 
                                                      test_size=0.2, random_state=42, 
                                                      stratify=y_sample)

print(f"Training set: {len(X_train):,} samples")
print(f"Test set: {len(X_test):,} samples")

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Build ANN model
print("\nBuilding ANN model...")
model = keras.Sequential([
    layers.Dense(64, activation='relu', input_shape=(X_train_scaled.shape[1],)),
    layers.BatchNormalization(),
    layers.Dropout(0.3),
    
    layers.Dense(32, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.3),
    
    layers.Dense(16, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.2),
    
    layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001),
              loss='binary_crossentropy',
              metrics=['accuracy', 'mae'])

print("\nModel Architecture:")
model.summary()

# Define callbacks
early_stopping = callbacks.EarlyStopping(monitor='val_loss', patience=15, 
                                         restore_best_weights=True, verbose=1)
reduce_lr = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, 
                                        min_lr=0.00001, verbose=1)

# Train model
print("\nTraining model...")
history = model.fit(X_train_scaled, y_train,
                   epochs=100,
                   batch_size=256,
                   validation_split=0.2,
                   callbacks=[early_stopping, reduce_lr],
                   verbose=1)

# Save model
model_path = os.path.join(MODEL_DIR, 'survival_ann_model.h5')
model.save(model_path)
print(f"\nModel saved to: {model_path}")

# Make predictions
print("\nEvaluating model...")
y_pred_proba = model.predict(X_test_scaled).flatten()
y_pred = (y_pred_proba > 0.5).astype(int)

# Calculate metrics
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=0)
recall = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)

# Try to calculate ROC AUC (may fail if only one class predicted)
try:
    roc_auc = roc_auc_score(y_test, y_pred_proba)
except:
    roc_auc = 0.5

print("\n" + "="*60)
print("MODEL PERFORMANCE METRICS")
print("="*60)
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1:.4f}")
print(f"ROC AUC: {roc_auc:.4f}")

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(cm)
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Died', 'Survived']))

# VISUALIZATIONS
print("\nGenerating visualizations...")

# 1. Training History (Loss and MAE)
print("1. Creating training history visualization...")
fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# Loss
axes[0, 0].plot(history.history['loss'], label='Training Loss', linewidth=2)
axes[0, 0].plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
axes[0, 0].set_xlabel('Epoch', fontweight='bold')
axes[0, 0].set_ylabel('Loss', fontweight='bold')
axes[0, 0].set_title('Training and Validation Loss', fontweight='bold', fontsize=12)
axes[0, 0].legend()
axes[0, 0].grid(alpha=0.3)

# MAE
axes[0, 1].plot(history.history['mae'], label='Training MAE', linewidth=2)
axes[0, 1].plot(history.history['val_mae'], label='Validation MAE', linewidth=2)
axes[0, 1].set_xlabel('Epoch', fontweight='bold')
axes[0, 1].set_ylabel('MAE', fontweight='bold')
axes[0, 1].set_title('Training and Validation MAE', fontweight='bold', fontsize=12)
axes[0, 1].legend()
axes[0, 1].grid(alpha=0.3)

# Overfitting Analysis
axes[0, 2].plot(np.array(history.history['loss']) - np.array(history.history['val_loss']), 
                linewidth=2, color='red')
axes[0, 2].set_xlabel('Epoch', fontweight='bold')
axes[0, 2].set_ylabel('Loss Difference', fontweight='bold')
axes[0, 2].set_title('Overfitting Analysis\n(Training Loss - Validation Loss)', 
                     fontweight='bold', fontsize=12)
axes[0, 2].axhline(y=0, color='black', linestyle='--', alpha=0.3)
axes[0, 2].grid(alpha=0.3)

# ROC Curve
try:
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    axes[1, 0].plot(fpr, tpr, linewidth=2, label=f'AUC = {roc_auc:.4f}')
    axes[1, 0].plot([0, 1], [0, 1], 'k--', alpha=0.3, label='Random')
    axes[1, 0].set_xlabel('False Positive Rate', fontweight='bold')
    axes[1, 0].set_ylabel('True Positive Rate', fontweight='bold')
    axes[1, 0].set_title('ROC Curve', fontweight='bold', fontsize=12)
    axes[1, 0].legend()
    axes[1, 0].grid(alpha=0.3)
except:
    axes[1, 0].text(0.5, 0.5, 'ROC Curve\nNot Available', ha='center', va='center')

# Confusion Matrix
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[1, 1], cbar=False,
            xticklabels=['Died', 'Survived'], yticklabels=['Died', 'Survived'])
axes[1, 1].set_xlabel('Predicted', fontweight='bold')
axes[1, 1].set_ylabel('Actual', fontweight='bold')
axes[1, 1].set_title('Confusion Matrix', fontweight='bold', fontsize=12)

# Performance Metrics Bar Chart
metrics_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
metrics_values = [accuracy, precision, recall, f1]
colors = ['#2E8B57', '#4169E1', '#FF6347', '#FFD700']
bars = axes[1, 2].bar(range(len(metrics_names)), metrics_values, color=colors, 
                      edgecolor='black', linewidth=1.5)
axes[1, 2].set_xticks(range(len(metrics_names)))
axes[1, 2].set_xticklabels(metrics_names, fontweight='bold', rotation=45, ha='right')
axes[1, 2].set_ylabel('Score', fontweight='bold')
axes[1, 2].set_title('Performance Metrics', fontweight='bold', fontsize=12)
axes[1, 2].set_ylim([0, 1.1])
axes[1, 2].grid(axis='y', alpha=0.3)
for bar, val in zip(bars, metrics_values):
    axes[1, 2].text(bar.get_x() + bar.get_width()/2., val + 0.02,
                    f'{val:.4f}', ha='center', fontweight='bold', fontsize=9)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'ann_model_results.png'), dpi=300, bbox_inches='tight')
plt.close()
print(f"   Saved: ann_model_results.png")

# 2. Feature Importance (Permutation Importance)
print("2. Calculating feature importance...")
sample_indices = np.random.choice(len(X_test_scaled), size=min(1000, len(X_test_scaled)), replace=False)
X_test_sample = X_test_scaled[sample_indices]
y_test_sample = y_test[sample_indices]

perm_importance = permutation_importance(model, X_test_sample, y_test_sample, 
                                        n_repeats=10, random_state=42, n_jobs=-1)

fig, ax = plt.subplots(figsize=(12, 8))
feature_names = ['Year', 'Time Period', 'Decade', 'Season', 
                 'Oxygen Used', 'Total Participants', 'Size Category', 'Is Hired']
importance_means = perm_importance.importances_mean
importance_std = perm_importance.importances_std

# Sort by importance
sorted_idx = importance_means.argsort()
sorted_features = [feature_names[i] for i in sorted_idx]
sorted_importance = importance_means[sorted_idx]
sorted_std = importance_std[sorted_idx]

colors_feat = plt.cm.viridis(np.linspace(0.2, 0.9, len(sorted_features)))
bars = ax.barh(range(len(sorted_features)), sorted_importance, 
               xerr=sorted_std, color=colors_feat, edgecolor='black', linewidth=1.2)
ax.set_yticks(range(len(sorted_features)))
ax.set_yticklabels(sorted_features, fontweight='bold')
ax.set_xlabel('Permutation Importance', fontweight='bold')
ax.set_title('Feature Importance Analysis\n(Permutation Importance)', 
             fontweight='bold', fontsize=13, pad=15)
ax.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'feature_importance.png'), dpi=300, bbox_inches='tight')
plt.close()
print(f"   Saved: feature_importance.png")

print("\n" + "="*60)
print("MODEL TRAINING COMPLETE")
print("="*60)
print(f"Model saved to: {model_path}")
print(f"Visualizations saved to: {OUTPUT_DIR}")
print("="*60)
