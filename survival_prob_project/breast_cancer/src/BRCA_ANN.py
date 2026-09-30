# BRCA Breast Cancer ANN Model Training with Cross-Validation - FIXED
# Part 2: Neural Network Model Development

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (classification_report, confusion_matrix, 
                             roc_auc_score, roc_curve, mean_absolute_error, 
                             accuracy_score, precision_score, recall_score, f1_score)
from sklearn.neural_network import MLPClassifier
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

print("="*80)
print("BRCA BREAST CANCER - ANN MODEL TRAINING")
print("Part 2: Cross-Validation and Model Building")
print("="*80)

# =============================================================================
# LOAD ENCODED DATA
# =============================================================================

# Load the encoded dataset created by visualization script
encoded_path = '/Users/daisiqi/Machine-Learning-for-Thermodynamic-Property-dataset-URS-/survival_prob_project/breast_cancer/data/brca_encoded.csv'
df = pd.read_csv(encoded_path)

print(f"\n✅ Encoded dataset loaded: {df.shape[0]} patients, {df.shape[1]} features")

# Define feature lists - FIXED: Only use numeric/encoded columns
numerical_features = ['Age', 'Protein1', 'Protein2', 'Protein3', 'Protein4']

# Check if Surgery_Year exists (from enhanced version)
if 'Surgery_Year' in df.columns:
    numerical_features.append('Surgery_Year')
    print("✅ Surgery_Year detected - including in features")

encoded_features = ['Tumour_Stage_Encoded', 'HER2_Binary', 'ER_Binary', 'PR_Binary']

# Get one-hot encoded columns (start with prefix)
histology_features = [col for col in df.columns if col.startswith('Histology_')]
surgery_features = [col for col in df.columns if col.startswith('Surgery_')]

# CRITICAL FIX: Only use numeric/encoded columns, exclude original categorical columns
all_features = numerical_features + encoded_features + histology_features + surgery_features

# Verify all features are numeric
print(f"\n🔍 Verifying feature types...")
feature_types = df[all_features].dtypes
non_numeric = feature_types[~feature_types.apply(lambda x: x.kind in 'biufc')]

if len(non_numeric) > 0:
    print(f"⚠️  WARNING: Non-numeric features detected!")
    print(non_numeric)
    print("\n❌ Removing non-numeric features...")
    all_features = [f for f in all_features if df[f].dtype.kind in 'biufc']
    print(f"✅ Fixed: Using only {len(all_features)} numeric features")
else:
    print(f"✅ All {len(all_features)} features are numeric")

print(f"\nFeatures for modeling: {len(all_features)}")
print(f"Feature list: {all_features}")

# =============================================================================
# PREPARE DATA FOR CROSS-VALIDATION
# =============================================================================

print("\n" + "="*80)
print("PREPARING DATA FOR CROSS-VALIDATION")
print("="*80)

# Extract features and target
X = df[all_features].values
y = df['Survival_Binary'].values
stages = df['Tumour_Stage'].values

print(f"\nFeature matrix: {X.shape}")
print(f"Target vector: {y.shape}")
print(f"Survival rate: {y.mean():.2%} ({y.sum()} alive, {len(y)-y.sum()} deceased)")

# Create fold assignments by tumor stage
folds = []
for stage in stages:
    if stage == 'III':
        folds.append(0)
    elif stage == 'II':
        folds.append(1)
    else:  # Stage I
        folds.append(2)

folds = np.array(folds)

print(f"\nCross-validation fold distribution:")
print(f"  Fold 0 (Test on Stage III): {np.sum(folds == 0)} patients")
print(f"  Fold 1 (Test on Stage II):  {np.sum(folds == 1)} patients")
print(f"  Fold 2 (Test on Stage I):   {np.sum(folds == 2)} patients")

# =============================================================================
# CROSS-VALIDATION TRAINING LOOP
# =============================================================================

print("\n" + "="*80)
print("TRAINING ANN MODELS - LEAVE-ONE-STAGE-OUT CROSS-VALIDATION")
print("="*80)

# Storage for results
results = {
    'fold': [],
    'test_stage': [],
    'train_size': [],
    'test_size': [],
    'train_survival_rate': [],
    'test_survival_rate': [],
    'accuracy': [],
    'precision': [],
    'recall': [],
    'f1': [],
    'auc': [],
    'mae': [],
    'loss_curves': [],
    'y_test': [],
    'y_pred': [],
    'y_proba': [],
    'confusion_matrix': []
}

fold_names = ['Fold 1 (Test: Stage III)', 'Fold 2 (Test: Stage II)', 'Fold 3 (Test: Stage I)']
test_stages = ['Stage III', 'Stage II', 'Stage I']

# Train model for each fold
for fold_idx in range(3):
    print(f"\n{'='*60}")
    print(f"TRAINING {fold_names[fold_idx]}")
    print(f"{'='*60}")
    
    # Create train/test split based on stage
    train_mask = folds != fold_idx
    test_mask = folds == fold_idx
    
    X_train, X_test = X[train_mask], X[test_mask]
    y_train, y_test = y[train_mask], y[test_mask]
    
    train_survival = y_train.mean()
    test_survival = y_test.mean()
    
    print(f"\nData split:")
    print(f"  Training:   {len(X_train)} patients (Survival: {train_survival:.1%})")
    print(f"  Test:       {len(X_test)} patients (Survival: {test_survival:.1%})")
    
    # Normalize features using StandardScaler
    print(f"\nNormalizing features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    print(f"✅ Features normalized (mean=0, std=1)")
    
    # Build ANN model architecture
    print(f"\nBuilding ANN architecture...")
    model = MLPClassifier(
        hidden_layer_sizes=(32, 16, 8),  # 3 hidden layers
        activation='relu',                # ReLU activation
        solver='adam',                    # Adam optimizer
        alpha=0.001,                      # L2 regularization
        batch_size=16,                    # Mini-batch size
        learning_rate_init=0.001,         # Initial learning rate
        max_iter=200,                     # Maximum epochs
        early_stopping=True,              # Stop if no improvement
        validation_fraction=0.15,         # 15% for validation
        n_iter_no_change=20,              # Patience for early stopping
        random_state=42,
        verbose=False
    )
    
    print(f"  Architecture: {len(all_features)} inputs → 32 → 16 → 8 → 1 output")
    print(f"  Activation: ReLU")
    print(f"  Optimizer: Adam (lr=0.001)")
    print(f"  Regularization: L2 (alpha=0.001)")
    print(f"  Early stopping: Enabled (patience=20)")
    
    # Train the model
    print(f"\n🚀 Training ANN model...")
    model.fit(X_train_scaled, y_train)
    
    print(f"✅ Training complete!")
    print(f"  Epochs trained: {model.n_iter_}")
    print(f"  Final training loss: {model.loss_:.4f}")
    
    # Make predictions
    print(f"\n📊 Evaluating on test set...")
    y_pred = model.predict(X_test_scaled)
    y_proba = model.predict_proba(X_test_scaled)[:, 1]
    
    # Calculate metrics
    acc = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    auc = roc_auc_score(y_test, y_proba)
    mae = mean_absolute_error(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    
    # Calculate sensitivity and specificity
    tn, fp, fn, tp = cm.ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    print(f"\n📈 Results:")
    print(f"  Accuracy:    {acc:.4f} ({acc*100:.2f}%)")
    print(f"  Precision:   {precision:.4f}")
    print(f"  Recall:      {recall:.4f}")
    print(f"  F1 Score:    {f1:.4f}")
    print(f"  AUC-ROC:     {auc:.4f}")
    print(f"  MAE:         {mae:.4f}")
    print(f"\n  Sensitivity (Recall): {sensitivity:.2%}")
    print(f"  Specificity:          {specificity:.2%}")
    print(f"\n  Confusion Matrix:")
    print(f"    TN={tn:3d}  FP={fp:3d}")
    print(f"    FN={fn:3d}  TP={tp:3d}")
    
    # Store results
    results['fold'].append(fold_idx + 1)
    results['test_stage'].append(test_stages[fold_idx])
    results['train_size'].append(len(X_train))
    results['test_size'].append(len(X_test))
    results['train_survival_rate'].append(train_survival)
    results['test_survival_rate'].append(test_survival)
    results['accuracy'].append(acc)
    results['precision'].append(precision)
    results['recall'].append(recall)
    results['f1'].append(f1)
    results['auc'].append(auc)
    results['mae'].append(mae)
    results['loss_curves'].append(model.loss_curve_)
    results['y_test'].append(y_test)
    results['y_pred'].append(y_pred)
    results['y_proba'].append(y_proba)
    results['confusion_matrix'].append(cm)

print("\n" + "="*80)
print("CROSS-VALIDATION TRAINING COMPLETE!")
print("="*80)

# =============================================================================
# AGGREGATE RESULTS AND SUMMARY
# =============================================================================

print("\n" + "="*80)
print("CROSS-VALIDATION SUMMARY")
print("="*80)

# Create summary DataFrame
summary_data = []
for i in range(3):
    summary_data.append({
        'Fold': f"Fold {i+1}",
        'Test Stage': results['test_stage'][i],
        'Train Size': results['train_size'][i],
        'Test Size': results['test_size'][i],
        'Accuracy': f"{results['accuracy'][i]:.4f}",
        'AUC': f"{results['auc'][i]:.4f}",
        'MAE': f"{results['mae'][i]:.4f}"
    })

summary_df = pd.DataFrame(summary_data)
print("\n📊 Per-Fold Performance:")
print(summary_df.to_string(index=False))

# Calculate average metrics
avg_acc = np.mean(results['accuracy'])
avg_precision = np.mean(results['precision'])
avg_recall = np.mean(results['recall'])
avg_f1 = np.mean(results['f1'])
avg_auc = np.mean(results['auc'])
avg_mae = np.mean(results['mae'])

std_acc = np.std(results['accuracy'])
std_auc = np.std(results['auc'])
std_mae = np.std(results['mae'])

print("\n" + "-"*80)
print("AVERAGE PERFORMANCE ACROSS ALL FOLDS:")
print("-"*80)
print(f"Mean Accuracy:  {avg_acc:.4f} ± {std_acc:.4f} ({avg_acc*100:.2f}%)")
print(f"Mean Precision: {avg_precision:.4f}")
print(f"Mean Recall:    {avg_recall:.4f}")
print(f"Mean F1 Score:  {avg_f1:.4f}")
print(f"Mean AUC-ROC:   {avg_auc:.4f} ± {std_auc:.4f}")
print(f"Mean MAE:       {avg_mae:.4f} ± {std_mae:.4f}")

# =============================================================================
# SAVE RESULTS FOR VISUALIZATION
# =============================================================================

print("\n" + "="*80)
print("SAVING RESULTS")
print("="*80)

# Save results as pickle file for visualization script
import pickle

results_path = '/Users/daisiqi/Machine-Learning-for-Thermodynamic-Property-dataset-URS-/survival_prob_project/breast_cancer/ann_results.pkl'
with open(results_path, 'wb') as f:
    pickle.dump(results, f)

print(f"\n✅ Results saved to: {results_path}")
print("   These results will be used by the visualization script.")

# =============================================================================
# FINAL MODEL ASSESSMENT
# =============================================================================

print("\n" + "="*80)
print("FINAL MODEL ASSESSMENT")
print("="*80)

# Define quality thresholds
ACCEPTABLE_ACC = 0.75
ACCEPTABLE_AUC = 0.70
ACCEPTABLE_MAE = 0.25

# Check if model meets thresholds
acc_pass = avg_acc >= ACCEPTABLE_ACC
auc_pass = avg_auc >= ACCEPTABLE_AUC
mae_pass = avg_mae <= ACCEPTABLE_MAE

print(f"\n📊 Model Performance Evaluation:")
print(f"{'-'*60}")

print(f"\n1. Accuracy Assessment:")
print(f"   Average:   {avg_acc:.2%}")
print(f"   Threshold: {ACCEPTABLE_ACC:.0%}")
print(f"   Status:    {'✅ PASS' if acc_pass else '❌ FAIL'}")

print(f"\n2. Discrimination (AUC) Assessment:")
print(f"   Average:   {avg_auc:.4f}")
print(f"   Threshold: {ACCEPTABLE_AUC:.2f}")
print(f"   Status:    {'✅ PASS' if auc_pass else '❌ FAIL'}")

print(f"\n3. Prediction Error (MAE) Assessment:")
print(f"   Average:   {avg_mae:.4f}")
print(f"   Threshold: ≤{ACCEPTABLE_MAE:.2f}")
print(f"   Status:    {'✅ PASS' if mae_pass else '❌ FAIL'}")

# Overall assessment
overall_pass = acc_pass and auc_pass and mae_pass

print("\n" + "="*80)
print("FINAL RECOMMENDATION")
print("="*80)

if overall_pass:
    print("\n✅ ✅ ✅ MODEL APPROVED FOR DEPLOYMENT ✅ ✅ ✅\n")
    print("The ANN model demonstrates satisfactory performance across")
    print("cross-validation folds and meets all quality thresholds.")
else:
    print("\n⚠️  MODEL NEEDS IMPROVEMENT ⚠️\n")
    print("The ANN model shows moderate performance but does NOT meet")
    print("all quality thresholds.")
    
    print("\nIssues Identified:")
    
    if not acc_pass:
        print(f"\n  ❌ Accuracy below threshold ({avg_acc:.1%})")
        print("     Recommendation: Collect more training data")
    else:
        print(f"\n  ✅ Accuracy acceptable ({avg_acc:.1%})")
    
    if not auc_pass:
        print(f"\n  ❌ AUC below threshold ({avg_auc:.3f})")
        print("     Major Issue: Poor discrimination ability")
        print("     Recommendations:")
        print("       • Address class imbalance with SMOTE")
        print("       • Apply class weighting")
        print("       • Try ensemble methods (Random Forest, XGBoost)")
        print("       • Add more discriminative features")
    else:
        print(f"\n  ✅ AUC acceptable ({avg_auc:.3f})")
    
    if not mae_pass:
        print(f"\n  ❌ MAE above threshold ({avg_mae:.3f})")
        print("     Recommendation: Improve probability calibration")
    else:
        print(f"\n  ✅ MAE acceptable ({avg_mae:.3f})")

print("\n" + "="*80)
print("KEY INSIGHTS")
print("="*80)

print("\n🔍 Model Performance by Tumor Stage:")
for i in range(3):
    stage = results['test_stage'][i]
    acc = results['accuracy'][i]
    auc = results['auc'][i]
    
    print(f"\n  {stage}:")
    print(f"    Accuracy: {acc:.2%}")
    print(f"    AUC:      {auc:.4f}")
    
    if auc < 0.6:
        print(f"    ⚠️  Very low discrimination - model struggles with {stage}")
    elif auc < 0.7:
        print(f"    ⚠️  Moderate discrimination - needs improvement")
    else:
        print(f"    ✅ Good discrimination ability")

print("\n🎯 Main Findings:")
print(f"  • Model achieves {avg_acc:.1%} average accuracy across folds")
print(f"  • AUC of {avg_auc:.2f} indicates {'good' if auc_pass else 'moderate'} discrimination")
print(f"  • MAE of {avg_mae:.3f} shows {'acceptable' if mae_pass else 'high'} prediction error")

if not auc_pass:
    print("\n💡 Critical Issue:")
    print("  Low AUC suggests model may be predicting majority class.")
    print("  Check confusion matrices - if TN=0, model predicts only 'Survived'")

print("\n" + "="*80)
print("ANN TRAINING COMPLETE")
print("="*80)
print("\n✅ Model training complete!")
print("✅ Results saved for visualization")
print("\n📊 Next step: Run brca_ann_visualization.py to generate graphs")
