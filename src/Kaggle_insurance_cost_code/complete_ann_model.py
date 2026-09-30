# ============================================================================
# COMPLETE ANN MODEL FOR MEDICAL INSURANCE COST PREDICTION
# WITH VISUALIZATION, PREDICTION FUNCTION, AND ACCURACY TESTING
# ============================================================================

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import seaborn as sns

# ============================================================================
# STEP 1: LOAD AND PREPARE DATA
# ============================================================================
print("="*80)
print("STEP 1: LOADING AND PREPARING DATA")
print("="*80)

df = pd.read_csv('/Users/daisiqi/Machine-Learning-for-Thermodynamic-Property-dataset-URS-/data/Kaggle_medical_insurance_dataset/insurance.csv')

# Encode categorical variables
df_encoded = pd.get_dummies(df, columns=['sex', 'smoker', 'region'], drop_first=True)

print(f"Dataset shape: {df_encoded.shape}")
print(f"Columns: {df_encoded.columns.tolist()}")

# Separate features and target
X = df_encoded.drop('charges', axis=1)
y = df_encoded['charges']

# ============================================================================
# STEP 2: SPLIT DATA INTO TRAINING AND TESTING SETS
# ============================================================================
print("\n" + "="*80)
print("STEP 2: SPLITTING DATA")
print("="*80)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"Training set size: {X_train.shape}")
print(f"Test set size: {X_test.shape}")

# ============================================================================
# STEP 3: NORMALIZE DATA (CRITICAL FOR NEURAL NETWORKS!)
# ============================================================================
print("\n" + "="*80)
print("STEP 3: NORMALIZING DATA")
print("="*80)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("✓ Data normalized successfully")
print(f"Mean of normalized training data: {X_train_scaled.mean():.4f}")
print(f"Std of normalized training data: {X_train_scaled.std():.4f}")

# ============================================================================
# STEP 4: BUILD THE NEURAL NETWORK MODEL
# ============================================================================
print("\n" + "="*80)
print("STEP 4: BUILDING NEURAL NETWORK")
print("="*80)

np.random.seed(42)
tf.random.set_seed(42)

n_features = int(X_train_scaled.shape[1])

# Create the model
model = keras.Sequential([
    # Hidden layer 1: 64 neurons with ReLU activation
    layers.Dense(64, activation='relu', input_shape=(n_features,), 
                 kernel_initializer='he_normal', name='hidden_layer_1'),
    
    # Hidden layer 2: 32 neurons with ReLU activation
    layers.Dense(32, activation='relu', 
                 kernel_initializer='he_normal', name='hidden_layer_2'),
    
    # Hidden layer 3: 16 neurons with ReLU activation
    layers.Dense(16, activation='relu', 
                 kernel_initializer='he_normal', name='hidden_layer_3'),
    
    # Output layer: 1 neuron with linear activation (for regression)
    layers.Dense(1, activation='linear', name='output_layer')
])

# ============================================================================
# STEP 5: COMPILE THE MODEL
# ============================================================================
print("\n" + "="*80)
print("STEP 5: COMPILING MODEL")
print("="*80)

model.compile(
    optimizer='adam',           # Adaptive learning rate optimizer
    loss='mse',                 # Mean Squared Error for regression
    metrics=['mae', 'mse']      # Track Mean Absolute Error and MSE
)

print("✓ Model compiled successfully")
model.summary()

# ============================================================================
# STEP 6: TRAIN THE MODEL
# ============================================================================
print("\n" + "="*80)
print("STEP 6: TRAINING NEURAL NETWORK")
print("="*80)

history = model.fit(
    X_train_scaled, y_train,
    epochs=100,                 # Number of complete passes through data
    batch_size=32,              # Process 32 samples at a time
    validation_split=0.2,       # Use 20% of training data for validation
    verbose=1                   # Show progress
)

# ============================================================================
# STEP 7: EVALUATE THE MODEL
# ============================================================================
print("\n" + "="*80)
print("STEP 7: EVALUATING MODEL PERFORMANCE")
print("="*80)

# Make predictions
y_pred_train = model.predict(X_train_scaled).flatten()
y_pred_test = model.predict(X_test_scaled).flatten()

# Calculate metrics
train_r2 = r2_score(y_train, y_pred_train)
train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
train_mae = mean_absolute_error(y_train, y_pred_train)

test_r2 = r2_score(y_test, y_pred_test)
test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
test_mae = mean_absolute_error(y_test, y_pred_test)

print(f"\nTRAINING SET PERFORMANCE:")
print(f"  R² Score: {train_r2:.4f}")
print(f"  RMSE: ${train_rmse:.2f}")
print(f"  MAE: ${train_mae:.2f}")

print(f"\nTEST SET PERFORMANCE:")
print(f"  R² Score: {test_r2:.4f}")
print(f"  RMSE: ${test_rmse:.2f}")
print(f"  MAE: ${test_mae:.2f}")

# ============================================================================
# STEP 8: CREATE PREDICTION COMPARISON TABLE
# ============================================================================
print("\n" + "="*80)
print("STEP 8: CREATING PREDICTION COMPARISON TABLE")
print("="*80)

# Create comparison dataframe
comparison_df = pd.DataFrame({
    'Actual_Cost': y_test.values,
    'Predicted_Cost': y_pred_test,
    'Absolute_Error': np.abs(y_test.values - y_pred_test),
    'Percentage_Error': np.abs((y_test.values - y_pred_test) / y_test.values) * 100
})

# Sort by actual cost for better visualization
comparison_df = comparison_df.sort_values('Actual_Cost').reset_index(drop=True)

# Save to CSV
comparison_df.to_csv('insurance_predictions_comparison.csv', index=False)
print("✓ Prediction comparison table saved to 'insurance_predictions_comparison.csv'")

# Display first 20 rows
print("\nFirst 20 predictions:")
print(comparison_df.head(20).to_string(index=False))

# Summary statistics
print("\n" + "-"*80)
print("PREDICTION ERROR STATISTICS:")
print("-"*80)
print(f"Mean Absolute Error: ${comparison_df['Absolute_Error'].mean():.2f}")
print(f"Mean Percentage Error: {comparison_df['Percentage_Error'].mean():.2f}%")
print(f"Median Percentage Error: {comparison_df['Percentage_Error'].median():.2f}%")
print(f"Max Percentage Error: {comparison_df['Percentage_Error'].max():.2f}%")
print(f"Min Percentage Error: {comparison_df['Percentage_Error'].min():.2f}%")

# ============================================================================
# STEP 9: VISUALIZATION 1 - TRAINING HISTORY
# ============================================================================
print("\n" + "="*80)
print("STEP 9: CREATING VISUALIZATIONS")
print("="*80)

def plot_training_history(history):
    """Plot training and validation loss over epochs"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Loss plot
    ax1.plot(history.history['loss'], label='Training Loss', linewidth=2)
    ax1.plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Mean Squared Error', fontsize=12)
    ax1.set_title('Model Loss During Training', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # MAE plot
    ax2.plot(history.history['mae'], label='Training MAE', linewidth=2)
    ax2.plot(history.history['val_mae'], label='Validation MAE', linewidth=2)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Mean Absolute Error ($)', fontsize=12)
    ax2.set_title('Model MAE During Training', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
    print("✓ Training history plot saved as 'training_history.png'")
    plt.show()

plot_training_history(history)

# ============================================================================
# STEP 10: VISUALIZATION 2 - ACTUAL VS PREDICTED
# ============================================================================

def plot_predictions(y_test, y_pred_test):
    """Plot actual vs predicted values"""
    plt.figure(figsize=(10, 8))
    plt.scatter(y_test, y_pred_test, alpha=0.6, s=50, edgecolors='k', linewidth=0.5)
    
    # Perfect prediction line
    min_val = min(y_test.min(), y_pred_test.min())
    max_val = max(y_test.max(), y_pred_test.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
    
    plt.xlabel('Actual Insurance Cost ($)', fontsize=12)
    plt.ylabel('Predicted Insurance Cost ($)', fontsize=12)
    plt.title('Actual vs. Predicted Insurance Costs (Test Set)', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    
    # Add R² to plot
    r2 = r2_score(y_test, y_pred_test)
    plt.text(0.05, 0.95, f'R² = {r2:.4f}', transform=plt.gca().transAxes,
             fontsize=12, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig('actual_vs_predicted.png', dpi=300, bbox_inches='tight')
    print("✓ Actual vs Predicted plot saved as 'actual_vs_predicted.png'")
    plt.show()

plot_predictions(y_test, y_pred_test)

# ============================================================================
# STEP 11: VISUALIZATION 3 - ERROR DISTRIBUTION
# ============================================================================

def plot_error_distribution(comparison_df):
    """Plot distribution of prediction errors"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Absolute error distribution
    ax1.hist(comparison_df['Absolute_Error'], bins=30, edgecolor='black', alpha=0.7)
    ax1.set_xlabel('Absolute Error ($)', fontsize=12)
    ax1.set_ylabel('Frequency', fontsize=12)
    ax1.set_title('Distribution of Absolute Errors', fontsize=14, fontweight='bold')
    ax1.axvline(comparison_df['Absolute_Error'].mean(), color='red', 
                linestyle='--', linewidth=2, label=f'Mean: ${comparison_df["Absolute_Error"].mean():.2f}')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Percentage error distribution
    ax2.hist(comparison_df['Percentage_Error'], bins=30, edgecolor='black', alpha=0.7, color='orange')
    ax2.set_xlabel('Percentage Error (%)', fontsize=12)
    ax2.set_ylabel('Frequency', fontsize=12)
    ax2.set_title('Distribution of Percentage Errors', fontsize=14, fontweight='bold')
    ax2.axvline(comparison_df['Percentage_Error'].mean(), color='red', 
                linestyle='--', linewidth=2, label=f'Mean: {comparison_df["Percentage_Error"].mean():.2f}%')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('error_distribution.png', dpi=300, bbox_inches='tight')
    print("✓ Error distribution plot saved as 'error_distribution.png'")
    plt.show()

plot_error_distribution(comparison_df)

# ============================================================================
# STEP 12: VISUALIZATION 4 - NEURAL NETWORK ARCHITECTURE
# ============================================================================

def visualize_ann_architecture():
    """Visualize the neural network architecture"""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Define layer positions
    layer_sizes = [8, 64, 32, 16, 1]  # Input, Hidden1, Hidden2, Hidden3, Output
    layer_names = ['Input\n(8 features)', 'Hidden 1\n(64 neurons)', 
                   'Hidden 2\n(32 neurons)', 'Hidden 3\n(16 neurons)', 
                   'Output\n(1 prediction)']
    
    # Calculate positions
    v_spacing = 1.0 / max(layer_sizes)
    h_spacing = 1.0 / (len(layer_sizes) - 1)
    
    # Draw nodes
    for layer_idx, (size, name) in enumerate(zip(layer_sizes, layer_names)):
        layer_top = v_spacing * (max(layer_sizes) - size) / 2
        
        # For large layers, only show representative nodes
        nodes_to_draw = min(size, 10)
        
        for node_idx in range(nodes_to_draw):
            x = layer_idx * h_spacing
            y = layer_top + node_idx * v_spacing
            
            # Choose color based on layer
            if layer_idx == 0:
                color = 'lightblue'
            elif layer_idx == len(layer_sizes) - 1:
                color = 'lightcoral'
            else:
                color = 'lightgreen'
            
            circle = plt.Circle((x, y), v_spacing/4, color=color, ec='black', zorder=4)
            ax.add_patch(circle)
        
        # Add layer label
        ax.text(layer_idx * h_spacing, -0.15, name, 
                ha='center', va='top', fontsize=11, fontweight='bold')
    
    # Draw connections (only between first few nodes for clarity)
    for layer_idx in range(len(layer_sizes) - 1):
        for i in range(min(5, layer_sizes[layer_idx])):
            for j in range(min(5, layer_sizes[layer_idx + 1])):
                x1 = layer_idx * h_spacing
                y1 = v_spacing * (max(layer_sizes) - layer_sizes[layer_idx]) / 2 + i * v_spacing
                
                x2 = (layer_idx + 1) * h_spacing
                y2 = v_spacing * (max(layer_sizes) - layer_sizes[layer_idx + 1]) / 2 + j * v_spacing
                
                ax.plot([x1, x2], [y1, y2], 'gray', alpha=0.3, linewidth=0.5, zorder=1)
    
    ax.set_xlim(-0.1, 1.1)
    ax.set_ylim(-0.2, 1.1)
    ax.axis('off')
    ax.set_title('Artificial Neural Network Architecture\nMedical Insurance Cost Prediction', 
                 fontsize=16, fontweight='bold', pad=20)
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='lightblue', edgecolor='black', label='Input Layer'),
                      Patch(facecolor='lightgreen', edgecolor='black', label='Hidden Layers (ReLU)'),
                      Patch(facecolor='lightcoral', edgecolor='black', label='Output Layer (Linear)')]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=10)
    
    plt.tight_layout()
    plt.savefig('ann_architecture.png', dpi=300, bbox_inches='tight')
    print("✓ ANN architecture visualization saved as 'ann_architecture.png'")
    plt.show()

visualize_ann_architecture()

# ============================================================================
# STEP 13: SAVE THE MODEL
# ============================================================================
print("\n" + "="*80)
print("STEP 13: SAVING MODEL")
print("="*80)

model.save('insurance_ann_model.h5')
print("✓ Model saved as 'insurance_ann_model.h5'")

# ============================================================================
# STEP 14: CREATE PREDICTION FUNCTION
# ============================================================================
print("\n" + "="*80)
print("STEP 14: DEFINING PREDICTION FUNCTION")
print("="*80)

def predict_insurance_cost(age, bmi, children, sex, smoker, region):
    """
    Predict insurance cost for a given individual.
    
    Parameters:
    -----------
    age : int
        Age of the individual
    bmi : float
        Body Mass Index
    children : int
        Number of children/dependents
    sex : str
        'male' or 'female'
    smoker : str
        'yes' or 'no'
    region : str
        'northwest', 'northeast', 'southeast', or 'southwest'
    
    Returns:
    --------
    float : Predicted insurance cost in dollars
    """
    # Create input dataframe
    input_data = pd.DataFrame({
        'age': [age],
        'bmi': [bmi],
        'children': [children],
        'sex_male': [1 if sex.lower() == 'male' else 0],
        'smoker_yes': [1 if smoker.lower() == 'yes' else 0],
        'region_northwest': [1 if region.lower() == 'northwest' else 0],
        'region_southeast': [1 if region.lower() == 'southeast' else 0],
        'region_southwest': [1 if region.lower() == 'southwest' else 0]
    })
    
    # Normalize
    input_scaled = scaler.transform(input_data)
    
    # Predict
    prediction = model.predict(input_scaled, verbose=0)
    
    return prediction

print("✓ Prediction function defined successfully")
print("\nFunction signature: predict_insurance_cost(age, bmi, children, sex, smoker, region)")

# ============================================================================
# STEP 15: TEST THE PREDICTION FUNCTION
# ============================================================================
print("\n" + "="*80)
print("STEP 15: TESTING PREDICTION FUNCTION")
print("="*80)

# Example predictions
test_cases = [
    {"age": 30, "bmi": 25.0, "children": 2, "sex": "male", "smoker": "no", "region": "southwest"},
    {"age": 45, "bmi": 32.5, "children": 1, "sex": "female", "smoker": "yes", "region": "northeast"},
    {"age": 55, "bmi": 28.0, "children": 0, "sex": "male", "smoker": "no", "region": "southeast"},
]

print("\nExample Predictions:")
print("-" * 80)
for i, case in enumerate(test_cases, 1):
    cost = predict_insurance_cost(**case)
    print(f"\nCase {i}:")
    print(f"  Input: {case}")
    print(f"  Predicted Insurance Cost: ${float(cost):.2f}")

# ============================================================================
# FINAL SUMMARY
# ============================================================================
print("\n" + "="*80)
print("ANALYSIS COMPLETE!")
print("="*80)
print("\nGenerated Files:")
print("  1. insurance_predictions_comparison.csv - Detailed prediction comparison")
print("  2. training_history.png - Training progress visualization")
print("  3. actual_vs_predicted.png - Prediction accuracy scatter plot")
print("  4. error_distribution.png - Error distribution histograms")
print("  5. ann_architecture.png - Neural network structure diagram")
print("  6. insurance_ann_model.h5 - Trained model (can be loaded later)")
print("\nModel Performance Summary:")
print(f"  Test R² Score: {test_r2:.4f} ({test_r2*100:.2f}% variance explained)")
print(f"  Test RMSE: ${test_rmse:.2f}")
print(f"  Test MAE: ${test_mae:.2f}")
print(f"  Mean Percentage Error: {comparison_df['Percentage_Error'].mean():.2f}%")
print("\n" + "="*80)
