import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LinearRegression
import warnings
warnings.filterwarnings('ignore')

# Load and encode data, basic (NO variable interaction)
df = pd.read_csv('/Users/daisiqi/Machine-Learning-for-Thermodynamic-Property-dataset-URS-/data/Kaggle_Tip_dataset/tip.csv')
X = df.drop('tip', axis=1)
y = df['tip'].values
X_encoded = pd.get_dummies(X, columns=['sex', 'smoker', 'day', 'time'], drop_first=False)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X_encoded.values, y, test_size=0.2, random_state=42)

# Scale
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ANN with history tracking
class TrackerANN:
    def __init__(self, hidden_layers=(32, 16)):
        self.model = MLPRegressor(hidden_layer_sizes=hidden_layers,
                                  activation='relu',
                                  solver='adam',
                                  learning_rate_init=0.001,
                                  alpha=0.01,
                                  max_iter=1,
                                  random_state=42,
                                  warm_start=True,
                                  verbose=False)
        self.train_losses = []
        self.val_losses = []
        self.train_mae = []
        self.val_mae = []
        self.epochs = []
    def fit_with_history(self, X_train, y_train, X_val, y_val, max_epochs=110, patience=20):
        best_val_loss = float('inf')
        patience_counter = 0
        for epoch in range(max_epochs):
            self.model.fit(X_train, y_train)
            train_pred = self.model.predict(X_train)
            val_pred = self.model.predict(X_val)
            train_loss = mean_squared_error(y_train, train_pred)
            val_loss = mean_squared_error(y_val, val_pred)
            train_mae = mean_absolute_error(y_train, train_pred)
            val_mae = mean_absolute_error(y_val, val_pred)
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_mae.append(train_mae)
            self.val_mae.append(val_mae)
            self.epochs.append(epoch)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    break
        return self.model

X_train_part, X_val, y_train_part, y_val = train_test_split(X_train_scaled, y_train, test_size=0.2, random_state=42)
tracker = TrackerANN((32, 16))
model_ann = tracker.fit_with_history(X_train_part, y_train_part, X_val, y_val)

# Baseline Linear Regression
lr_model = LinearRegression()
lr_model.fit(X_train_scaled, y_train)
y_test_pred_lr = lr_model.predict(X_test_scaled)

# ANN predictions for test set
y_test_pred_ann = model_ann.predict(X_test_scaled)

# --- Graph 1: Model Loss & MAE During Training ---
fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
ax1.plot(tracker.epochs, tracker.train_losses, 'b-', label='Training Loss')
ax1.plot(tracker.epochs, tracker.val_losses, 'orange', label='Validation Loss')
ax1.set_xlabel('Epoch', fontsize=12)
ax1.set_ylabel('Mean Squared Error')
ax1.set_title('Model Loss During Training')
ax1.legend()
ax1.grid(True, alpha=0.3)
ax2.plot(tracker.epochs, tracker.train_mae, 'b-', label='Training MAE')
ax2.plot(tracker.epochs, tracker.val_mae, 'orange', label='Validation MAE')
ax2.set_xlabel('Epoch', fontsize=12)
ax2.set_ylabel('Mean Absolute Error ($)')
ax2.set_title('Model MAE During Training')
ax2.legend()
ax2.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('prev_ann_graph1_training.png', dpi=200)
plt.show()

# --- Graph 2: ANN vs Actual and Linear Regression (TEST SET) ---
fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
ax1.scatter(y_test, y_test_pred_ann, edgecolor='black', s=80, alpha=0.7)
ax1.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', label='Perfect Prediction')
ax1.set_xlabel('Actual Tip ($)')
ax1.set_ylabel('Predicted Tip ($)')
ax1.set_title('ANN: Predicted vs Actual')
ax1.legend()
ax1.grid(True, alpha=0.3)
ax2.scatter(y_test, y_test_pred_lr, edgecolor='black', s=80, alpha=0.7, color='green')
ax2.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', label='Perfect Prediction')
ax2.set_xlabel('Actual Tip ($)')
ax2.set_ylabel('Predicted Tip ($)')
ax2.set_title('Linear Regression: Predicted vs Actual')
ax2.legend()
ax2.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('prev_ann_graph2_predictions.png', dpi=200)
plt.show()

# --- Graph 3: Overfitting Indicator ---
fig3, ax = plt.subplots(figsize=(8, 5))
gap = np.array(tracker.val_losses) - np.array(tracker.train_losses)
ax.plot(tracker.epochs, gap, color='purple', lw=2, label='Val Loss - Train Loss')
ax.axhline(0, color='black', ls='--', lw=1)
ax.fill_between(tracker.epochs, 0, gap, where=(gap>=0), color='red', alpha=0.3, label='Overfitting Region')
ax.set_xlabel('Epoch')
ax.set_ylabel('Validation Loss - Training Loss (MSE)')
ax.set_title('Overfitting Indicator')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('prev_ann_graph3_overfit.png', dpi=200)
plt.show()

# ---- Print concise model summary and results ---
ann_r2 = r2_score(y_test, y_test_pred_ann)
ann_mae = mean_absolute_error(y_test, y_test_pred_ann)
ann_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred_ann))
lr_r2 = r2_score(y_test, y_test_pred_lr)
lr_mae = mean_absolute_error(y_test, y_test_pred_lr)
lr_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred_lr))

print(f"\nModel summary (No interactions):")
print(f"ANN - Test R²: {ann_r2:.4f}, MAE: ${ann_mae:.3f}, RMSE: ${ann_rmse:.3f}")
print(f"Linear Regression - Test R²: {lr_r2:.4f}, MAE: ${lr_mae:.3f}, RMSE: ${lr_rmse:.3f}")
