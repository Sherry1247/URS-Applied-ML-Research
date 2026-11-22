# Trains the 6-input "key" ANN exactly as ann_only_key+visual.py (same split/seed/hyperparameters)
# and saves it as a pkl in the same dict format as ann_all_inputs_results.pkl. No plots.
import os, pickle, warnings
import numpy as np, pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
warnings.filterwarnings('ignore')
root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
df = pd.read_csv(os.path.join(root, 'df_processed.csv'))
key_inputs = ['Torque','p_0','T_IM','P_IM','EGR_Rate','ECU_VTG_Pos']
outputs = ['MF_IA','NOx_EO','SOC']
scaler_X = StandardScaler(); X_norm = scaler_X.fit_transform(df[key_inputs].values)
X_train, X_test = train_test_split(X_norm, test_size=0.2, random_state=42)
res = {}
for out in outputs:
    sy = StandardScaler(); y = sy.fit_transform(df[out].values.reshape(-1,1)).ravel()
    _,_,y_tr,y_te = train_test_split(X_norm, y, test_size=0.2, random_state=42)
    ann = MLPRegressor(hidden_layer_sizes=(64,32,16), activation='relu', solver='adam', alpha=0.0001,
        max_iter=500, random_state=42, early_stopping=True, validation_fraction=0.2, n_iter_no_change=20)
    ann.fit(X_train, y_tr)
    inv = lambda a: sy.inverse_transform(np.asarray(a).reshape(-1,1)).ravel()
    ytr, yte = inv(y_tr), inv(y_te); ptr, pte = inv(ann.predict(X_train)), inv(ann.predict(X_test))
    res[out] = dict(model=ann, scaler_X=scaler_X, scaler_y=sy,
        mae_train=mean_absolute_error(ytr,ptr), rmse_train=np.sqrt(mean_squared_error(ytr,ptr)), r2_train=r2_score(ytr,ptr),
        mae_test=mean_absolute_error(yte,pte), rmse_test=np.sqrt(mean_squared_error(yte,pte)), r2_test=r2_score(yte,pte),
        y_train=ytr, y_test=yte, y_pred_train=ptr, y_pred_test=pte, features=key_inputs)
    print(f"{out}: test MAE={res[out]['mae_test']:.4f} R2={res[out]['r2_test']:.4f}")
p = os.path.join(root,'test_results','visualizations','ann_key_inputs_results.pkl')
pickle.dump(res, open(p,'wb')); print('saved', p)
