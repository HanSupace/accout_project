import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
from xgboost import XGBRegressor

df = pd.read_excel("final.xlsx")

num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
def clip_outliers_iqr(df, cols, factor=2.0):
    df_clipped = df.copy()
    for col in cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - factor * IQR
        upper = Q3 + factor * IQR
        df_clipped[col] = np.clip(df[col], lower, upper)
    return df_clipped

df_cleaned = clip_outliers_iqr(df.copy(), num_cols, factor=2.0)

df_cleaned["log_traffic"] = np.log1p(df_cleaned["traffic"])
df_cleaned["log_store_count"] = np.log1p(df_cleaned["store_count"])
df_cleaned["log_rent_price"] = np.log1p(df_cleaned["rent_price"])
df_cleaned["log_working_population"] = np.log1p(df_cleaned["working_population"])
df_cleaned["rent_to_income_ratio"] = df_cleaned["log_rent_price"] / (df_cleaned["income_quintile"] + 1e-5)
df_cleaned["store_density"] = df_cleaned["traffic"] / (df_cleaned["store_count"] + 1e-5)
df_cleaned["rent_div_income"] = df_cleaned["rent_price"] / (df_cleaned["income_quintile"] + 1e-5)

df_encoded = pd.get_dummies(df_cleaned, columns=["category", "district", "neighborhood"], drop_first=True)

# ✅ category × old 상호작용 항
category_cols = [col for col in df_encoded.columns if col.startswith("category_")]
for col in category_cols:
    df_encoded[f"{col}_old_interaction"] = df_encoded[col] * df_encoded["old"]

# ✅ 사용 변수 설정
selected_cols = [
    'log_rent_price', 'log_traffic', 'income_quintile', 'store_density', 'old', 
    'log_working_population', 'rent_to_income_ratio', 'rent_div_income'
] + category_cols + \
    [col for col in df_encoded.columns if col.startswith("district_")] + \
    [f"{col}_old_interaction" for col in category_cols]

X = df_encoded[selected_cols]
y = np.log1p(df_encoded["avg_annual_sales"])

# 📊 학습/테스트 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ✅ XGBoost 모델 정의 및 학습
xgb = XGBRegressor(
    n_estimators=500, max_depth=6, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8,
     random_state=42
)

xgb.fit(X_train, y_train)

# ✅ 평가
y_pred = xgb.predict(X_test)
r2 = r2_score(y_test, y_pred)
mae_log = mean_absolute_error(y_test, y_pred)
mae_real = mean_absolute_error(np.expm1(y_test), np.expm1(y_pred))

# 📢 출력
print("✅ R² Score (XGBoost):", r2)
print("✅ MAE (log scale):", mae_log)
print("✅ MAE (실제 단위):", mae_real)

# 📦 저장 옵션 (선택사항)
'''
import joblib
joblib.dump(xgb, "xgboost_model.pkl")
joblib.dump(selected_cols, "feature_columns.pkl")
'''

