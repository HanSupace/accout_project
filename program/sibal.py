import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.linear_model import Ridge
from sklearn.ensemble import StackingRegressor

# 📥 데이터 불러오기
df = pd.read_excel("final.xlsx")

# ✅ IQR 기반 이상치 클리핑
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


# ✅ 로그 변환
df_cleaned["log_traffic"] = np.log1p(df_cleaned["traffic"])
df_cleaned["log_store_count"] = np.log1p(df_cleaned["store_count"])
df_cleaned["log_rent_price"] = np.log1p(df_cleaned["rent_price"])
df_cleaned["log_working_population"] = np.log1p(df_cleaned["working_population"])
df_cleaned["rent_to_income_ratio"] = df_cleaned["log_rent_price"] / (df_cleaned["income_quintile"] + 1e-5)

# ✅ 파생 변수
df_cleaned["store_density"] = df_cleaned["traffic"] / (df_cleaned["store_count"] + 1e-5)
df_cleaned["rent_div_income"] = df_cleaned["rent_price"] / (df_cleaned["income_quintile"] + 1e-5)

# ✅ 원-핫 인코딩
df_encoded = pd.get_dummies(df_cleaned, columns=["category", "district","neighborhood"], drop_first=True)
df_encoded.columns = df_encoded.columns.str.replace(r'[\"\\\/{}\[\]:.,]', '_', regex=True)

# ✅ category × old 상호작용 항
category_cols = [col for col in df_encoded.columns if col.startswith("category_")]
for col in category_cols:
    df_encoded[f"{col}_old_interaction"] = df_encoded[col] * df_encoded["old"]

# ✅ 사용 변수 설정
selected_cols = [
    'log_rent_price', 'log_traffic', 'income_quintile', 'store_density', 'old', 'log_working_population','rent_to_income_ratio',"rent_div_income"
] + category_cols + \
    [col for col in df_encoded.columns if col.startswith("district_")] + \
    [col for col in df_encoded.columns if col.startswith("neighborhood_")] +\
    [f"{col}_old_interaction" for col in category_cols]

X = df_encoded[selected_cols]
y = np.log1p(df_encoded["avg_month_sales"])

# 📊 학습/테스트 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ✅ 개별 모델 정의 (XGB 파라미터는 GridSearch 결과값으로 설정)
xgb = XGBRegressor(
    n_estimators=500, max_depth=6, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8,
    random_state=42
)

lgb = LGBMRegressor(n_estimators=500, learning_rate=0.05, max_depth=6, random_state=42)
ridge = Ridge(alpha=8.0)

# ✅ Stacking 앙상블 모델 구성
stack_model = StackingRegressor(
    estimators=[
        ('xgb', xgb),
        ('lgb', lgb)
    ],
    final_estimator=ridge,
    n_jobs=-1
)

# 🏋️ 학습
stack_model.fit(X_train, y_train)

# ✅ 평가
y_pred_stack = stack_model.predict(X_test)
r2_stack = r2_score(y_test, y_pred_stack)
mae_stack_log = mean_absolute_error(y_test, y_pred_stack)
mae_stack_real = mean_absolute_error(np.expm1(y_test), np.expm1(y_pred_stack))

# 📢 출력
print("✅ R² Score:", r2_stack)
print("✅ MAE (log scale):", mae_stack_log)
print("✅ MAE (실제 단위):", mae_stack_real)

import joblib
joblib.dump(stack_model, "stacked_model.pkl")
print("✅ 모델이 stacked_model.pkl로 저장되었습니다.")

joblib.dump(selected_cols, "feature_columns.pkl")
print("✅ feature 목록이 feature_columns.pkl로 저장되었습니다.")
