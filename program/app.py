import streamlit as st
import numpy as np
import pandas as pd
import joblib
from pyproj import Transformer
import pydeck as pdk

import os
import streamlit as st

import os
import streamlit as st

st.title("📂 디버깅 도구")
st.write("현재 작업 디렉토리:", os.getcwd())
st.write("파일 목록:", os.listdir(os.getcwd()))
st.write("📁 program 내부 목록:", os.listdir("program"))


model = joblib.load("program/stacked_model.pkl")
feature_columns = joblib.load("program/feature_columns.pkl")

df = pd.read_excel("final.xlsx")
area_df = df[["district", "neighborhood"]].drop_duplicates()

coords_df = pd.read_excel("xy.xlsx") 

transformer = Transformer.from_crs("EPSG:5174", "EPSG:4326", always_xy=True)

def transform_xy(row):
    lon, lat = transformer.transform(row["x"], row["y"])
    return pd.Series({"lon": lon, "lat": lat})

coords_df[["lon", "lat"]] = coords_df.apply(transform_xy, axis=1)

category_options = sorted(df["category"].unique())

st.set_page_config(page_title="서울시 매출 예측", layout="centered")
st.title("서울시 매출 예측 AI")
st.markdown("입력한 조건으로 예상 **월 매출**을 예측합니다.")

st.subheader("지역과 업종을 선택해주세요")
selected_district = st.selectbox("자치구 선택", sorted(area_df["district"].unique()))
filtered_neighborhoods = area_df[area_df["district"] == selected_district]["neighborhood"].unique()
selected_neighborhood = st.selectbox("동 선택", sorted(filtered_neighborhoods))
selected_category = st.selectbox("업종 선택", category_options)

st.divider()
st.subheader("상권 정보 입력")
rent_price = st.slider("월 임대료 (평당)", 10_000.0, 10_000_000.0, 5_005_000.0, step=10_000.0)
traffic = st.slider("유동 인구 수", 100.0, 100_000.0, 50_050.0, step=100.0)
store_count = st.slider("유사 점포 수", 1.0, 5000.0, 1000.0, step=1.0)
income_quintile = st.slider("소득 분위", min_value=1, max_value=10, value=5)
old = st.slider("고령 인구 수 (65세 이상)", 0.0, 200_000.0, 100_000.0, step=1_000.0)
working_population = st.slider("직장인 인구 수", 100.0, 100_000.0, 50_050.0, step=100.0)


store_density = traffic / (store_count + 1e-5)
log_rent_price = np.log1p(rent_price)
log_traffic = np.log1p(traffic)
log_working_population = np.log1p(working_population)
rent_to_income_ratio = log_rent_price / (income_quintile + 1e-5)
rent_div_income = rent_price / (income_quintile + 1e-5)

input_dict = {
    'log_rent_price': log_rent_price,
    'log_traffic': log_traffic,
    'income_quintile': income_quintile,
    'store_density': store_density,
    'old': old,
    'log_working_population': log_working_population,
    'rent_to_income_ratio': rent_to_income_ratio,
    'rent_div_income': rent_div_income,
}

X_input = pd.DataFrame([0] * len(feature_columns), index=feature_columns).T
for key, value in input_dict.items():
    if key in X_input.columns:
        X_input[key] = value

for pre, val in [("district", selected_district), ("neighborhood", selected_neighborhood), ("category", selected_category)]:
    col_name = f"{pre}_{val}"
    if col_name in X_input.columns:
        X_input[col_name] = 1

st.divider()
if st.button("매출 예측해보기💡"):
    log_pred = model.predict(X_input)[0]
    pred_sales = np.expm1(log_pred)
    st.success(f"예상 월 매출: **{pred_sales:,.0f} 만원**")

    selected_coords = coords_df[coords_df["neighborhood"] == selected_neighborhood]
    if not selected_coords.empty:
        lat = selected_coords["lat"].values[0]
        lon = selected_coords["lon"].values[0]

        if 35.0 <= lat <= 40.0 and 125.0 <= lon <= 130.0:
            st.subheader("선택한 지역 위치")

            map_data = pd.DataFrame([{
                "lat": lat,
                "lon": lon,
                "tooltip": f"{selected_district} {selected_neighborhood}\n업종: {selected_category}\n예측 매출: {pred_sales:,.0f}만원"
            }])
            layer = pdk.Layer(
                "ScatterplotLayer",
                data=map_data,
                get_position='[lon, lat]',
                get_radius=300,
                get_fill_color=[255, 140, 0, 120],
                pickable=True
            )

            view_state = pdk.ViewState(latitude=lat, longitude=lon, zoom=13)
            st.pydeck_chart(pdk.Deck(
                layers=[layer],
                initial_view_state=view_state,
                tooltip={"text": "{tooltip}"}
            ))

        else:
            st.warning("지도를 표시할 수 없습니다.")
    else:
        st.warning("해당 동에 대한 좌표 정보가 없습니다.")
