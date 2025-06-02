# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


st.title("船舶燃油效率分析儀表板")
st.success('分析環境載入成功 ✅')
st.info("請使用側邊欄進行篩選與互動分析", icon='ℹ')


# 載入資料
df = pd.read_csv("ship_fuel_efficiency.csv")


st.header("原始資料預覽")
st.dataframe(df.head(50))


st.sidebar.header("🔎 資料篩選器")
if 'engine_efficiency' in df.columns:
    eff_min, eff_max = float(df['engine_efficiency'].min()), float(df['engine_efficiency'].max())
    eff_range = st.sidebar.slider("效率範圍 (Engine Efficiency)", eff_min, eff_max, (eff_min, eff_max))
else:
    eff_range = None


ship_type = st.sidebar.selectbox("船型 (Ship Type)", ["All"] + sorted(df['ship_type'].dropna().unique())) if 'ship_type' in df.columns else "All"
fuel_type = st.sidebar.selectbox("燃料種類 (Fuel Type)", ["All"] + sorted(df['fuel_type'].dropna().unique())) if 'fuel_type' in df.columns else "All"


# 資料篩選
filtered_df = df.copy()
if eff_range:
    filtered_df = filtered_df[(filtered_df['engine_efficiency'] >= eff_range[0]) & (filtered_df['engine_efficiency'] <= eff_range[1])]
if ship_type != "All":
    filtered_df = filtered_df[filtered_df['ship_type'] == ship_type]
if fuel_type != "All":
    filtered_df = filtered_df[filtered_df['fuel_type'] == fuel_type]


st.subheader("篩選後的資料")
st.dataframe(filtered_df)


st.header("統計摘要")
st.write(filtered_df.describe())


st.header("欄位相關係數 Heatmap")
if set(['distance', 'fuel_consumption', 'engine_efficiency']).issubset(filtered_df.columns):
    corr_df = filtered_df[['distance', 'fuel_consumption', 'engine_efficiency']].dropna()
    fig_corr, ax = plt.subplots()
    sns.heatmap(corr_df.corr(), annot=True, cmap='coolwarm', ax=ax)
    st.pyplot(fig_corr)


st.header("互動式圖表分析")
tab1, tab2 = st.tabs(["📦 箱型圖", "⚫ 散佈圖"])


with tab1:
    if 'fuel_type' in filtered_df.columns and 'engine_efficiency' in filtered_df.columns:
        fig1 = px.box(filtered_df, x="fuel_type", y="engine_efficiency", title="燃料種類與效率的箱型圖")
        st.plotly_chart(fig1)


with tab2:
    if 'distance' in filtered_df.columns and 'engine_efficiency' in filtered_df.columns:
        fig2 = px.scatter(filtered_df, x="distance", y="engine_efficiency", color="ship_type" if 'ship_type' in filtered_df.columns else None, title="航程與效率的關係")
        st.plotly_chart(fig2)


st.header("🎯 標準化後的線性迴歸模型：預測效率")


if set(["distance", "fuel_consumption", "engine_efficiency"]).issubset(filtered_df.columns):
    model_df = filtered_df[["distance", "fuel_consumption", "engine_efficiency"]].dropna()
    X = model_df[["distance", "fuel_consumption"]]
    y = model_df["engine_efficiency"]


    # 資料標準化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)


    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    score = model.score(X_test, y_test)


    st.write(f"模型準確度 R²：{score:.2f}")


    fig_pred = px.scatter(x=y_test, y=y_pred, labels={'x': '實際效率', 'y': '預測效率'}, title="實際 vs 預測 效率")
    fig_pred.add_shape(type='line', x0=y_test.min(), y0=y_test.min(), x1=y_test.max(), y1=y_test.max(), line=dict(color='red', dash='dash'))
    st.plotly_chart(fig_pred)


    st.subheader("🔍 使用者輸入預測效率")
    input_distance = st.number_input("輸入航程距離 (distance)", min_value=0.0, value=100.0)
    input_fc = st.number_input("輸入燃料消耗 (fuel_consumption)", min_value=0.0, value=1000.0)


    if st.button("預測效率"):
        input_scaled = scaler.transform([[input_distance, input_fc]])
        prediction = model.predict(input_scaled)[0]
        st.success(f"🌟 預測燃油效率為：{prediction:.2f}")