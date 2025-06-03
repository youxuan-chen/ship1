import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


# 1. 標題與資料載入
st.title("船舶燃油效率分析系統")


file_path = "ship_fuel_efficiency.csv"  # 請確認與 .py 同目錄或改為正確路徑
df = pd.read_csv(file_path)


# 修正月份欄位轉換
month_map = {
    'January': 1, 'February': 2, 'March': 3, 'April': 4,
    'May': 5, 'June': 6, 'July': 7, 'August': 8,
    'September': 9, 'October': 10, 'November': 11, 'December': 12
}
df["month_num"] = df["month"].map(month_map)


st.success("✅ 資料載入成功！")
st.info("請使用左側選單進行篩選")


# 2. 側邊欄篩選器
st.sidebar.header("🔎 篩選條件")


ship_type = st.sidebar.selectbox("船舶類型", ["All"] + df["ship_type"].unique().tolist())
fuel_type = st.sidebar.selectbox("燃料種類", ["All"] + df["fuel_type"].unique().tolist())
month_range = st.sidebar.slider("月份範圍 (1-12)", 1, 12, (1, 12))


# 資料篩選
filtered_df = df.copy()
if ship_type != "All":
    filtered_df = filtered_df[filtered_df["ship_type"] == ship_type]
if fuel_type != "All":
    filtered_df = filtered_df[filtered_df["fuel_type"] == fuel_type]


filtered_df = filtered_df[
    (filtered_df["month_num"] >= month_range[0]) &
    (filtered_df["month_num"] <= month_range[1])
]


st.subheader("📄 篩選後資料")
st.dataframe(filtered_df)


# 3. 描述統計
st.header("📊 統計摘要")
st.write(filtered_df.describe())


# 4. 圖表視覺化
st.header("📈 圖表分析")
tab1, tab2, tab3 = st.tabs(["箱型圖", "散佈圖", "直方圖"])


with tab1:
    st.plotly_chart(px.box(filtered_df, y="fuel_consumption", title="燃料消耗 Box Plot"))


with tab2:
    st.plotly_chart(px.scatter(filtered_df, x="engine_efficiency", y="CO2_emissions",
                               color="fuel_type", title="引擎效率 vs CO2 排放"))


with tab3:
    st.plotly_chart(px.histogram(filtered_df, x="CO2_emissions", nbins=30, title="CO2 排放分布"))


# 5. 線性迴歸模型
st.header("🎯 線性迴歸模型：預測 CO2 排放量")


model_df = filtered_df[["distance", "fuel_consumption", "engine_efficiency", "CO2_emissions"]].dropna()


if model_df.shape[0] >= 20:  # 避免資料太少導致模型錯誤
    X = model_df[["distance", "fuel_consumption", "engine_efficiency"]]
    y = model_df["CO2_emissions"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    score = model.score(X_test, y_test)


    st.write(f"📈 模型準確度 R²：{score:.2f}")


    # 實際 vs 預測圖
    fig_pred = px.scatter(x=y_test, y=y_pred, labels={'x': '實際值', 'y': '預測值'},
                          title="實際 vs 預測 CO2 排放量")
    fig_pred.add_shape(
        type='line', x0=y_test.min(), y0=y_test.min(), x1=y_test.max(), y1=y_test.max(),
        line=dict(color='red', dash='dash')
    )
    st.plotly_chart(fig_pred)


    # 6. 使用者輸入預測
    st.subheader("🔍 輸入數值預測 CO2 排放")
    input_dist = st.number_input("航行距離 (km)", min_value=0.0, value=500.0)
    input_fuel = st.number_input("燃料消耗 (公升)", min_value=0.0, value=100.0)
    input_eff = st.number_input("引擎效率", min_value=0.0, value=0.9)


    if st.button("立即預測"):
        user_input = pd.DataFrame([[input_dist, input_fuel, input_eff]],
                                  columns=["distance", "fuel_consumption", "engine_efficiency"])
        co2_pred = model.predict(user_input)[0]
        st.success(f"🌍 預測 CO2 排放量為：{co2_pred:.2f}")
else:
    st.warning("⚠ 篩選後資料不足，無法建立可靠模型，請調整條件。")