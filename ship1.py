import streamlit as st
import pandas as pd


# 標題
st.title("船舶燃油效率儀表板")


# 讀取 CSV 檔案
file_path = "ship_fuel_efficiency.csv"
df = pd.read_csv(file_path)


# 將英文月份轉成數字
month_map = {
    'January': 1, 'February': 2, 'March': 3, 'April': 4,
    'May': 5, 'June': 6, 'July': 7, 'August': 8,
    'September': 9, 'October': 10, 'November': 11, 'December': 12
}
df['month_num'] = df['month'].map(month_map)


# 側邊欄篩選器
st.sidebar.header("篩選條件")


# 船舶類型篩選
ship_type_options = ["All"] + df['ship_type'].dropna().unique().tolist()
selected_ship_type = st.sidebar.selectbox("選擇船舶類型", ship_type_options)


# 燃料種類篩選
fuel_type_options = ["All"] + df['fuel_type'].dropna().unique().tolist()
selected_fuel_type = st.sidebar.selectbox("選擇燃料種類", fuel_type_options)


# 月份範圍篩選（顯示 1 到 12）
selected_month_range = st.sidebar.slider("選擇月份範圍 (1-12)", 1, 12, (1, 12))


# 篩選資料
filtered_df = df.copy()


if selected_ship_type != "All":
    filtered_df = filtered_df[filtered_df['ship_type'] == selected_ship_type]


if selected_fuel_type != "All":
    filtered_df = filtered_df[filtered_df['fuel_type'] == selected_fuel_type]


filtered_df = filtered_df[
    (filtered_df['month_num'] >= selected_month_range[0]) &
    (filtered_df['month_num'] <= selected_month_range[1])
]


# 顯示結果
st.subheader("篩選後的資料")
st.dataframe(filtered_df)