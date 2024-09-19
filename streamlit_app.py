import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

# ตั้งค่าหน้าเว็บ Streamlit
st.set_page_config(page_title='การพยากรณ์ด้วย Linear Regression', page_icon=':ocean:')

# ชื่อของแอป
st.title("การจัดการค่าระดับน้ำและการพยากรณ์ด้วย Linear Regression")

# ฟังก์ชันสำหรับการแสดงกราฟข้อมูลหลังลบค่า
def plot_filtered_data(data):
    fig = px.line(data, x=data.index, y='wl_up', title='Water Level Over Time (Filtered Data)', labels={'x': 'Date', 'wl_up': 'Water Level (wl_up)'})
    fig.update_layout(xaxis_title="Date", yaxis_title="Water Level (wl_up)")
    return fig

# ฟังก์ชันสำหรับการแสดงกราฟข้อมูลช่วงที่เลือกและกราฟพยากรณ์
def plot_selected_and_forecasted(data, forecasted, start_date, end_date):
    selected_data = data[(data.index.date >= start_date) & (data.index.date <= end_date)]
    
    # สร้างกราฟข้อมูลช่วงเวลาที่เลือก
    fig = px.line(selected_data, x=selected_data.index, y='wl_up', title=f'Water Level from {start_date} to {end_date} (with Forecast)', labels={'x': 'Date', 'wl_up': 'Water Level (wl_up)'})
    
    # เพิ่มกราฟการพยากรณ์ (เส้นสีเขียว)
    fig.add_scatter(x=forecasted.index, y=forecasted['wl_up'], mode='lines', name='Forecasted Values', line=dict(color='green'))

    fig.update_layout(xaxis_title="Date", yaxis_title="Water Level (wl_up)")
    return fig

# ฟังก์ชันสำหรับการพยากรณ์ด้วย LinearRegression
def forecast_with_regression(data, forecast_start_date):
    forecasted_data = pd.DataFrame(index=pd.date_range(start=forecast_start_date, periods=2*96, freq='15T'))  # พยากรณ์ 2 วันถัดไป
    forecasted_data['wl_up'] = np.nan
    
    # สร้าง lag features ย้อนหลัง 14 วัน (336 rows = 14 วัน)
    for i in range(1, 337):
        data[f'lag_{i}'] = data['wl_up'].shift(i)

    # ใช้ข้อมูลย้อนหลัง 14 วันล่าสุดในการพยากรณ์
    for idx in forecasted_data.index:
        X_train = data.loc[idx - pd.Timedelta(days=14): idx - pd.Timedelta(minutes=15)].dropna().iloc[-336:][[f'lag_{i}' for i in range(1, 337)]]
        y_train = data.loc[idx - pd.Timedelta(days=14): idx - pd.Timedelta(minutes=15)].dropna().iloc[-336:]['wl_up']

        if len(X_train) == 336:
            model = LinearRegression()
            model.fit(X_train, y_train)
            X_pred = X_train.iloc[-1].values.reshape(1, -1)  # ใช้ข้อมูลล่าสุดที่ถูกต้อง
            forecasted_value = model.predict(X_pred)[0]
            forecasted_data.loc[idx, 'wl_up'] = forecasted_value

    return forecasted_data

# อัปโหลดไฟล์ CSV ข้อมูลจริง
uploaded_file = st.file_uploader("เลือกไฟล์ CSV ข้อมูลจริง", type="csv")

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    data['datetime'] = pd.to_datetime(data['datetime'])
    data['datetime'] = data['datetime'].dt.tz_localize(None)
    data.set_index('datetime', inplace=True)
    data['hour'] = data.index.hour
    data['day_of_week'] = data.index.dayofweek
    data['minute'] = data.index.minute
    data['lag_1'] = data['wl_up'].shift(1)
    data['lag_2'] = data['wl_up'].shift(2)
    data['lag_1'].ffill(inplace=True)
    data['lag_2'].ffill(inplace=True)

    # ตัดข้อมูลที่มีค่า wl_up น้อยกว่า 100 ออก
    filtered_data = data[data['wl_up'] >= 100]

    # แสดงตัวอย่างข้อมูลหลังจากลบค่าที่น้อยกว่า 100 ออก
    st.subheader('แสดงตัวอย่างข้อมูลหลังจากลบค่าที่น้อยกว่า 100 ออก')
    st.write(filtered_data.head())

    # แสดงกราฟข้อมูลหลังจากลบค่าที่น้อยกว่า 100 ออก
    st.subheader('กราฟข้อมูลหลังจากลบค่าที่น้อยกว่า 100 ออก')
    st.plotly_chart(plot_filtered_data(filtered_data))

    # ให้ผู้ใช้เลือกช่วงวันที่ที่สนใจและพยากรณ์ต่อจากข้อมูลที่เลือก
    st.subheader("เลือกช่วงวันที่ที่สนใจและแสดงการพยากรณ์ต่อ")
    start_date = st.date_input("เลือกวันเริ่มต้น (ดูข้อมูล)", pd.to_datetime(filtered_data.index.min()).date())
    end_date = st.date_input("เลือกวันสิ้นสุด (ดูข้อมูล)", pd.to_datetime(filtered_data.index.max()).date())

    if st.button("ตกลง (พยากรณ์ต่อจากช่วงที่เลือก)"):
        # พยากรณ์ 2 วันถัดไปจากช่วงวันที่เลือก
        forecast_start_date = end_date + pd.Timedelta(days=1)
        forecasted_data = forecast_with_regression(filtered_data, forecast_start_date)

        # แสดงกราฟข้อมูลช่วงเวลาที่เลือกและกราฟการพยากรณ์
        st.plotly_chart(plot_selected_and_forecasted(filtered_data, forecasted_data, start_date, end_date))

