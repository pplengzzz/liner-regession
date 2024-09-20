import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.linear_model import Ridge
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

# ฟังก์ชันสำหรับการพยากรณ์ด้วย Ridge Regression ที่ปรับปรุงแล้ว
def forecast_with_regression(data, forecast_start_date):
    # สร้าง DataFrame สำหรับการพยากรณ์ 1 วันถัดไป (96 ช่วงเวลา ทุกๆ 15 นาที)
    forecasted_data = pd.DataFrame(index=pd.date_range(start=forecast_start_date, periods=96, freq='15T'))
    forecasted_data['wl_up'] = np.nan

    # ใช้ข้อมูลย้อนหลัง 7 วันในการเทรนโมเดล
    training_data_end = forecast_start_date - pd.Timedelta(minutes=15)
    training_data_start = training_data_end - pd.Timedelta(days=7) + pd.Timedelta(minutes=15)

    # ปรับช่วงข้อมูลหากมีข้อมูลไม่เพียงพอ
    if training_data_start < data.index.min():
        training_data_start = data.index.min()

    training_data = data.loc[training_data_start:training_data_end].copy()

    # เติมค่า missing values ด้วยการ interpolate
    training_data['wl_up'].interpolate(method='time', inplace=True)

    # สร้างฟีเจอร์เพิ่มเติม เช่น Moving Average และ Trend
    training_data['ma_96'] = training_data['wl_up'].rolling(window=96).mean()  # ค่าเฉลี่ยเคลื่อนที่ 1 วัน
    training_data['trend'] = np.arange(len(training_data))  # เทรนด์เชิงเส้น

    # สร้างฟีเจอร์ lag หลายระดับ
    lags = [1, 4, 96, 192, 288, 384, 480, 576]  # ใช้ข้อมูลย้อนหลังถึง 6 วัน
    for lag in lags:
        training_data[f'lag_{lag}'] = training_data['wl_up'].shift(lag)

    # ลบแถวที่มีค่า NaN ในฟีเจอร์
    training_data.dropna(inplace=True)

    # รวมฟีเจอร์เวลา
    training_data['hour'] = training_data.index.hour
    training_data['minute'] = training_data.index.minute
    training_data['day_of_week'] = training_data.index.dayofweek

    # ฟีเจอร์และตัวแปรเป้าหมาย
    feature_cols = [f'lag_{lag}' for lag in lags] + ['ma_96', 'trend', 'hour', 'minute', 'day_of_week']
    X_train = training_data[feature_cols]
    y_train = training_data['wl_up']

    # ตรวจสอบว่ามีข้อมูลเพียงพอหรือไม่
    if X_train.empty:
        st.error("ไม่สามารถพยากรณ์ได้เนื่องจากข้อมูลสำหรับการเทรนไม่เพียงพอ")
        return pd.DataFrame()

    # เทรนโมเดล Ridge Regression
    model = Ridge(alpha=1.0)
    model.fit(X_train, y_train)

    # การพยากรณ์
    for i in range(len(forecasted_data)):
        forecast_time = forecasted_data.index[i]

        # สร้างฟีเจอร์ lag
        lag_features = {}
        for lag in lags:
            lag_time = forecast_time - pd.Timedelta(minutes=15 * lag)
            if lag_time in data.index:
                lag_value = data.at[lag_time, 'wl_up']
            elif lag_time in forecasted_data.index:
                lag_value = forecasted_data.at[lag_time, 'wl_up']
            else:
                lag_value = np.nan
            lag_features[f'lag_{lag}'] = lag_value

        # ถ้ามีค่า lag ที่หายไป ให้ข้ามการพยากรณ์
        if np.any(pd.isnull(list(lag_features.values()))):
            continue

        # ฟีเจอร์เพิ่มเติม
        # Moving Average
        ma_time = forecast_time - pd.Timedelta(minutes=15)
        if ma_time in data.index:
            ma_96 = data.loc[ma_time - pd.Timedelta(minutes=15*95): ma_time]['wl_up'].mean()
        elif ma_time in forecasted_data.index:
            recent_values = forecasted_data['wl_up'].dropna()
            if len(recent_values) >= 96:
                ma_96 = recent_values.iloc[-96:].mean()
            else:
                ma_96 = np.nan
        else:
            ma_96 = np.nan
        lag_features['ma_96'] = ma_96

        # ถ้ามีค่า ma_96 ที่หายไป ให้ข้ามการพยากรณ์
        if np.isnan(ma_96):
            continue

        # Trend
        lag_features['trend'] = len(training_data) + i  # ต่อเนื่องจากข้อมูลเทรน

        # ฟีเจอร์เวลา
        lag_features['hour'] = forecast_time.hour
        lag_features['minute'] = forecast_time.minute
        lag_features['day_of_week'] = forecast_time.dayof_week

        X_pred = pd.DataFrame([lag_features])

        # พยากรณ์ค่า
        forecasted_value = model.predict(X_pred)[0]
        forecasted_data.at[forecast_time, 'wl_up'] = forecasted_value

    return forecasted_data

# อัปโหลดไฟล์ CSV ข้อมูลจริง
uploaded_file = st.file_uploader("เลือกไฟล์ CSV ข้อมูลจริง", type="csv")

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    data['datetime'] = pd.to_datetime(data['datetime'])
    data['datetime'] = data['datetime'].dt.tz_localize(None)
    data.set_index('datetime', inplace=True)

    # ตัดข้อมูลที่มีค่า wl_up น้อยกว่า 100 ออก
    filtered_data = data[data['wl_up'] >= 100]

    # แสดงกราฟข้อมูลหลังจากลบค่าที่น้อยกว่า 100 ออก
    st.subheader('กราฟข้อมูลหลังจากลบค่าที่น้อยกว่า 100 ออก')
    st.plotly_chart(plot_filtered_data(filtered_data))

    # ให้ผู้ใช้เลือกช่วงวันที่ที่สนใจและพยากรณ์ต่อจากข้อมูลที่เลือก
    st.subheader("เลือกช่วงวันที่ที่สนใจและแสดงการพยากรณ์ต่อ")
    start_date = st.date_input("เลือกวันเริ่มต้น (ดูข้อมูล)", pd.to_datetime(filtered_data.index.min()).date())
    end_date = st.date_input("เลือกวันสิ้นสุด (ดูข้อมูล)", pd.to_datetime(filtered_data.index.max()).date())

    if st.button("ตกลง (พยากรณ์ต่อจากช่วงที่เลือก)"):
        # เลือกข้อมูลช่วงวันที่ที่สนใจ
        selected_data = filtered_data[(filtered_data.index.date >= start_date) & (filtered_data.index.date <= end_date)]

        # ตรวจสอบว่ามีข้อมูลเพียงพอหรือไม่
        if selected_data.empty:
            st.error("ไม่มีข้อมูลในช่วงวันที่ที่เลือก กรุณาเลือกวันที่ใหม่")
        else:
            # กำหนดวันที่เริ่มพยากรณ์เป็นวันถัดไปจากวันที่สิ้นสุดที่เลือก
            forecast_start_date = pd.to_datetime(end_date) + pd.Timedelta(days=1)

            # พยากรณ์ 1 วันถัดไปจากวันที่ที่เลือก
            forecasted_data = forecast_with_regression(filtered_data, forecast_start_date)

            # ตรวจสอบว่ามีการพยากรณ์หรือไม่
            if not forecasted_data.empty:
                # แสดงกราฟข้อมูลช่วงเวลาที่เลือกและกราฟการพยากรณ์
                st.plotly_chart(plot_selected_and_forecasted(filtered_data, forecasted_data, start_date, end_date))

                # คำนวณค่าประเมินผล (ถ้ามีข้อมูลจริงในช่วงพยากรณ์)
                actual_data = filtered_data.loc[forecasted_data.index]
                if not actual_data.empty:
                    y_true = actual_data['wl_up']
                    y_pred = forecasted_data['wl_up'].loc[actual_data.index]
                    mae = mean_absolute_error(y_true, y_pred)
                    rmse = mean_squared_error(y_true, y_pred, squared=False)
                    st.write(f"Mean Absolute Error (MAE): {mae:.2f}")
                    st.write(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
            else:
                st.error("ไม่สามารถพยากรณ์ได้เนื่องจากข้อมูลไม่เพียงพอ")


