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

# ฟังก์ชันสำหรับการพยากรณ์ด้วย Linear Regression ที่ปรับปรุงแล้ว
def forecast_with_linear_regression(data, forecast_start_date):
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

    # สร้างฟีเจอร์ lag ที่สำคัญ
    lags = [96, 672]  # ใช้ lag 1 วันและ 7 วันก่อนหน้า
    for lag in lags:
        training_data[f'lag_{lag}'] = training_data['wl_up'].shift(lag)

    # สร้างฟีเจอร์ค่าเฉลี่ยเคลื่อนที่
    training_data['ma_4'] = training_data['wl_up'].rolling(window=4).mean()  # ค่าเฉลี่ยใน 1 ชั่วโมงที่ผ่านมา
    training_data['ma_24'] = training_data['wl_up'].rolling(window=24).mean()  # ค่าเฉลี่ยใน 6 ชั่วโมงที่ผ่านมา
    training_data['ma_48'] = training_data['wl_up'].rolling(window=48).mean()  # ค่าเฉลี่ยใน 12 ชั่วโมงที่ผ่านมา

    # ลบแถวที่มีค่า NaN ในฟีเจอร์
    training_data.dropna(inplace=True)

    # แปลงฟีเจอร์เวลาเป็นรูปแบบวงกลม
    training_data['hour_sin'] = np.sin(2 * np.pi * training_data.index.hour / 24)
    training_data['hour_cos'] = np.cos(2 * np.pi * training_data.index.hour / 24)
    training_data['dayofweek_sin'] = np.sin(2 * np.pi * training_data.index.dayofweek / 7)
    training_data['dayofweek_cos'] = np.cos(2 * np.pi * training_data.index.dayofweek / 7)

    # ฟีเจอร์และตัวแปรเป้าหมาย
    feature_cols = [f'lag_{lag}' for lag in lags] + ['ma_4', 'ma_24', 'ma_48', 'hour_sin', 'hour_cos', 'dayofweek_sin', 'dayofweek_cos']
    X_train = training_data[feature_cols]
    y_train = training_data['wl_up']

    # ตรวจสอบว่ามีข้อมูลเพียงพอหรือไม่
    if X_train.empty:
        st.error("ไม่สามารถพยากรณ์ได้เนื่องจากข้อมูลสำหรับการเทรนไม่เพียงพอ")
        return pd.DataFrame()

    # เทรนโมเดล Linear Regression
    model = LinearRegression()
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

        # สร้างฟีเจอร์ค่าเฉลี่ยเคลื่อนที่
        ma_4 = forecasted_data['wl_up'].iloc[i-4:i].mean() if i >= 4 else np.nan
        ma_24 = forecasted_data['wl_up'].iloc[i-24:i].mean() if i >= 24 else np.nan
        ma_48 = forecasted_data['wl_up'].iloc[i-48:i].mean() if i >= 48 else np.nan

        # ถ้ามีค่า MA ที่หายไป ให้ข้ามการพยากรณ์
        if np.isnan(ma_4) or np.isnan(ma_24) or np.isnan(ma_48):
            continue

        lag_features['ma_4'] = ma_4
        lag_features['ma_24'] = ma_24
        lag_features['ma_48'] = ma_48

        # แปลงฟีเจอร์เวลาเป็นรูปแบบวงกลม
        lag_features['hour_sin'] = np.sin(2 * np.pi * forecast_time.hour / 24)
        lag_features['hour_cos'] = np.cos(2 * np.pi * forecast_time.hour / 24)
        lag_features['dayofweek_sin'] = np.sin(2 * np.pi * forecast_time.dayofweek / 7)
        lag_features['dayofweek_cos'] = np.cos(2 * np.pi * forecast_time.dayofweek / 7)

        X_pred = pd.DataFrame([lag_features])

        # พยากรณ์ค่า
        forecasted_value = model.predict(X_pred)[0]
        forecasted_data.at[forecast_time, 'wl_up'] = forecasted_value

    # ใช้ moving average เพื่อทำให้ค่าพยากรณ์เรียบขึ้น
    forecasted_data['wl_up'] = forecasted_data['wl_up'].rolling(window=4, min_periods=1).mean()

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
            forecasted_data = forecast_with_linear_regression(filtered_data, forecast_start_date)

            # ตรวจสอบว่ามีการพยากรณ์หรือไม่
            if not forecasted_data.empty:
                # แสดงกราฟข้อมูลช่วงเวลาที่เลือกและกราฟการพยากรณ์
                st.plotly_chart(plot_selected_and_forecasted(filtered_data, forecasted_data, start_date, end_date))

                # ตรวจสอบว่ามีข้อมูลจริงสำหรับช่วงเวลาที่พยากรณ์หรือไม่
                common_indices = forecasted_data.index.intersection(filtered_data.index)
                if not common_indices.empty:
                    actual_data = filtered_data.loc[common_indices]
                    y_true = actual_data['wl_up']
                    y_pred = forecasted_data['wl_up'].loc[common_indices]
                    mae = mean_absolute_error(y_true, y_pred)
                    rmse = mean_squared_error(y_true, y_pred, squared=False)
                    st.write(f"Mean Absolute Error (MAE): {mae:.2f}")
                    st.write(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
                else:
                    st.info("ไม่มีข้อมูลจริงสำหรับช่วงเวลาที่พยากรณ์ ไม่สามารถคำนวณค่า MAE และ RMSE ได้")
            else:
                st.error("ไม่สามารถพยากรณ์ได้เนื่องจากข้อมูลไม่เพียงพอ")


