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
def plot_selected_and_forecasted(data, forecasted):
    # รวมข้อมูลจริงและค่าพยากรณ์
    combined_data = pd.concat([data, forecasted])
    # สร้างกราฟ
    fig = px.line(combined_data, x=combined_data.index, y='wl_up', title='Water Level with Forecast', labels={'x': 'Date', 'wl_up': 'Water Level (wl_up)'})
    fig.update_layout(xaxis_title="Date", yaxis_title="Water Level (wl_up)")
    return fig

# ฟังก์ชันสำหรับการพยากรณ์ด้วย Linear Regression ที่ปรับปรุงแล้ว
def forecast_with_linear_regression(data, forecast_start_date):
    # สร้าง DataFrame สำหรับการพยากรณ์ 1 วันถัดไป (96 ช่วงเวลา ทุกๆ 15 นาที)
    forecast_periods = 96
    forecasted_data = pd.DataFrame(index=pd.date_range(start=forecast_start_date, periods=forecast_periods, freq='15T'))
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

    # สร้างฟีเจอร์ lag หลายระดับ
    lags = [1, 4, 96, 672]  # ใช้ lag 15 นาที, 1 ชั่วโมง, 1 วัน, และ 7 วัน
    for lag in lags:
        training_data[f'lag_{lag}'] = training_data['wl_up'].shift(lag)

    # ลบแถวที่มีค่า NaN ในฟีเจอร์
    training_data.dropna(inplace=True)

    # ตรวจสอบว่ามีข้อมูลเพียงพอหรือไม่
    if training_data.empty:
        st.error("ไม่สามารถพยากรณ์ได้เนื่องจากข้อมูลสำหรับการเทรนไม่เพียงพอ")
        return pd.DataFrame()

    # ฟีเจอร์และตัวแปรเป้าหมาย
    feature_cols = [f'lag_{lag}' for lag in lags]
    X_train = training_data[feature_cols]
    y_train = training_data['wl_up']

    # เทรนโมเดล Linear Regression
    model = LinearRegression()
    model.fit(X_train, y_train)

    # การพยากรณ์
    for idx in forecasted_data.index:
        lag_features = {}
        for lag in lags:
            lag_time = idx - pd.Timedelta(minutes=15 * lag)
            if lag_time in data.index:
                lag_features[f'lag_{lag}'] = data.at[lag_time, 'wl_up']
            elif lag_time in forecasted_data.index:
                lag_features[f'lag_{lag}'] = forecasted_data.at[lag_time, 'wl_up']
            else:
                lag_features[f'lag_{lag}'] = np.nan

        # ถ้ามีค่า lag ที่หายไป ให้ข้ามการพยากรณ์
        if np.any(pd.isnull(list(lag_features.values()))):
            continue

        X_pred = pd.DataFrame([lag_features])

        # พยากรณ์ค่า
        forecast_value = model.predict(X_pred)[0]
        forecasted_data.at[idx, 'wl_up'] = forecast_value

    # ทำให้ค่าพยากรณ์ต่อเนื่องจากค่าจริง
    last_actual_value = data['wl_up'].iloc[-1]
    first_forecast_value = forecasted_data['wl_up'].iloc[0]
    value_diff = last_actual_value - first_forecast_value
    forecasted_data['wl_up'] += value_diff

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
    data = data[data['wl_up'] >= 100]

    # เติมค่า missing values ใน wl_up
    data['wl_up'].interpolate(method='time', inplace=True)

    # แสดงกราฟข้อมูลหลังจากลบค่าที่น้อยกว่า 100 ออก
    st.subheader('กราฟข้อมูลหลังจากลบค่าที่น้อยกว่า 100 ออก')
    st.plotly_chart(plot_filtered_data(data))

    # ให้ผู้ใช้เลือกช่วงวันที่ที่สนใจและพยากรณ์ต่อจากข้อมูลที่เลือก
    st.subheader("เลือกช่วงวันที่ที่สนใจและแสดงการพยากรณ์ต่อ")
    start_date = st.date_input("เลือกวันเริ่มต้น (ดูข้อมูล)", pd.to_datetime(data.index.min()).date())
    end_date = st.date_input("เลือกวันสิ้นสุด (ดูข้อมูล)", pd.to_datetime(data.index.max()).date())

    if st.button("ตกลง (พยากรณ์ต่อจากช่วงที่เลือก)"):
        # เลือกข้อมูลช่วงวันที่ที่สนใจ
        selected_data = data[(data.index.date >= start_date) & (data.index.date <= end_date)]

        # ตรวจสอบว่ามีข้อมูลเพียงพอหรือไม่
        if selected_data.empty:
            st.error("ไม่มีข้อมูลในช่วงวันที่ที่เลือก กรุณาเลือกวันที่ใหม่")
        else:
            # กำหนดวันที่เริ่มพยากรณ์เป็นเวลาถัดไปจากข้อมูลที่เลือก
            forecast_start_date = selected_data.index.max() + pd.Timedelta(minutes=15)

            # พยากรณ์ 1 วันถัดไปจากช่วงวันที่เลือก
            forecasted_data = forecast_with_linear_regression(selected_data, forecast_start_date)

            # ตรวจสอบว่ามีการพยากรณ์หรือไม่
            if not forecasted_data.empty:
                # แสดงกราฟข้อมูลช่วงเวลาที่เลือกและกราฟการพยากรณ์
                st.plotly_chart(plot_selected_and_forecasted(selected_data, forecasted_data))

                # ตรวจสอบว่ามีข้อมูลจริงสำหรับช่วงเวลาที่พยากรณ์หรือไม่
                common_indices = forecasted_data.index.intersection(data.index)
                if not common_indices.empty:
                    actual_data = data.loc[common_indices]
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


