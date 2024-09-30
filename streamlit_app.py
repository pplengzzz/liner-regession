import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

# ตั้งค่าหน้าเว็บ Streamlit
st.set_page_config(page_title='การพยากรณ์ระดับน้ำด้วย Linear Regression', page_icon=':ocean:')

# ชื่อของแอป
st.title("การพยากรณ์ระดับน้ำด้วย Linear Regression")

# -------------------------------
# ฟังก์ชันสำหรับการแสดงกราฟข้อมูล
# -------------------------------
def plot_data(data, forecasted=None):
    fig = px.line(data, x=data.index, y='wl_up', title='ระดับน้ำตามเวลา', labels={'x': 'วันที่', 'wl_up': 'ระดับน้ำ (wl_up)'})
    if forecasted is not None and not forecasted.empty:
        fig.add_scatter(x=forecasted.index, y=forecasted['wl_up'], mode='lines', name='ค่าที่พยากรณ์', line=dict(color='red'))
    fig.update_layout(xaxis_title="วันที่", yaxis_title="ระดับน้ำ (wl_up)")
    return fig

# --------------------------------------------
# ฟังก์ชันสำหรับการพยากรณ์ด้วย Linear Regression
# --------------------------------------------
def forecast_with_linear_regression(data, forecast_start_date):
    # เติมค่า missing values ด้วยการ interpolate
    data['wl_up'].interpolate(method='time', inplace=True)

    # ใช้ข้อมูลย้อนหลัง 3 วันในการเทรนโมเดล
    training_data_end = forecast_start_date - pd.Timedelta(minutes=15)
    training_data_start = training_data_end - pd.Timedelta(days=3) + pd.Timedelta(minutes=15)

    # ตรวจสอบว่ามีข้อมูลเพียงพอหรือไม่
    if training_data_start < data.index.min():
        st.error("ไม่สามารถพยากรณ์ได้เนื่องจากข้อมูลสำหรับการเทรนไม่เพียงพอ")
        return pd.DataFrame()

    # สร้างชุดข้อมูลสำหรับการเทรน
    training_data = data.loc[training_data_start:training_data_end].copy()

    # สร้างฟีเจอร์ lag
    lags = [1, 4, 96, 192]  # lag 15 นาที, 1 ชั่วโมง, 1 วัน, 2 วัน
    for lag in lags:
        training_data[f'lag_{lag}'] = training_data['wl_up'].shift(lag)

    # ลบแถวที่มีค่า NaN ในฟีเจอร์ lag
    training_data.dropna(inplace=True)

    # ตรวจสอบว่ามีข้อมูลเพียงพอหลังจากสร้างฟีเจอร์ lag
    if training_data.empty:
        st.error("ไม่สามารถพยากรณ์ได้เนื่องจากข้อมูลไม่เพียงพอหลังจากสร้างฟีเจอร์ lag")
        return pd.DataFrame()

    # กำหนดฟีเจอร์และตัวแปรเป้าหมาย
    feature_cols = [f'lag_{lag}' for lag in lags]
    X_train = training_data[feature_cols]
    y_train = training_data['wl_up']

    # เทรนโมเดล Linear Regression
    model = LinearRegression()
    model.fit(X_train, y_train)

    # สร้าง DataFrame สำหรับการพยากรณ์
    forecast_periods = 96  # พยากรณ์ 1 วัน (96 ช่วงเวลา 15 นาที)
    forecast_index = pd.date_range(start=forecast_start_date, periods=forecast_periods, freq='15T')
    forecasted_data = pd.DataFrame(index=forecast_index)
    forecasted_data['wl_up'] = np.nan

    # การพยากรณ์
    for idx in forecasted_data.index:
        lag_features = {}
        for lag in lags:
            lag_time = idx - pd.Timedelta(minutes=15 * lag)
            # ดึงค่าจากข้อมูลจริงหรือค่าที่พยากรณ์ไว้ก่อนหน้า
            if lag_time in data.index:
                lag_value = data.at[lag_time, 'wl_up']
            elif lag_time in forecasted_data.index and not pd.isnull(forecasted_data.at[lag_time, 'wl_up']):
                lag_value = forecasted_data.at[lag_time, 'wl_up']
            else:
                lag_value = np.nan
            lag_features[f'lag_{lag}'] = lag_value

        # ถ้ามีค่า lag ที่หายไป ให้ข้ามการพยากรณ์
        if np.any(pd.isnull(list(lag_features.values()))):
            continue

        # สร้าง DataFrame สำหรับฟีเจอร์ที่จะใช้ในการพยากรณ์
        X_pred = pd.DataFrame([lag_features])

        # พยากรณ์ค่า
        forecast_value = model.predict(X_pred)[0]
        forecasted_data.at[idx, 'wl_up'] = forecast_value

    # ลบแถวที่ไม่มีการพยากรณ์
    forecasted_data.dropna(inplace=True)

    return forecasted_data

# -------------------------------
# ส่วนหลักของโปรแกรม
# -------------------------------

# อัปโหลดไฟล์ CSV ข้อมูลจริง
uploaded_file = st.file_uploader("เลือกไฟล์ CSV ข้อมูลระดับน้ำ", type="csv")

if uploaded_file is not None:
    # อ่านข้อมูลจากไฟล์ CSV
    data = pd.read_csv(uploaded_file)
    data['datetime'] = pd.to_datetime(data['datetime'])
    data['datetime'] = data['datetime'].dt.tz_localize(None)
    data.set_index('datetime', inplace=True)

    # ตัดข้อมูลที่มีค่า wl_up น้อยกว่า 100 ออก
    data = data[data['wl_up'] >= 100]

    # เติมค่า missing values ใน wl_up
    data['wl_up'].interpolate(method='time', inplace=True)

    # แสดงกราฟข้อมูล
    st.subheader('กราฟข้อมูลระดับน้ำ')
    st.plotly_chart(plot_data(data))

    # ให้ผู้ใช้เลือกช่วงวันที่ที่สนใจ
    st.subheader("เลือกช่วงวันที่ที่สนใจ")
    start_date = st.date_input("เลือกวันเริ่มต้น", data.index.min().date())
    end_date = st.date_input("เลือกวันสิ้นสุด", data.index.max().date())

    if start_date > end_date:
        st.error("วันเริ่มต้นต้องไม่เกินวันสิ้นสุด")
    else:
        if st.button("พยากรณ์"):
            # เลือกข้อมูลช่วงวันที่ที่สนใจ
            selected_data = data[(data.index.date >= start_date) & (data.index.date <= end_date)].copy()

            # ตรวจสอบว่ามีข้อมูลเพียงพอหรือไม่
            if selected_data.empty:
                st.error("ไม่มีข้อมูลในช่วงวันที่ที่เลือก กรุณาเลือกวันที่ใหม่")
            else:
                # กำหนดวันที่เริ่มพยากรณ์เป็นเวลาถัดไปจากข้อมูลที่เลือก
                forecast_start_date = selected_data.index.max() + pd.Timedelta(minutes=15)

                # พยากรณ์
                forecasted_data = forecast_with_linear_regression(data, forecast_start_date)

                # ตรวจสอบว่ามีการพยากรณ์หรือไม่
                if not forecasted_data.empty:
                    # แสดงกราฟข้อมูลพร้อมการพยากรณ์
                    st.subheader('กราฟข้อมูลพร้อมการพยากรณ์')
                    st.plotly_chart(plot_data(selected_data, forecasted_data))

                    # ตรวจสอบว่ามีข้อมูลจริงสำหรับช่วงเวลาที่พยากรณ์หรือไม่
                    common_indices = forecasted_data.index.intersection(data.index)
                    if not common_indices.empty:
                        actual_data = data.loc[common_indices]
                        y_true = actual_data['wl_up']
                        y_pred = forecasted_data['wl_up'].loc[common_indices]
                        mae = mean_absolute_error(y_true, y_pred)
                        rmse = mean_squared_error(y_true, y_pred, squared=False)

                        # แสดงตารางที่มี datetime, ค่าจริง, ค่าพยากรณ์
                        st.subheader('ตารางข้อมูลเปรียบเทียบ')
                        comparison_table = pd.DataFrame({
                            'datetime': forecasted_data.index,
                            'ค่าจริง (ถ้ามี)': y_true.values,
                            'ค่าที่พยากรณ์': y_pred.values
                        })
                        st.dataframe(comparison_table)

                        # แสดงค่า MAE และ RMSE
                        st.write(f"Mean Absolute Error (MAE): {mae:.2f}")
                        st.write(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
                    else:
                        st.info("ไม่มีข้อมูลจริงสำหรับช่วงเวลาที่พยากรณ์ ไม่สามารถคำนวณค่า MAE และ RMSE ได้")
                else:
                    st.error("ไม่สามารถพยากรณ์ได้เนื่องจากข้อมูลไม่เพียงพอ")



