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
def plot_data(data, forecasted=None, label='ระดับน้ำ'):
    # ใช้ 'wl_up' ซึ่งเป็นคอลัมน์ระดับน้ำที่ถูกต้อง
    fig = px.line(data, x=data.index, y='wl_up', title=f'ระดับน้ำที่สถานี {label}', labels={'x': 'วันที่', 'wl_up': 'ระดับน้ำ (wl_up)'})
    if forecasted is not None and not forecasted.empty:
        fig.add_scatter(x=forecasted.index, y=forecasted['wl_up'], mode='lines', name='ค่าที่พยากรณ์', line=dict(color='red'))
    fig.update_layout(xaxis_title="วันที่", yaxis_title="ระดับน้ำ (wl_up)")
    return fig

# --------------------------------------------
# ฟังก์ชันสำหรับการพยากรณ์ด้วย Linear Regression
# --------------------------------------------
def forecast_with_linear_regression(up_data, target_data, forecast_start_date):
    # ใช้ข้อมูล 288 แถวสุดท้ายจาก up_data ในการเทรนโมเดล
    training_data = up_data.iloc[-288:].copy()

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
            # ดึงค่าจากข้อมูลจริงเท่านั้น
            if lag_time in up_data.index:
                lag_value = up_data.at[lag_time, 'wl_up']
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

# อัปโหลดไฟล์ CSV สำหรับสถานี up และสถานีที่ต้องการทำนาย
uploaded_up_file = st.file_uploader("เลือกไฟล์ CSV ของสถานีข้างบน (up)", type="csv")
uploaded_target_file = st.file_uploader("เลือกไฟล์ CSV ของสถานีที่ต้องการทำนาย", type="csv")

if uploaded_up_file is not None and uploaded_target_file is not None:
    # อ่านข้อมูลจากไฟล์ CSV
    up_data = pd.read_csv(uploaded_up_file)
    target_data = pd.read_csv(uploaded_target_file)

    # แปลง column datetime เป็น datetime และตั้งให้เป็น index
    for data in [up_data, target_data]:
        data['datetime'] = pd.to_datetime(data['datetime'])
        data['datetime'] = data['datetime'].dt.tz_localize(None)
        data.set_index('datetime', inplace=True)

    # แสดงกราฟข้อมูล
    st.subheader('กราฟข้อมูลระดับน้ำ')
    st.plotly_chart(plot_data(up_data, label='สถานีข้างบน (up)'))
    st.plotly_chart(plot_data(target_data, label='สถานีที่ต้องการทำนาย'))

    # ส่วนการพยากรณ์ใช้ target_data เพื่อเทรนและทำนาย
    # ให้ผู้ใช้เลือกช่วงวันที่ที่สนใจ
    st.subheader("เลือกช่วงวันสำหรับพยากรณ์ในอีก 1 วันข้างหน้า")
    start_date = st.date_input("เลือกวันเริ่มต้น", target_data.index.min().date())
    end_date = st.date_input("เลือกวันสิ้นสุด", target_data.index.max().date())

    if start_date > end_date:
        st.error("วันเริ่มต้นต้องไม่เกินวันสิ้นสุด")
    else:
        if st.button("พยากรณ์"):
            # เลือกข้อมูลช่วงวันที่ที่สนใจ
            selected_data = target_data[(target_data.index.date >= start_date) & (target_data.index.date <= end_date)].copy()

            # ตรวจสอบว่ามีข้อมูลเพียงพอหรือไม่
            if selected_data.empty:
                st.error("ไม่มีข้อมูลในช่วงวันที่ที่เลือก กรุณาเลือกวันที่ใหม่")
            else:
                # กำหนดวันที่เริ่มพยากรณ์เป็นเวลาถัดไปจากข้อมูลที่เลือก
                forecast_start_date = selected_data.index.max() + pd.Timedelta(minutes=15)

                # พยากรณ์
                forecasted_data = forecast_with_linear_regression(up_data, target_data, forecast_start_date)

                # ตรวจสอบว่ามีการพยากรณ์หรือไม่
                if not forecasted_data.empty:
                    # แสดงกราฟข้อมูลพร้อมการพยากรณ์
                    st.subheader('กราฟข้อมูลพร้อมการพยากรณ์')
                    st.plotly_chart(plot_data(selected_data, forecasted_data, label='สถานีที่ต้องการทำนาย'))

                    # ตรวจสอบว่ามีข้อมูลจริงสำหรับช่วงเวลาที่พยากรณ์หรือไม่
                    common_indices = forecasted_data.index.intersection(target_data.index)
                    if not common_indices.empty:
                        actual_data = target_data.loc[common_indices]
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


