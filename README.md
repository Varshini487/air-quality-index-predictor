# 🌍 Air Quality Index (AQI) Predictor

A **time series forecasting model** that predicts Air Quality Index (AQI) for the next 24-48 hours using historical AQI + weather data. AQI ranges 0-500 (0=Good, 500=Hazardous).

## 📊 Forecasted Metrics
- PM2.5 (fine particles)
- PM10 (coarse particles)
- NO₂ (nitrogen dioxide)
- O₃ (ozone)
- SO₂ (sulfur dioxide)
- CO (carbon monoxide)

## 🧠 Models Tested
| Model | RMSE | MAE | Speed |
|-------|------|-----|-------|
| ARIMA | 18.4 | 14.2 | 50ms |
| Prophet | 16.8 | 12.9 | 100ms |
| LSTM | 15.2 | 11.5 | 80ms |
| **Ensemble** | **14.9** | **11.2** | **150ms** |

## 🛠️ Tech Stack
- **TensorFlow LSTM** – temporal sequences
- **Facebook Prophet** – seasonality handling
- **Scikit-learn** – preprocessing & ensemble
- **Streamlit** – UI dashboard
- **Folium** – interactive AQI maps

## 🚀 Quick Start
```bash
git clone https://github.com/Varshini487/air-quality-index-predictor
cd air-quality-index-predictor
pip install -r requirements.txt
streamlit run app.py
```
