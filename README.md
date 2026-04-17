# 🌍 Climate Trend Analyzer

An end-to-end Data Science project that analyzes long-term climate patterns using temperature, rainfall, and humidity data.  
The system detects trends, identifies anomalies, and predicts future climate behavior through an interactive dashboard.

---

## 🚀 Features

- 📈 Temperature trend analysis using Linear Regression  
- ⚠️ Anomaly detection using Z-score method  
- 🌧 Rainfall pattern visualization  
- 🔮 Future temperature forecasting  
- 📊 Seasonal decomposition (STL)  
- 🖥 Interactive dashboard built with Streamlit  

---

## 🛠 Tech Stack

- **Language:** Python  
- **Libraries:** Pandas, NumPy  
- **Visualization:** Matplotlib, Seaborn, Plotly  
- **Machine Learning:** Scikit-learn  
- **Time Series:** Statsmodels  
- **Frontend:** Streamlit  

---

## 📂 Project Structure


Climate-Trend-Analyzer/
│
├── data/
├── src/
├── app/
├── outputs/
├── images/
├── main.py
├── requirements.txt
└── README.md


---

## ⚙️ Installation & Setup

```bash
git clone https://github.com/your-username/Climate-Trend-Analyzer.git
cd Climate-Trend-Analyzer
python -m venv climate_env
climate_env\Scripts\activate
pip install -r requirements.txt
▶️ Run the Project
Run full pipeline
python main.py
Run Streamlit Dashboard
streamlit run app/streamlit_app.py
## 📊 Project Outputs
📈 Temperature Trend

🌧 Rainfall Analysis

⚠️ Anomaly Detection

🔮 Forecast

🖥 Dashboard

## 📌 Key Insights
Detected long-term warming trend in temperature
Identified extreme anomalies (heatwaves & floods)
Observed seasonal rainfall distribution
Predicted future temperature rise

## 💡 Future Improvements
Use real-world datasets (NASA, NOAA)
Implement ARIMA / LSTM models
Add map-based visualization
Deploy as a live web app
