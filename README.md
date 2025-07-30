# End2End_Flight_Price_Predictor

This is an end-to-end Machine Learning project where I built a web app to predict flight ticket prices based on various travel-related features. Users can input details like airline, source/destination cities, departure time, class, stops, and get an instant prediction of the ticket price.

## 📌 Problem Statement

Flight prices can fluctuate based on various factors like timing, airline, travel class, and number of days left before the journey. This app helps users estimate the ticket price beforehand, using a trained ML model.

---

## 📂 Dataset

- 📍 **Source**: Kaggle  
- 🧾 **Features Used**:
  - Airline
  - Source City
  - Destination City
  - Departure Time
  - Arrival Time
  - Number of Stops
  - Class (Economy/Business)
  - Duration
  - Days Left before journey

---

## 🔎 Exploratory Data Analysis (EDA)

- Handled missing values
- Categorical encoding (One-Hot and Label)
- Removed outliers based on IQR
- Feature importance visualization
- Added interactive map for source-destination routes

---

## 🤖 Model Training

- Tried multiple regressors: **Linear Regression** and **Random Forest Regressor**
- Final model: **Random Forest Regressor**
- Trained on cleaned dataset
- Saved using `joblib`

---

## 🖥️ App Features

- Built with **Streamlit**
- Clean UI with dropdowns and sliders for user input
- Interactive **map visualization** of source-destination pairs using Plotly
- Instant price prediction upon submission

---

## 🚀 Deployment

- Deployed using **Streamlit Cloud**
- [🔗 Live App Link](#) *(Update once deployed)*

---

## 🛠️ Tech Stack

- Python
- Pandas, NumPy
- Scikit-learn
- Plotly, Matplotlib, Seaborn
- Streamlit

---

## 📁 Project Structure

```bash
├── app.py
├── model/
│   └── flight_price_model.pkl
├── data/
│   └── flight_data.csv
├── requirements.txt
└── README.md
