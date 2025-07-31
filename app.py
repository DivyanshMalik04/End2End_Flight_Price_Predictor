import streamlit as st
import numpy as np
import pandas as pd
import joblib
import pydeck as pdk
import datetime
import os
import gdown


# Page config
st.set_page_config(page_title="Flight Price Predictor", page_icon="✈️", layout="centered")


file_id = '1L89dc0hzPiLciEQaI5qJ2Y6OmPdFwO7b'
url = f'https://drive.google.com/uc?id={file_id}'
model_filename = 'rfr_model.joblib'

@st.cache_resource
def load_model_from_drive():
    if not os.path.exists(model_filename):
        gdown.download(url, model_filename, quiet=False)
        model = joblib.load(model_filename)
        return model

model = load_model_from_drive()

#def load_model():
    #return joblib.load("rfr_model.joblib")  # Load from local file

#model = load_model()


# Title & Description
st.title("✈️ Flight Price Prediction App")
st.markdown("Predict flight ticket prices based on route, airline, time, class, and more — powered by Machine Learning.")
st.markdown("---")

# ------------------ INPUT SECTION ------------------

# 2-column layout for user inputs
col1, col2 = st.columns(2)

with col1:
    airline = st.selectbox("🛫 Airline", ['IndiGo', 'Air India', 'Vistara', 'SpiceJet', 'GO_FIRST', 'Akasa_Air'])
    source_city = st.selectbox("🌆 Source City", ['Delhi', 'Mumbai', 'Kolkata', 'Hyderabad', 'Bangalore', 'Chennai'])
    departure_time = st.selectbox("🕔 Departure Time", ['Early Morning', 'Morning', 'Afternoon', 'Evening', 'Night', 'Late Night'])
    stops = st.selectbox("🛑 Stops", ['zero', 'one', 'two_or_more'])

with col2:
    destination_city = st.selectbox("🏙 Destination City", ['Delhi', 'Mumbai', 'Kolkata', 'Hyderabad', 'Bangalore', 'Chennai'])
    arrival_time = st.selectbox("🕓 Arrival Time", ['Early Morning', 'Morning', 'Afternoon', 'Evening', 'Night', 'Late Night'])
    travel_class = st.selectbox("💺 Travel Class", ['Economy', 'Business'])
    duration = st.number_input("⏱ Duration (in hours)", min_value=0.0, max_value=1440.0, step=1.0)

# Days left with date picker
departure_date = st.date_input("📅 Departure Date", min_value=datetime.date.today())
days_left = (departure_date - datetime.date.today()).days

# ------------------ ENCODING ------------------

stops_mapping = {'zero': 0, 'one': 1, 'two_or_more': 2}
class_mapping = {'Economy': 0, 'Business': 1}
time_mapping = {
    'Early Morning': 0, 'Morning': 1, 'Afternoon': 2,
    'Evening': 3, 'Night': 4, 'Late Night': 5
}

stops_encoded = stops_mapping[stops]
class_encoded = class_mapping[travel_class]
departure_encoded = time_mapping[departure_time]
arrival_encoded = time_mapping[arrival_time]

# One-hot encoding for airline, source, destination
airline_dummies = {
    'airline_Air_India': 1 if airline == 'Air India' else 0,
    'airline_GO_FIRST': 1 if airline == 'GO_FIRST' else 0,
    'airline_IndiGo': 1 if airline == 'IndiGo' else 0,
    'airline_SpiceJet': 1 if airline == 'SpiceJet' else 0,
    'airline_Vistara': 1 if airline == 'Vistara' else 0
}
if airline == 'Akasa_Air':
    for key in airline_dummies:
        airline_dummies[key] = 0

source_dummies = {
    'source_city_Chennai': 1 if source_city == 'Chennai' else 0,
    'source_city_Delhi': 1 if source_city == 'Delhi' else 0,
    'source_city_Hyderabad': 1 if source_city == 'Hyderabad' else 0,
    'source_city_Kolkata': 1 if source_city == 'Kolkata' else 0,
    'source_city_Mumbai': 1 if source_city == 'Mumbai' else 0
}
if source_city == 'Bangalore':
    for key in source_dummies:
        source_dummies[key] = 0

destination_dummies = {
    'destination_city_Chennai': 1 if destination_city == 'Chennai' else 0,
    'destination_city_Delhi': 1 if destination_city == 'Delhi' else 0,
    'destination_city_Hyderabad': 1 if destination_city == 'Hyderabad' else 0,
    'destination_city_Kolkata': 1 if destination_city == 'Kolkata' else 0,
    'destination_city_Mumbai': 1 if destination_city == 'Mumbai' else 0
}
if destination_city == 'Bangalore':
    for key in destination_dummies:
        destination_dummies[key] = 0

final_input = [
    departure_encoded, stops_encoded, arrival_encoded, class_encoded, duration, days_left,
    airline_dummies['airline_Air_India'],
    airline_dummies['airline_GO_FIRST'],
    airline_dummies['airline_IndiGo'],
    airline_dummies['airline_SpiceJet'],
    airline_dummies['airline_Vistara'],
    source_dummies['source_city_Chennai'],
    source_dummies['source_city_Delhi'],
    source_dummies['source_city_Hyderabad'],
    source_dummies['source_city_Kolkata'],
    source_dummies['source_city_Mumbai'],
    destination_dummies['destination_city_Chennai'],
    destination_dummies['destination_city_Delhi'],
    destination_dummies['destination_city_Hyderabad'],
    destination_dummies['destination_city_Kolkata'],
    destination_dummies['destination_city_Mumbai']
]

# ------------------ PREDICTION ------------------

if st.button("🎯 Predict Price"):
    predicted_log_price = model.predict([final_input])[0]
    predicted_price = np.expm1(predicted_log_price)
    st.success(f"💰 Estimated Flight Price: ₹{int(predicted_price):,}")

# ------------------ MAP SECTION ------------------

st.markdown("---")
st.markdown("### 🗺 Source → Destination Map")

city_coordinates = {
    'Delhi': {'lat': 28.6139, 'lon': 77.2090},
    'Mumbai': {'lat': 19.0760, 'lon': 72.8777},
    'Bangalore': {'lat': 12.9716, 'lon': 77.5946},
    'Hyderabad': {'lat': 17.3850, 'lon': 78.4867},
    'Kolkata': {'lat': 22.5726, 'lon': 88.3639},
    'Chennai': {'lat': 13.0827, 'lon': 80.2707}
}

src_coords = city_coordinates[source_city]
dst_coords = city_coordinates[destination_city]

df = pd.DataFrame([
    {"name": "Source", "lat": src_coords['lat'], "lon": src_coords['lon']},
    {"name": "Destination", "lat": dst_coords['lat'], "lon": dst_coords['lon']}
])

point_layer = pdk.Layer(
    "ScatterplotLayer",
    data=df,
    get_position='[lon, lat]',
    get_color='[200, 30, 0, 160]',
    get_radius=50000,
)

line_layer = pdk.Layer(
    "LineLayer",
    data=pd.DataFrame([{
        "from_lon": src_coords['lon'], "from_lat": src_coords['lat'],
        "to_lon": dst_coords['lon'], "to_lat": dst_coords['lat']
    }]),
    get_source_position='[from_lon, from_lat]',
    get_target_position='[to_lon, to_lat]',
    get_width=5,
    get_color=[0, 0, 255],
    pickable=True
)

view_state = pdk.ViewState(
    latitude=(src_coords['lat'] + dst_coords['lat']) / 2,
    longitude=(src_coords['lon'] + dst_coords['lon']) / 2,
    zoom=4,
    pitch=0
)

st.pydeck_chart(pdk.Deck(
    map_style='https://basemaps.cartocdn.com/gl/positron-gl-style/style.json',
    initial_view_state=view_state,
    layers=[point_layer, line_layer]
))

# ------------------ FOOTER ------------------

st.markdown("---")
st.caption("📍 Made with Streamlit · Developed by Divyansh Malik")




