import pickle
import streamlit as st

model = pickle.load(open('bodyfat.sav', 'rb'))

st.title('Estimasi Persentase Lemak Tubuh')

Density = st.number_input('Kepadatan')
BodyFat = st.number_input('Lemak Badan')
Age = st.number_input('Umur')
Weight = st.number_input('Berat')
Height = st.number_input('Tinggi')
Neck = st.number_input('Leher')
Chest = st.number_input('Dada')
Abdomen = st.number_input('Perut')
Hip = st.number_input('Pinggul')
Thigh = st.number_input('Paha')
Knee = st.number_input('Lutut')
Ankle = st.number_input('Pergelangan Kaki')
Biceps = st.number_input('Bisep')
Forearm = st.number_input('Lengan Bawah')

predict = None

if st.button('Estimasi Persentase Lemak'):
    # Prepare input data based on your model's requirements
    input_data = [[Density, BodyFat, Age, Weight, Height, Neck, Chest, Abdomen, Hip, Thigh, Knee, Ankle, Biceps, Forearm]]

    # Use your body fat estimation model to make predictions
    predict = model.predict(input_data)
    st.write(f'Estimasi Persentase Lemak Tubuh: {predict[0]} %')