import pickle
import streamlit as st

model = pickle.load(open('1-student-performance-prediction.sav', 'rb'))

st.title('Estimasi Kinerja Siswa dalam Matematika')

study_hours = st.number_input('Jam Belajar per Minggu')
absences = st.number_input('Jumlah Absensi')
previous_scores = st.number_input('Nilai Ujian Sebelumnya')

predict = ' '

if st.button('Estimasi Kinerja'):
  
    prediction = model.predict(
        [[study_hours, absences, previous_scores]]
    )
    st.write('Estimasi Kinerja Siswa dalam Matematika: ', prediction)