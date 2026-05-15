import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import joblib
import warnings
warnings.filterwarnings("ignore")


model = joblib.load("final_model.pkl")

st.title("Student Exam Score Prediction")

study_hours = st.slider("Study Hours per Day", 0.0, 12.0, 2.0)
attendance = st.slider("Attendance Percentage", 0.0, 100.0, 80.0)
mental_health = st.slider("Mental Health Rating (1-10)", 1, 10, 5)
sleep_hours = st.slider("Sleep Hours per Night", 0.0, 12.0, 7.0)
part_time_job = st.selectbox("Part-time Job", ["No", "Yes"])

ptj_encoded = 1 if part_time_job == "Yes" else 0

if st.button("Predict Exam Score"):

    input_data = np.array([[study_hours, attendance, mental_health, sleep_hours, ptj_encoded]])
    prediction = model.predict(input_data)[0]

    prediction = max(0, min(100, prediction))

    st.success(f"Predicted Exam Score: {prediction:.2f}")


    chart_data = pd.DataFrame({
        "Factors": [
            "Study Hours",
            "Attendance",
            "Mental Health",
            "Sleep Hours",
            "Predicted Score"
        ],
        "Values": [
            study_hours,
            attendance,
            mental_health,
            sleep_hours,
            prediction
        ]
    })


    st.subheader("📊 Performance Bar Graph")

    bar_fig = px.bar(
        chart_data,
        x="Factors",
        y="Values",
        text="Values",
        title="Student Performance Analysis"
    )

    st.plotly_chart(bar_fig, use_container_width=True)


    st.subheader("📈 Performance Trend Graph")

    line_fig = px.line(
        chart_data,
        x="Factors",
        y="Values",
        markers=True,
        title="Performance Trend Analysis"
    )

    st.plotly_chart(line_fig, use_container_width=True)

