import streamlit as st
import pickle
import pandas as pd

# -----------------------------
# تحميل الموديل والـ encoder
# -----------------------------
try:
    model = pickle.load(open("model.pkl", "rb"))
    encoder = pickle.load(open("encoder.pkl", "rb"))
except FileNotFoundError:
    st.error("Model or encoder not found. تأكد من حفظهم أولاً.")
    st.stop()

# -----------------------------
# عنوان التطبيق
# -----------------------------
st.title("Car Evaluation App 🚗")

# -----------------------------
# اختيار القيم من المستخدم
# -----------------------------
buying = st.selectbox("Buying", ["low", "med", "high", "vhigh"])
maint = st.selectbox("Maintenance", ["low", "med", "high", "vhigh"])
doors = st.selectbox("Doors", ["2", "3", "4", "5more"])
persons = st.selectbox("Persons", ["2", "4", "more"])
lug_boot = st.selectbox("Luggage Boot", ["small", "med", "big"])
safety = st.selectbox("Safety", ["low", "med", "high"])

# -----------------------------
# زر التنبؤ
# -----------------------------
if st.button("Predict"):

    # تحويل البيانات لإطار بيانات
    input_df = pd.DataFrame([[buying, maint, doors, persons, lug_boot, safety]],
                            columns=["buying", "maint", "doors", "persons", "lug_boot", "safety"])

    # تشفير البيانات باستخدام الـ encoder
    encoded_input = encoder.transform(input_df)

    # التنبؤ باستخدام الموديل
    prediction = model.predict(encoded_input)

    # عرض النتيجة
    st.success(f"Prediction Result: {prediction[0]}")