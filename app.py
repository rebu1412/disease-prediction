import streamlit as st
import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler
import google.generativeai as genai

# Cấu hình API key cho Gemini hoặc mô hình ngôn ngữ tương tự
genai.configure(api_key='your-api-key')

# Tải mô hình Random Forest đã lưu
loaded_rf_model = joblib.load('rf_model.pkl')

# Tiêu đề ứng dụng
st.title("Dự đoán bệnh")

# Form nhập liệu
with st.form(key='health_form'):
    glucose = st.number_input("Mức đường huyết (Glucose):", step=0.1)
    cholesterol = st.number_input("Mức Cholesterol:", step=0.1)
    hemoglobin = st.number_input("Mức Hemoglobin:", step=0.1)
    platelets = st.number_input("Mức Tiểu cầu (Platelets):", step=0.1)
    white_blood_cells = st.number_input("Mức Bạch cầu (White Blood Cells):", step=0.1)
    red_blood_cells = st.number_input("Mức Hồng cầu (Red Blood Cells):", step=0.1)
    hematocrit = st.number_input("Mức Hematocrit:", step=0.1)
    mean_corpuscular_volume = st.number_input("Thể tích trung bình hồng cầu (Mean Corpuscular Volume):", step=0.1)
    mean_corpuscular_hemoglobin = st.number_input("Hemoglobin trung bình trong hồng cầu (Mean Corpuscular Hemoglobin):", step=0.1)
    mean_corpuscular_hemoglobin_concentration = st.number_input("Nồng độ Hemoglobin trung bình trong hồng cầu (Mean Corpuscular Hemoglobin Concentration):", step=0.1)
    insulin = st.number_input("Mức Insulin:", step=0.1)
    bmi = st.number_input("Chỉ số khối cơ thể (BMI):", step=0.1)
    systolic_blood_pressure = st.number_input("Huyết áp tâm thu (Systolic Blood Pressure):", step=0.1)
    diastolic_blood_pressure = st.number_input("Huyết áp tâm trương (Diastolic Blood Pressure):", step=0.1)
    triglycerides = st.number_input("Mức Triglycerides:", step=0.1)
    hba1c = st.number_input("Mức độ HbA1c:", step=0.1)
    ldl_cholesterol = st.number_input("Mức LDL Cholesterol:", step=0.1)
    hdl_cholesterol = st.number_input("Mức HDL Cholesterol:", step=0.1)
    alt = st.number_input("Mức ALT:", step=0.1)
    ast = st.number_input("Mức AST:", step=0.1)
    heart_rate = st.number_input("Nhịp tim (Heart Rate):", step=0.1)
    creatinine = st.number_input("Mức Creatinine:", step=0.1)
    troponin = st.number_input("Mức Troponin:", step=0.1)
    c_reactive_protein = st.number_input("Mức C-reactive Protein:", step=0.1)
    
    submit_button = st.form_submit_button("Dự đoán")

# Xử lý khi nhấn nút "Dự đoán"
if submit_button:
    # Tạo DataFrame từ thông tin người dùng
    input_data = pd.DataFrame({
        'Glucose': [glucose],
        'Cholesterol': [cholesterol],
        'Hemoglobin': [hemoglobin],
        'Platelets': [platelets],
        'White Blood Cells': [white_blood_cells],
        'Red Blood Cells': [red_blood_cells],
        'Hematocrit': [hematocrit],
        'Mean Corpuscular Volume': [mean_corpuscular_volume],
        'Mean Corpuscular Hemoglobin': [mean_corpuscular_hemoglobin],
        'Mean Corpuscular Hemoglobin Concentration': [mean_corpuscular_hemoglobin_concentration],
        'Insulin': [insulin],
        'BMI': [bmi],
        'Systolic Blood Pressure': [systolic_blood_pressure],
        'Diastolic Blood Pressure': [diastolic_blood_pressure],
        'Triglycerides': [triglycerides],
        'HbA1c': [hba1c],
        'LDL Cholesterol': [ldl_cholesterol],
        'HDL Cholesterol': [hdl_cholesterol],
        'ALT': [alt],
        'AST': [ast],
        'Heart Rate': [heart_rate],
        'Creatinine': [creatinine],
        'Troponin': [troponin],
        'C-reactive Protein': [c_reactive_protein]
    })

    # Chuẩn hóa dữ liệu mới
    scaler_new = StandardScaler()
    input_scaled = scaler_new.fit_transform(input_data)

    # Dự đoán trên dữ liệu mới
    y_pred_rf = loaded_rf_model.predict(input_scaled)

    # Tạo một dictionary ánh xạ từ số mã hóa sang tên bệnh
    label_mapping = {
        0: "Healthy",
        1: "Diabetes",
        2: "Anemia",
        3: "Thalasse"
    }

    # Chuyển đổi mã hóa thành tên bệnh
    predicted_disease = label_mapping[y_pred_rf[0]]

    # Hiển thị kết quả dự đoán
    st.write(f"## Dự đoán: Bạn có khả năng mắc bệnh {predicted_disease}.")

# Tạo ô nhập liệu cho câu hỏi bên ngoài form
question = st.text_input("Bạn có câu hỏi nào về tình trạng sức khỏe của mình không?")

# Xử lý câu hỏi
if question:
    # Tạo prompt cho mô hình ngôn ngữ
    prompt = (
        "Bạn là một nhân viên y tế chuyên phân tích và đưa lời khuyên. "
        f"Bệnh nhân đã nhập các thông số sau:\n"
        f"Mức đường huyết (Glucose) (mg/dL): {glucose}\n"
        f"Mức Cholesterol (mg/dL): {cholesterol}\n"
        f"Mức Hemoglobin (g/dL): {hemoglobin}\n"
        f"Mức Tiểu cầu (Platelets) (10^3/µL): {platelets}\n"
        f"Mức Bạch cầu (White Blood Cells) (10^3/µL): {white_blood_cells}\n"
        f"Mức Hồng cầu (Red Blood Cells) (10^6/µL): {red_blood_cells}\n"
        f"Mức Hematocrit (%): {hematocrit}\n"
        f"Thể tích trung bình hồng cầu (Mean Corpuscular Volume) (fL): {mean_corpuscular_volume}\n"
        f"Hemoglobin trung bình trong hồng cầu (Mean Corpuscular Hemoglobin) (pg): {mean_corpuscular_hemoglobin}\n"
        f"Nồng độ Hemoglobin trung bình trong hồng cầu (Mean Corpuscular Hemoglobin Concentration) (g/dL): {mean_corpuscular_hemoglobin_concentration}\n"
        f"Mức Insulin (µU/mL): {insulin}\n"
        f"Chỉ số khối cơ thể (BMI) (kg/m²): {bmi}\n"
        f"Huyết áp tâm thu (Systolic Blood Pressure) (mmHg): {systolic_blood_pressure}\n"
        f"Huyết áp tâm trương (Diastolic Blood Pressure) (mmHg): {diastolic_blood_pressure}\n"
        f"Mức Triglycerides (mg/dL): {triglycerides}\n"
        f"Mức độ HbA1c (%): {hba1c}\n"
        f"Mức LDL Cholesterol (mg/dL): {ldl_cholesterol}\n"
        f"Mức HDL Cholesterol (mg/dL): {hdl_cholesterol}\n"
        f"Mức ALT (U/L): {alt}\n"
        f"Mức AST (U/L): {ast}\n"
        f"Nhịp tim (Heart Rate) (bpm): {heart_rate}\n"
        f"Mức Creatinine (mg/dL): {creatinine}\n"
        f"Mức Troponin (ng/mL): {troponin}\n"
        f"Mức C-reactive Protein (mg/L): {c_reactive_protein}\n\n"
        f"Câu hỏi: {question}\n\n"
        "Hãy cung cấp câu trả lời cho câu hỏi này."
    )
    # Gửi yêu cầu đến mô hình ngôn ngữ
    try:
        llm = genai.GenerativeModel('gemini-1.5-flash')
        response = llm.generate_content(prompt)
        
        # Kiểm tra xem response có kết quả không
        if hasattr(response, 'text') and response.text:
            st.write("Trả lời từ Chatbot:")
            st.write(response.text)
        else:
            st.write("Không có câu trả lời nào được trả về từ mô hình.")
    except Exception as e:
        st.error(f"Error connecting to API: {e}")
