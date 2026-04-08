# 🩺 Diabetes Risk Predictor

A machine learning-powered web application built with **Streamlit** that predicts diabetes risk based on key health parameters using a **Logistic Regression** model trained on the Pima Indians Diabetes Dataset.

---

## Live Link : https://diabetes-predictor-kvahbgzyuvzphchmnzdm5u.streamlit.app/


## 🖥️ Livr Demo Check



> Run locally using the steps below ↓

---

## 📸 Screenshots

| Input Panel | Prediction Result |
|---|---|
| ![Input](https://via.placeholder.com/400x250?text=Input+Panel) | ![Result](https://via.placeholder.com/400x250?text=Prediction+Result) |

---

## ✨ Features

- 🎚️ **Interactive sliders** for all health parameters
- 📊 **Real-time summary** metrics update as you adjust inputs
- 🔬 **Instant prediction** with probability percentage
- 🟢 / 🔴 **Color-coded results** — Low Risk or Elevated Risk
- 💡 **Key Risk Indicators** — color-coded health chips (High Glucose, Obese BMI, Family History, etc.)
- 🏷️ **BMI category auto-detection** — Underweight / Healthy / Overweight / Obese
- 🌙 **Dark glassmorphism UI** — clean, modern design

---

## 🧬 Input Parameters 

| Parameter | Description |
|---|---|
| Age | Patient age in years |
| Pregnancies | Number of pregnancies (0 if male) |
| BMI | Body Mass Index |
| Glucose Level | Plasma glucose (mg/dL) |
| Blood Pressure | Diastolic blood pressure (mmHg) |
| Insulin | 2-hour serum insulin (μU/mL) |
| Skin Thickness | Triceps skinfold thickness (mm) |
| Diabetes Pedigree Function | Family history score |

---

## 🤖 Model Details

| Property | Value |
|---|---|
| Algorithm | Logistic Regression |
| Dataset | Pima Indians Diabetes Dataset |
| Preprocessing | StandardScaler (feature normalization) |
| BMI Encoding | One-hot encoded (healthy / overweight / obese) |
| Output Classes | `0` — No Diabetes, `1` — Diabetes |
| sklearn version | 1.6.1 |

---

## 📁 Project Structure

```
diabetes-predictor/
├── app.py                  # Streamlit application
├── diabetes_model.pkl      # Trained Logistic Regression model
├── scaler.pkl              # StandardScaler for feature normalization
├── columns.pkl             # Feature column names
└── requirements.txt        # Python dependencies
```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- pip

### Installation

**1. Clone the repository**
```bash
git clone https://github.com/Pranjal007-AI/diabetes-predictor.git
cd diabetes-predictor
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Run the app**
```bash
streamlit run app.py
```

**4. Open in browser**
```
http://localhost:8501
```

---

## 📦 Requirements

```txt
streamlit
scikit-learn==1.6.1
joblib
numpy
```

> ⚠️ **Important:** Use `scikit-learn==1.6.1` exactly — the model was trained and saved with this version. Using a different version may cause loading errors.

---

## 🔮 How It Works

```
User Input (sliders)
        ↓
BMI Category Encoding (one-hot)
        ↓
Feature Array Assembly (11 features)
        ↓
StandardScaler Transform
        ↓
Logistic Regression Predict
        ↓
Probability Score + Risk Classification
```

---

## ⚕️ Disclaimer

> This application is for **educational and informational purposes only**. It does **not** replace professional medical advice, diagnosis, or treatment. Always consult a qualified healthcare provider for medical decisions.

---

## 🛠️ Built With

- [Streamlit](https://streamlit.io/) — Web app framework
- [scikit-learn](https://scikit-learn.org/) — Machine learning
- [NumPy](https://numpy.org/) — Numerical computing
- [Joblib](https://joblib.readthedocs.io/) — Model serialization

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

---

## 🙋‍♂️ Author

Made with ❤️ by **[Pranjal Parashar](https://github.com/Pranjal007-AI)**

⭐ If you found this useful, please star the repo!
