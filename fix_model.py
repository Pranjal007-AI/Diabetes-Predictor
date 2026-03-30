import joblib

model = joblib.load("diabetes_model.pkl")
scaler = joblib.load("scaler.pkl")

joblib.dump(model, "diabetes_model.pkl")
joblib.dump(scaler, "scaler.pkl")