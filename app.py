from flask import Flask, request, render_template
import joblib
import pandas as pd

app = Flask(__name__)

# Load trained model, scaler, and encoders
model = joblib.load('depression_model.pkl')
scaler = joblib.load('scaler.pkl')
encoders = joblib.load('label_encoders.pkl')

@app.route('/', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        input_data = request.form.to_dict()
        df = pd.DataFrame([input_data])
        # Drop identifier columns if present—safe for all cases
        df = df.drop(['Name', 'ID'], axis=1, errors='ignore')
        # Encode only columns present in input
        for col, le in encoders.items():
            if col in df.columns:
                df[col] = le.transform(df[col])
        X = scaler.transform(df)
        prediction = model.predict(X)[0]
        result = "High Risk" if prediction == 1 else "Low Risk"
        return render_template("result.html", result=result)
    return render_template("form.html")

if __name__ == "__main__":
    app.run(debug=True)
