from flask import Flask, request, render_template, jsonify, redirect, url_for
import joblib
import pandas as pd
import os
import io
import base64
import matplotlib.pyplot as plt

app = Flask(__name__)

# load previously trained model pipeline and label encoder
MODEL_PATH = os.path.join('model', 'saved_model.pkl')
if os.path.exists(MODEL_PATH):
    model_data = joblib.load(MODEL_PATH)
    model = model_data['pipeline']
    label_encoder = model_data['label_encoder']
else:
    model = None
    label_encoder = None


def get_recommendations(risk_label, behavior=None):
    """Return a list of counselling recommendations based on risk and
    behavior.
    """
    rec = []
    if risk_label == 'High':
        rec.append('Academic counselling')
        rec.append('Attendance monitoring')
        rec.append('Mentorship programs')
        if behavior and behavior.lower() == 'poor':
            rec.append('Psychological support')
    elif risk_label == 'Medium':
        rec.append('Periodic check-ins')
        rec.append('Encourage study groups')
    else:
        rec.append('Keep up the good work!')
    return rec


def preprocess_input(data_dict):
    """Convert the incoming form data into a pandas DataFrame for the
    pipeline to consume.  Numeric fields are coerced to floats.
    """
    df = pd.DataFrame([data_dict])
    # ensure numeric types
    for col in ['grades', 'gpa', 'attendance']:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)
    return df

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # gather input from form
    fields = ['grades', 'gpa', 'attendance', 'behavior', 'socio_eco']
    input_data = {f: request.form.get(f, '') for f in fields}

    df = preprocess_input(input_data)

    if model is None:
        return "Model not found. Please train the model first."

    pred_encoded = model.predict(df)[0]
    proba = model.predict_proba(df)[0]
    risk_label = label_encoder.inverse_transform([pred_encoded])[0]
    prob_score = round(float(max(proba)), 3)

    recommendations = get_recommendations(risk_label, input_data.get('behavior'))

    return render_template(
        'result.html',
        risk=risk_label,
        probability=prob_score,
        recommendations=recommendations,
        input_data=input_data
    )

@app.route('/recommend', methods=['POST'])
def recommend():
    """API endpoint that returns recommendations as JSON. Useful for AJAX
    calls from the frontend.
    """
    data = request.json or {}
    risk = data.get('risk')
    behavior = data.get('behavior')
    recs = get_recommendations(risk, behavior)
    return jsonify({'recommendations': recs})


@app.route('/dashboard')
def dashboard():
    """Simple dashboard showing a bar chart of risk distribution from the
    current dataset. The plot is embedded as a base64-encoded PNG.
    """
    data_path = os.path.join('dataset', 'student_data.csv')
    if not os.path.exists(data_path):
        return "No dataset available to display dashboard."
    df = pd.read_csv(data_path)
    img = io.BytesIO()
    plt.figure(figsize=(6,4))
    if 'risk' in df.columns:
        df['risk'].value_counts().plot(kind='bar', color=['green','orange','red'])
        plt.title('Risk Distribution')
        plt.tight_layout()
        plt.savefig(img, format='png')
        img.seek(0)
        plot_url = base64.b64encode(img.getvalue()).decode()
        return render_template('dashboard.html', plot_url=plot_url)
    else:
        return "Dataset does not contain 'risk' column."

@app.route('/retrain', methods=['GET', 'POST'])
def retrain():
    """Allow an admin to retrain the model by uploading a new CSV. This
    demonstrates a model-retraining feature. In a real app, authentication
    would be required.
    """
    message = ''
    if request.method == 'POST':
        file = request.files.get('file')
        if file and file.filename.endswith('.csv'):
            path = os.path.join('dataset', 'student_data.csv')
            file.save(path)
            # re-run training script
            from model.train_model import load_data, train_and_select_model
            df = load_data(path)
            train_and_select_model(df)
            message = 'Model retrained with uploaded data.'
        else:
            message = 'Please upload a CSV file.'
    return render_template('retrain.html', message=message)

if __name__ == '__main__':
    app.run(debug=True)
