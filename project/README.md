# Dropout & Counselling Prediction System

This repository contains a simple full-stack application that predicts
a student's dropout risk and recommends counselling actions using a
machine learning model.

## Structure

```
project/
├── app.py                 # Flask backend
├── model/
│   ├── train_model.py     # script to load data and train the model
│   └── saved_model.pkl    # serialized pipeline (created after training)
├── templates/             # HTML templates for frontend
│   ├── index.html
│   ├── result.html
│   ├── retrain.html
│   └── dashboard.html
├── static/
│   └── css/style.css
├── dataset/
│   └── student_data.csv   # example data
└── README.md
```

## Requirements

- Python 3.8+
- Flask
- pandas
- numpy
- scikit-learn
- matplotlib
- joblib

Install with:

```sh
pip install -r requirements.txt
```

*(or manually install the packages)*

## Usage

1. **Train the model**

   ```sh
   python model/train_model.py
   ```

   This will read `dataset/student_data.csv` (or generate synthetic data)
   and save a trained model to `model/saved_model.pkl`.

2. **Run the server**

   ```sh
   python app.py
   ```

   Visit `http://localhost:5000` to see the input form.

3. **Predict**

   Fill in student details and submit; the app displays risk category,
   probability, and counselling recommendations.

4. **Admin Features**
   - `/dashboard` shows a simple bar chart of risk distribution.
   - `/retrain` lets you upload a new CSV and retrain the model.

## Data Format

CSV file (`student_data.csv`) should have the following columns:

```
grades,gpa,attendance,behavior,socio_eco,risk
```

`risk` can be `Low`, `Medium`, or `High`.

## Extending

- Add authentication for admin pages.
- Enhance preprocessing and feature engineering.
- Replace synthetic data with real student records.
- Add more models or hyperparameter tuning.

---

*Code is commented to explain each part of the system.*