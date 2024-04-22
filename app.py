from flask import Flask, request, jsonify, render_template
from classifier import *

app = Flask(__name__)

# Load the HTML file
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict_diabetes():
    data = request.get_json()
    features = [data.get(key) for key in ['pregnancies', 'glucose', 'blood_pressure', 'skin_thickness', 'insulin', 'bmi', 'diabetes_pedigree_function', 'age']]

    # Check if any of the fields is empty
    if None in features:
        return jsonify({'error': 'Please provide all fields.'}), 400

    # If all fields are filled, proceed with classification
    try:
        pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree_function, age = map(float, features)
    except ValueError:
        return jsonify({'error': 'Invalid input. Please provide numerical values.'}), 400

    result = ensemble_prediction([pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree_function, age])

    return jsonify({'result': 'Diabetic' if result == 1 else 'Not diabetic'})


if __name__ == '__main__':
    app.run(debug=True)
