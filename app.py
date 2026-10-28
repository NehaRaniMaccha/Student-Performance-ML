from flask import Flask, request, jsonify
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor

app = Flask(__name__)

# Load the dataset
data = pd.read_csv("student_performance.csv")

# Features and target
X = data[
    [
        "study_hours",
        "attendance",
        "previous_score",
        "assignments_completed",
        "sleep_hours"
    ]
]

y = data["final_score"]

# Train the model
model = RandomForestRegressor(
    n_estimators=100,
    max_depth=None,
    min_samples_split=2,
    random_state=42
)

model.fit(X, y)


@app.route("/")
def home():
    return "Student Performance Prediction API is running!"


@app.route("/predict", methods=["POST"])
def predict():

    try:
        input_data = request.get_json()

        required_features = [
            "study_hours",
            "attendance",
            "previous_score",
            "assignments_completed",
            "sleep_hours"
        ]

        # Check that all required values are present
        for feature in required_features:
            if feature not in input_data:
                return jsonify({
                    "error": f"Missing feature: {feature}"
                }), 400

        # Create input DataFrame
        student = pd.DataFrame([{
            feature: float(input_data[feature])
            for feature in required_features
        }])

        # Make prediction
        prediction = model.predict(student)[0]

        return jsonify({
            "predicted_final_score": round(float(prediction), 2)
        })

    except Exception as e:
        return jsonify({
            "error": str(e)
        }), 400


if __name__ == "__main__":
    app.run(debug=True)