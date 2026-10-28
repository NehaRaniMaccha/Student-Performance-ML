import requests

url = "http://127.0.0.1:5000/predict"

student_data = {
    "study_hours": 6,
    "attendance": 85,
    "previous_score": 75,
    "assignments_completed": 8,
    "sleep_hours": 7
}

response = requests.post(url, json=student_data)

print("Status Code:", response.status_code)
print("Prediction:", response.json())