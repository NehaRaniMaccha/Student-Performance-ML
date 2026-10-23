import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ==========================================
# 1. LOAD DATASET
# ==========================================

df = pd.read_csv("student_performance.csv")

print("\nFirst 5 rows of the dataset:")
print(df.head())

print("\nDataset shape:")
print(df.shape)

print("\nMissing values:")
print(df.isnull().sum())


# ==========================================
# 2. SELECT FEATURES AND TARGET
# ==========================================

X = df[
    [
        "study_hours",
        "attendance",
        "previous_score",
        "assignments_completed",
        "sleep_hours"
    ]
]

y = df["final_score"]


# ==========================================
# 3. SPLIT DATA INTO TRAINING AND TESTING
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\nTraining data size:", X_train.shape)
print("Testing data size:", X_test.shape)


# ==========================================
# 4. CREATE MACHINE LEARNING MODELS
# ==========================================

linear_model = LinearRegression()

decision_tree = DecisionTreeRegressor(
    random_state=42
)

random_forest = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)


# ==========================================
# 5. TRAIN THE MODELS
# ==========================================

linear_model.fit(X_train, y_train)

decision_tree.fit(X_train, y_train)

random_forest.fit(X_train, y_train)


# ==========================================
# 6. MAKE PREDICTIONS
# ==========================================

linear_prediction = linear_model.predict(X_test)

tree_prediction = decision_tree.predict(X_test)

forest_prediction = random_forest.predict(X_test)


# ==========================================
# 7. CALCULATE PERFORMANCE METRICS
# ==========================================

def calculate_metrics(actual, predicted):

    mae = mean_absolute_error(actual, predicted)

    mse = mean_squared_error(actual, predicted)

    rmse = np.sqrt(mse)

    r2 = r2_score(actual, predicted)

    return mae, mse, rmse, r2


# Linear Regression metrics
linear_mae, linear_mse, linear_rmse, linear_r2 = calculate_metrics(
    y_test,
    linear_prediction
)


# Decision Tree metrics
tree_mae, tree_mse, tree_rmse, tree_r2 = calculate_metrics(
    y_test,
    tree_prediction
)


# Random Forest metrics
forest_mae, forest_mse, forest_rmse, forest_r2 = calculate_metrics(
    y_test,
    forest_prediction
)


# ==========================================
# 8. CREATE FINAL RESULTS TABLE
# ==========================================

results = pd.DataFrame({

    "Model": [
        "Linear Regression",
        "Decision Tree",
        "Random Forest"
    ],

    "MAE": [
        linear_mae,
        tree_mae,
        forest_mae
    ],

    "MSE": [
        linear_mse,
        tree_mse,
        forest_mse
    ],

    "RMSE": [
        linear_rmse,
        tree_rmse,
        forest_rmse
    ],

    "R2 Score": [
        linear_r2,
        tree_r2,
        forest_r2
    ]
})


# ==========================================
# 9. DISPLAY FINAL RESULTS
# ==========================================

print("\n")
print("=" * 80)
print("FINAL MODEL PERFORMANCE COMPARISON")
print("=" * 80)

print(
    results.round(4).to_string(index=False)
)

print("=" * 80)


# ==========================================
# 10. FIND BEST MODEL
# ==========================================

best_model = results.loc[
    results["R2 Score"].idxmax(),
    "Model"
]

best_r2 = results["R2 Score"].max()

print("\nBest Model:", best_model)

print("Best R2 Score:", round(best_r2, 4))


# ==========================================
# 11. MODEL COMPARISON GRAPH
# ==========================================

plt.figure(figsize=(8, 5))

sns.barplot(
    x="Model",
    y="R2 Score",
    data=results
)

plt.title("Model Performance Comparison")

plt.xlabel("Machine Learning Model")

plt.ylabel("R2 Score")

plt.xticks(rotation=15)

plt.tight_layout()

plt.show()


# ==========================================
# 12. ACTUAL VS PREDICTED GRAPH
# ==========================================

plt.figure(figsize=(7, 5))

plt.scatter(
    y_test,
    forest_prediction
)

plt.xlabel("Actual Final Score")

plt.ylabel("Predicted Final Score")

plt.title("Actual vs Predicted Scores - Random Forest")

plt.tight_layout()

plt.show()


# ==========================================
# 13. FEATURE IMPORTANCE
# ==========================================

importance = pd.Series(
    random_forest.feature_importances_,
    index=X.columns
)

importance = importance.sort_values(
    ascending=False
)

plt.figure(figsize=(8, 5))

importance.plot(
    kind="bar"
)

plt.title("Feature Importance - Random Forest")

plt.xlabel("Features")

plt.ylabel("Importance")

plt.xticks(rotation=30)

plt.tight_layout()

plt.show()


# ==========================================
# 14. PREDICT SCORE FOR A NEW STUDENT
# ==========================================

new_student = pd.DataFrame({

    "study_hours": [6],

    "attendance": [85],

    "previous_score": [75],

    "assignments_completed": [8],

    "sleep_hours": [7]
})


prediction = random_forest.predict(new_student)


print("\n")
print("=" * 40)
print("NEW STUDENT PREDICTION")
print("=" * 40)

print(
    "Predicted Final Score:",
    round(prediction[0], 2)
)

print("=" * 40)