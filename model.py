import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)
from sklearn.inspection import permutation_importance


# ============================================
# 1. LOAD DATASET
# ============================================

data = pd.read_csv("student_performance.csv")

print("\n" + "=" * 50)
print("STUDENT PERFORMANCE DATASET")
print("=" * 50)

print("\nFirst 5 rows:")
print(data.head())

print("\nDataset Shape:")
print(data.shape)

print("\nColumn Names:")
print(data.columns.tolist())

print("\nMissing Values:")
print(data.isnull().sum())


# ============================================
# 2. PREPARE DATA
# ============================================

# Remove rows containing missing values
data = data.dropna()

# Display information about the dataset
print("\nDataset after removing missing values:")
print(data.shape)


# ============================================
# 3. SELECT FEATURES AND TARGET
# ============================================

# Target column
target_column = "final_score"

# Features
X = data.drop(columns=[target_column])

# Target
y = data[target_column]

# Convert categorical columns into numerical values
X = pd.get_dummies(X, drop_first=True)

print("\nFeatures used for prediction:")
print(X.columns.tolist())

print("\nTarget:")
print(target_column)


# ============================================
# 4. TRAIN-TEST SPLIT
# ============================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))


# ============================================
# 5. RANDOM FOREST MODEL
# ============================================

random_forest = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

random_forest.fit(X_train, y_train)

# Predictions
y_pred = random_forest.predict(X_test)


# ============================================
# 6. MODEL EVALUATION
# ============================================

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("\n" + "=" * 50)
print("MODEL EVALUATION")
print("=" * 50)

print(f"\nMean Absolute Error (MAE): {mae:.2f}")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"R² Score: {r2:.2f}")


# ============================================
# 7. CROSS-VALIDATION
# ============================================

cv_scores = cross_val_score(
    random_forest,
    X,
    y,
    cv=5,
    scoring="r2"
)

print("\n" + "=" * 50)
print("5-FOLD CROSS-VALIDATION")
print("=" * 50)

print("\nCross-validation scores:")
print(cv_scores)

print(f"\nAverage CV R² Score: {cv_scores.mean():.2f}")
print(f"Standard Deviation: {cv_scores.std():.2f}")


# ============================================
# 8. HYPERPARAMETER TUNING
# ============================================

print("\n" + "=" * 50)
print("HYPERPARAMETER TUNING")
print("=" * 50)

param_grid = {
    "n_estimators": [50, 100, 150],
    "max_depth": [None, 10, 20],
    "min_samples_split": [2, 5]
}

grid_search = GridSearchCV(
    estimator=RandomForestRegressor(random_state=42),
    param_grid=param_grid,
    cv=5,
    scoring="r2",
    n_jobs=-1
)

grid_search.fit(X_train, y_train)

print("\nBest Parameters:")
print(grid_search.best_params_)

print(f"\nBest Cross-Validation R² Score: {grid_search.best_score_:.2f}")


# ============================================
# 9. EVALUATE TUNED MODEL
# ============================================

best_model = grid_search.best_estimator_

tuned_predictions = best_model.predict(X_test)

tuned_mae = mean_absolute_error(y_test, tuned_predictions)
tuned_mse = mean_squared_error(y_test, tuned_predictions)
tuned_rmse = np.sqrt(tuned_mse)
tuned_r2 = r2_score(y_test, tuned_predictions)

print("\n" + "=" * 50)
print("TUNED MODEL EVALUATION")
print("=" * 50)

print(f"\nTuned MAE: {tuned_mae:.2f}")
print(f"Tuned MSE: {tuned_mse:.2f}")
print(f"Tuned RMSE: {tuned_rmse:.2f}")
print(f"Tuned R² Score: {tuned_r2:.2f}")


# ============================================
# 10. FEATURE IMPORTANCE
# ============================================

feature_importance = pd.Series(
    best_model.feature_importances_,
    index=X.columns
).sort_values(ascending=False)

print("\n" + "=" * 50)
print("FEATURE IMPORTANCE")
print("=" * 50)

print(feature_importance)


# ============================================
# 11. FEATURE IMPORTANCE GRAPH
# ============================================

plt.figure(figsize=(10, 6))

feature_importance.head(10).plot(kind="bar")

plt.title("Top 10 Important Features")
plt.xlabel("Features")
plt.ylabel("Importance")
plt.xticks(rotation=45)
plt.tight_layout()

plt.show()


# ============================================
# 12. ACTUAL VS PREDICTED VALUES
# ============================================

plt.figure(figsize=(8, 6))

plt.scatter(y_test, tuned_predictions)

plt.xlabel("Actual Final Score")
plt.ylabel("Predicted Final Score")
plt.title("Actual vs Predicted Student Performance")

plt.tight_layout()
plt.show()


# ============================================
# 13. RESIDUAL ANALYSIS
# ============================================

residuals = y_test - tuned_predictions

plt.figure(figsize=(8, 6))

plt.scatter(tuned_predictions, residuals)

plt.axhline(y=0, linestyle="--")

plt.xlabel("Predicted Final Score")
plt.ylabel("Residual")
plt.title("Residual Analysis")

plt.tight_layout()
plt.show()


# ============================================
# 14. PERMUTATION IMPORTANCE
# ============================================

permutation = permutation_importance(
    best_model,
    X_test,
    y_test,
    n_repeats=10,
    random_state=42,
    scoring="r2"
)

permutation_importance_df = pd.DataFrame({
    "Feature": X.columns,
    "Importance": permutation.importances_mean
}).sort_values(
    by="Importance",
    ascending=False
)

print("\n" + "=" * 50)
print("PERMUTATION FEATURE IMPORTANCE")
print("=" * 50)

print(permutation_importance_df)


# ============================================
# 15. NEW STUDENT PREDICTION
# ============================================

print("\n" + "=" * 50)
print("NEW STUDENT PREDICTION")
print("=" * 50)

# Use the average values of the test data
new_student = X_test.iloc[[0]]

prediction = best_model.predict(new_student)

print("\nPredicted Final Score:",
      round(prediction[0], 2))

print("\n" + "=" * 50)
print("MODEL EVALUATION COMPLETED")
print("=" * 50)