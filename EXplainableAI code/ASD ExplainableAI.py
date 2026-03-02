import numpy as np
import matplotlib.pyplot as plt

# Assuming voting_clf is your trained VotingClassifier with lgb_model, xgb_model, catboost_model
# and X_train, y_train, X.columns are defined

# ---------- 1. Extract direct feature importances from individual models ---------
models = {
    "LightGBM": lgb_model,
    "XGBoost": xgb_model,
    "CatBoost": catboost_model
}

# Get feature importances from each model
importances = {}
for name, model in models.items():
    # Train the model before getting feature importance
    model.fit(X_train, y_train)
    if hasattr(model, 'feature_importances_'):  # XGBoost, LightGBM
        importances[name] = model.feature_importances_
    elif hasattr(model, 'get_feature_importance'):  # CatBoost
        importances[name] = model.get_feature_importance()

# Convert to numpy arrays and ensure same length as feature names
importances_array = np.array([imp for imp in importances.values()])
if importances_array.shape[1] != len(X.columns):
    raise ValueError("Number of features in importances does not match feature names")

# ---------- 2. Average feature importances across models ---------
mean_importances = np.mean(importances_array, axis=0)

# ---------- 3. Normalize all feature importances to 100% ---------
total_importance = sum(mean_importances)
mean_importances_percent = (mean_importances / total_importance) * 100

# ---------- 4. Get top 10 features ---------
top_10_indices = np.argsort(mean_importances_percent)[::-1][:10]
top_10_importances = mean_importances_percent[top_10_indices]
top_10_feature_names = [X.columns[i] for i in top_10_indices]

# ---------- 5. Print top 10 feature importances in 100% ---------
print("Top 10 Features and Their Importance Percentages (Normalized to 100%):")
for name, importance in zip(top_10_feature_names, top_10_importances):
    print(f"{name}: {importance:.2f}%")

# ---------- 6. Plot the top 10 feature importances in percentage ---------
plt.figure(figsize=(10, 6))
plt.bar(top_10_feature_names, top_10_importances, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEEAD', '#D4A5A5', '#9B59B6', '#3498DB', '#E74C3C', '#2ECC71'])
plt.title("Top 10 Feature Importance (Percentage) across XGB + LGBM + Catboost")
plt.xlabel("Features")
plt.ylabel("Importance (%)")
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('top_10_feature_importance_100_percent.png', dpi=300, bbox_inches='tight')
plt.show()