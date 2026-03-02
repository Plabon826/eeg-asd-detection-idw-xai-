# ======================================================
#   Voting Classifier (CatBoost, XGBoost, LightGBM) · Lever-2  +  4-way reporting + 5-Fold CV
# ======================================================
#
#   ▸ Repro-friendly: fixed random seeds
#   ▸ Handles   – Train-ORIG / Val / Test
#   ▸ Spits out – Accuracy, full classification report,
#                 and a confusion-matrix heat-map for
#                 EVERY split (so you spot trouble fast)
# ------------------------------------------------------

# ---------- installs (uncomment the next line if needed) ----------
# !pip install -q numpy pandas matplotlib seaborn scikit-learn catboost xgboost lightgbm

# ---------- imports ----------
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import VotingClassifier
from catboost import CatBoostClassifier
import lightgbm as lgb
import xgboost as xgb

# ---------- data ----------
DATA_PATH = '/content/MothercsvfileAutism.csv'  # Replace this with your actual file path
df = pd.read_csv(DATA_PATH).dropna()

# Optional “house-keeping” column removal if it exists (e.g., if you have an unnecessary column)
df = df.drop(columns=['File_Epoch']) if 'File_Epoch' in df.columns else df

# Label preprocessing: Convert labels to zero-origin (VotingClassifier likes that)
X = df.drop(columns=['Label'])
y = df['Label'] - df['Label'].min()  # Ensure zero-based labels for classification
NUM_CLASSES = y.nunique()

# ---------- 80 / 10 / 10 split (stratified) ----------
X_train, X_tmp, y_train, y_tmp = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=42)

X_val, X_test, y_val, y_test = train_test_split(
    X_tmp, y_tmp, test_size=0.50, stratify=y_tmp, random_state=42)

# ---------- CatBoost Model ----------
catboost_model = CatBoostClassifier(
    iterations=300,
    depth=4,
    learning_rate=0.1,
    loss_function='MultiClass',
    random_seed=42,
    verbose=False
)

# ---------- XGBoost Model ----------
xgb_model = xgb.XGBClassifier(
    objective='multi:softmax',
    num_class=NUM_CLASSES,
    eval_metric='mlogloss',
    max_depth=4,
    learning_rate=0.1,
    n_estimators=300,
    random_state=42
)

# ---------- LightGBM Model ----------
lgb_model = lgb.LGBMClassifier(
    objective='multiclass',
    num_class=NUM_CLASSES,
    metric='multi_logloss',
    max_depth=4,
    learning_rate=0.1,
    n_estimators=300,
    random_state=42
)

# ---------- Voting Classifier ----------
voting_classifier = VotingClassifier(
    estimators=[
        ('catboost', catboost_model),
        ('xgb', xgb_model),
        ('lgb', lgb_model)
    ],
    voting='soft'  # Soft voting uses predicted probabilities
)

# Train Voting Classifier model
voting_classifier.fit(X_train, y_train)

# ---------- 5-Fold Cross-Validation -----------
# Perform Stratified K-Fold cross-validation
stratified_kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Perform cross-validation and calculate the mean accuracy
cv_scores = cross_val_score(voting_classifier, X_train, y_train, cv=stratified_kfold, scoring='accuracy')

# Print cross-validation scores and the mean accuracy
print(f'Cross-validation scores: {cv_scores}')
print(f'Average cross-validation score (Mean Accuracy): {cv_scores.mean() * 100:.2f}%')

# ---------- Universal Reporter ----------
def full_report(model):
    """Accuracy, classification report & confusion matrix for each split."""
    sets = {
        'Train-ORIG ': (X_train,     y_train),
        'Val        ': (X_val,       y_val),
        'Test       ': (X_test,      y_test)
    }
    for tag, (X_set, y_set) in sets.items():
        preds = model.predict(X_set).astype(int).ravel()
        acc   = accuracy_score(y_set, preds)
        print(f'\n📊 Voting Classifier | {tag} accuracy: {acc:.4f}')
        print(classification_report(y_set, preds, digits=3, zero_division=0))
        cm = confusion_matrix(y_set, preds)
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm,
                    annot=True, fmt='d', cmap='coolwarm',
                    cbar=False, square=True,
                    xticklabels=range(NUM_CLASSES),
                    yticklabels=range(NUM_CLASSES))
        plt.title(f'Voting Classifier – {tag.strip()} Confusion Matrix')
        plt.xlabel('Predicted label'); plt.ylabel('True label')
        plt.tight_layout()
        plt.show()

# ---------- Go! ----------
full_report(voting_classifier)