import numpy as np
import pandas as pd
import scipy.sparse
from sklearn.preprocessing import LabelEncoder
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC


dataset = pd.read_csv('data/cleaned_resume.csv')
X = scipy.sparse.load_npz('tfidf_matrix.npz')
y = dataset['Category']  # Assuming 'Category' is the column name for labels

# print(X.shape)
# print(y.shape)
# print(y.value_counts())

# Encode the labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)
# print(y_encoded[:10])
# print(le.classes_)
joblib.dump(le, 'label_encoder.pkl')

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y)
# print("X_train:", X_train.shape)
# print("X_test:", X_test.shape)
# print("y_train:", y_train.shape)
# print("y_test:", y_test.shape)

# Logistic Regression
classifier_lr = LogisticRegression(random_state=42, max_iter=1000)
classifier_lr.fit(X_train, y_train)

# Predictions
y_pred_lr = classifier_lr.predict(X_test)

# print the report
# print(accuracy_score(y_test, y_pred_lr))
# print(classification_report(y_test, y_pred_lr, target_names=le.classes_))


# random forest
classifier = RandomForestClassifier(
    n_estimators=200, class_weight='balanced', random_state=42, n_jobs=-1)
classifier.fit(X_train, y_train)
y_pred_rf = classifier.predict(X_test)
print(accuracy_score(y_test, y_pred_rf))
print(classification_report(y_test, y_pred_rf, target_names=le.classes_))


# svc
classifier_svc = LinearSVC(class_weight='balanced',
                           random_state=42, max_iter=2000)
classifier_svc.fit(X_train, y_train)
y_pred_svc = classifier_svc.predict(X_test)
print(accuracy_score(y_test, y_pred_svc))
print(classification_report(y_test, y_pred_svc, target_names=le.classes_))


# save the best model
joblib.dump(classifier_svc, 'model.pkl')
print("Model saved as model.pkl")
tfidf = joblib.load('tfidf.pkl')
le = joblib.load('label_encoder.pkl')
