import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

# Load extracted features
data = pd.read_csv('audio_features.csv')
X = data.drop('label', axis=1)
y = data['label'].map({'real': 0, 'fake': 1})  # Encode labels as 0 for real, 1 for fake

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Random Forest model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")

# Save the model for use in Django
joblib.dump(model, 'audio_fake_detector.pkl')
