import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import joblib

# Load dataset
data = pd.read_csv(r"C:\Users\Aaron\PANDAS12\optimal_fertilizer_project\fertilizer data.csv")

print(data.head())

# Encode categorical variables: Crop and Fertilizer_Type
crop_encoder = LabelEncoder()
data['Crop_enc'] = crop_encoder.fit_transform(data['Crop'])

fertilizer_encoder = LabelEncoder()
data['Fertilizer_Type_enc'] = fertilizer_encoder.fit_transform(data['Fertilizer_Type'])

# Features common for both models
feature_cols = ['N', 'P', 'K', 'pH', 'Temperature', 'Humidity', 'Rainfall', 'Crop_enc']

X = data[feature_cols]

# Target variables
y_type = data['Fertilizer_Type_enc']       # classification
y_amount = data['Fertilizer_Amount (kg/ha)']  # regression

# Split for fertilizer type model
X_train_t, X_test_t, y_train_t, y_test_t = train_test_split(X, y_type, test_size=0.2, random_state=42)

# Train classifier for fertilizer type
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train_t, y_train_t)

# Split for fertilizer amount model
X_train_a, X_test_a, y_train_a, y_test_a = train_test_split(X, y_amount, test_size=0.2, random_state=42)

# Train regressor for fertilizer amount
reg = RandomForestRegressor(n_estimators=100, random_state=42)
reg.fit(X_train_a, y_train_a)

# Save models and encoders
joblib.dump(clf, 'models/fertilizer_type_model.pkl')
joblib.dump(reg, 'models/fertilizer_amount_model.pkl')
joblib.dump(crop_encoder, 'models/crop_encoder.pkl')
joblib.dump(fertilizer_encoder, 'models/fertilizer_encoder.pkl')
