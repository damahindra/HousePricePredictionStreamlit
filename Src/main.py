import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_squared_error
import xgboost as xgb
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.metrics import classification_report
import joblib
import matplotlib.pyplot as plt

file_path = ('Dataset/Cleaned_data_for_model.csv')
data = pd.read_csv(file_path)

# Data Preprocessing
# Handle missing values (fill numerical columns with median)
data.fillna(data.median(numeric_only=True), inplace=True)

# Convert categorical features to dummy variables
data = pd.get_dummies(data, columns=['property_type', 'location', 'city', 'purpose'], drop_first=True)

# Define features and target variable
X = data.drop("price", axis=1)
y = data["price"]
bins = [0, 100000, 200000, 300000, 400000, np.inf]  # Define price ranges
labels = ['Low', 'Medium', 'High', 'Very High', 'Luxury']  # Assign category labels
y_category = pd.cut(y, bins=bins, labels=labels, right=False)

# Now split the data again based on the new categorical target
X_train, X_test, y_train, y_test = train_test_split(X, y_category, test_size=0.2, random_state=42)

label_encoder = LabelEncoder()
y_train = label_encoder.fit_transform(y_train)
y_test = label_encoder.transform(y_test)

joblib.dump(label_encoder, 'label_encoder.pkl')

# # Standardize the features
# scaler = StandardScaler()
# X_train = scaler.fit_transform(X_train)
# X_test = scaler.transform(X_test)

# joblib.dump(scaler, 'scaler.pkl')

# # --- Model 1: Feedforward Neural Network ---
# nn_model = Sequential([
#     Dense(64, activation='relu', input_dim=X_train.shape[1]),
#     Dense(32, activation='relu'),
#     Dense(len(labels), activation='softmax')  # Use softmax for classification
# ])

# # Compile the model
# nn_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# # Train the model
# nn_model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=1)

# # Predict and evaluate
# nn_predictions = np.argmax(nn_model.predict(X_test), axis=1)
# print("Neural Network Classification Report:")
# print(classification_report(y_test, nn_predictions))

# joblib.dump(nn_model, 'nn_model.pkl')
# nn_model_loaded = joblib.load('nn_model.pkl')

# # --- Model 2: Random Forest Regressor ---
# rf_model = RandomForestClassifier(random_state=42, n_estimators=100)

# # Train the model
# rf_model.fit(X_train, y_train)

# # Predict and evaluate
# rf_predictions = rf_model.predict(X_test)
# print("Random Forest Classification Report:")
# print(classification_report(y_test, rf_predictions))

# joblib.dump(rf_model, 'rf_model.pkl')
# rf_model_loaded = joblib.load('rf_model.pkl')

# # --- Model 3: XGBoost Regressor ---
# xgb_model = xgb.XGBClassifier(objective='multi:softmax', num_class=len(labels), random_state=42, n_estimators=100)

# # Train the model
# xgb_model.fit(X_train, y_train)

# # Predict and evaluate
# xgb_predictions = xgb_model.predict(X_test)
# print("XGBoost Classification Report:")
# print(classification_report(y_test, xgb_predictions))

# joblib.dump(xgb_model, 'xgb_model.pkl')
# xgb_model_loaded = joblib.load('xgb_model.pkl')


# accuracies = {
#     'Neural Network': classification_report(y_test, nn_predictions, output_dict=True)['accuracy'],
#     'Random Forest': classification_report(y_test, rf_predictions, output_dict=True)['accuracy'],
#     'XGBoost': classification_report(y_test, xgb_predictions, output_dict=True)['accuracy']
# }

# # Print out the accuracies
# for model, accuracy in accuracies.items():
#     print(f"{model} Accuracy: {accuracy:.4f}")

# # Visualisasi Akurasi
# models = list(accuracies.keys())
# accuracy_values = list(accuracies.values())

# plt.figure(figsize=(8, 6))
# plt.bar(models, accuracy_values, color=['skyblue', 'salmon', 'lightgreen'])
# plt.xlabel('Model')
# plt.ylabel('Accuracy')
# plt.title('Accuracy Comparison of Regression Models')
# plt.ylim(0, 1)
# plt.show()