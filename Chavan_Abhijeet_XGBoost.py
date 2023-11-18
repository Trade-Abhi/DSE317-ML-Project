# %%
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import xgboost as xgb
from sklearn.metrics import classification_report, confusion_matrix


# %%

# Load the data
train_data = pd.read_csv('Train_Data.csv')
train_labels = pd.read_csv('Traindata_classlabels.csv').values.ravel()

# Continuous features
continuous_features = ['battery_power', 'clock_speed', 'fc', 'int_memory', 'm_dep', 'mobile_wt', 'n_cores', 'pc', 'px_height', 'px_width', 'ram', 'sc_h', 'sc_w', 'talk_time']

# Binary features
binary_features = ['blue', 'dual_sim', 'four_g', 'three_g', 'touch_screen', 'wifi']

# Scale continuous features
scaler = StandardScaler()
train_data[continuous_features] = scaler.fit_transform(train_data[continuous_features])

# One-hot encode binary features
encoder = OneHotEncoder(sparse=False)
binary_encoded = encoder.fit_transform(train_data[binary_features])

# Combine continuous and binary features
X_combined = pd.concat([pd.DataFrame(train_data[continuous_features]), pd.DataFrame(binary_encoded)], axis=1)

# Split the dataset into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_combined, train_labels, test_size=0.2, random_state=42, stratify=train_labels)

# Convert the dataset into DMatrix
dtrain = xgb.DMatrix(data=X_train, label=y_train)
dval = xgb.DMatrix(data=X_val, label=y_val)

# Set the parameters for XGBoost
params = {
    'max_depth': 6,
    'eta': 0.3,
    'objective': 'multi:softmax',
    'num_class': 4  # Number of classes
}

# Train the model
xgb_model = xgb.train(params, dtrain, num_boost_round=10)

# Predict on validation set
val_predictions_xgb = xgb_model.predict(dval)

# Classification Report and Confusion Matrix
print("Classification Report:")
print(classification_report(y_val, val_predictions_xgb))
print("Confusion Matrix:")
print(confusion_matrix(y_val, val_predictions_xgb))

# Load the test data
test_data = pd.read_csv('Test_Data.csv')

# Preprocess test data
test_data[continuous_features] = scaler.transform(test_data[continuous_features])
test_data_binary_encoded = encoder.transform(test_data[binary_features])
test_data_combined = pd.concat([pd.DataFrame(test_data[continuous_features]), pd.DataFrame(test_data_binary_encoded)], axis=1)

# Convert test data into DMatrix
dtest = xgb.DMatrix(test_data_combined)

# Make predictions on test data
test_predictions = xgb_model.predict(dtest)

# Save the predictions to a CSV file
pd.DataFrame(test_predictions, columns=['Predicted_Price_Range']).to_csv('Test_Predictions_XGBoost.csv', index=False)



