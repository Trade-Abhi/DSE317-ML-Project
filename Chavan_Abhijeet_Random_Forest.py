# %%
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV



# %%
# Load the data
train_data = pd.read_csv('Train_Data.csv')
train_labels = pd.read_csv('Traindata_classlabels.csv').values.ravel()

# Continuous features
continuous_features = ['battery_power', 'clock_speed', 'fc', 'int_memory', 'm_dep', 'mobile_wt', 'n_cores', 'pc', 'px_height', 'px_width', 'ram', 'sc_h', 'sc_w', 'talk_time']

# Scale continuous features
scaler = StandardScaler()
train_data[continuous_features] = scaler.fit_transform(train_data[continuous_features])

# No transformation needed for binary features
# Split the dataset into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(train_data, train_labels, test_size=0.2, random_state=42, stratify=train_labels)

# Initialize and train the Random Forest Classifier
rf_classifier = RandomForestClassifier(random_state=42)
rf_classifier.fit(X_train, y_train)

# Predict on the validation set and evaluate
val_predictions = rf_classifier.predict(X_val)
print("Simple Random Forest - Classification Report:")
print(classification_report(y_val, val_predictions))
print("Simple Random Forest - Confusion Matrix:")
print(confusion_matrix(y_val, val_predictions))

# Hyperparameter tuning using GridSearchCV
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, 30, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}
grid_search = GridSearchCV(estimator=RandomForestClassifier(random_state=42), param_grid=param_grid, cv=5, n_jobs=-1, verbose=2, scoring='accuracy')
grid_search.fit(X_train, y_train)

# Best parameters and score
print("Best Parameters:", grid_search.best_params_)
print("Best Score:", grid_search.best_score_)

# Train and evaluate the Random Forest with best hyperparameters
best_model = grid_search.best_estimator_
val_predictions_HP = best_model.predict(X_val)
print("Random Forest with Hyperparameters - Classification Report:")
print(classification_report(y_val, val_predictions_HP))
print("Random Forest with Hyperparameters - Confusion Matrix:")
print(confusion_matrix(y_val, val_predictions_HP))

# Preprocess and predict on test data
test_data = pd.read_csv('Test_Data.csv')
test_data[continuous_features] = scaler.transform(test_data[continuous_features])
test_predictions = best_model.predict(test_data)

# Save predictions
pd.DataFrame(test_predictions, columns=['Predicted_Price_Range']).to_csv('Test_Predictions_RandomForest_HP.csv', index=False)



