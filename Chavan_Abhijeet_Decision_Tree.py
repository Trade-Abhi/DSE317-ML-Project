# %%
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix



# %%
# Load the data
train_data = pd.read_csv('Train_Data.csv')
train_labels = pd.read_csv('Traindata_classlabels.csv').values.ravel()

# Continuous features
continuous_features = ['battery_power', 'clock_speed', 'fc', 'int_memory', 'm_dep', 'mobile_wt', 'n_cores', 'pc', 'px_height', 'px_width', 'ram', 'sc_h', 'sc_w', 'talk_time']

# Scale continuous features
scaler = StandardScaler()
train_data[continuous_features] = scaler.fit_transform(train_data[continuous_features])

# No need to transform binary features; they are already in a suitable format
# Split the dataset into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(train_data, train_labels, test_size=0.2, random_state=42, stratify=train_labels)

# Initialize and train the Decision Tree Classifier
dt_classifier = DecisionTreeClassifier(random_state=42)
dt_classifier.fit(X_train, y_train)

# Predict on the validation set and evaluate
val_predictions = dt_classifier.predict(X_val)
print("Simple Decision Tree - Classification Report:")
print(classification_report(y_val, val_predictions))
print("Simple Decision Tree - Confusion Matrix:")
print(confusion_matrix(y_val, val_predictions))

# Feature importance and selection with selected features
importances = dt_classifier.feature_importances_
feature_importances = pd.DataFrame({'feature': train_data.columns, 'importance': importances})
feature_importances.sort_values(by='importance', ascending=False, inplace=True)
selected_features = feature_importances[feature_importances['importance'] >= 0.05]['feature']

# Train a new Decision Tree with selected features
X_train_selected = X_train[selected_features]
X_val_selected = X_val[selected_features]
dt_classifier_selected = DecisionTreeClassifier(random_state=42)
dt_classifier_selected.fit(X_train_selected, y_train)

# Predict and evaluate with selected features
val_predictions_selected = dt_classifier_selected.predict(X_val_selected)
print("Decision Tree with Selected Features - Classification Report:")
print(classification_report(y_val, val_predictions_selected))
print("Decision Tree with Selected Features - Confusion Matrix:")
print(confusion_matrix(y_val, val_predictions_selected))

# Preprocess and predict on test data
test_data = pd.read_csv('Test_Data.csv')
test_data[continuous_features] = scaler.transform(test_data[continuous_features])
test_data_selected = test_data[selected_features]
test_predictions = dt_classifier_selected.predict(test_data_selected)

# Save predictions
pd.DataFrame(test_predictions, columns=['Predicted_Price_Range']).to_csv('Test_Predictions_DT.csv', index=False)



