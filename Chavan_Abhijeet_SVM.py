# %%
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV

# %%
# Load the data
train_data = pd.read_csv('Train_Data.csv')
train_labels = pd.read_csv('Traindata_classlabels.csv').values.ravel()

# Define continuous and binary features
continuous_features = ['battery_power', 'clock_speed', 'fc', 'int_memory', 'm_dep', 'mobile_wt', 'n_cores', 'pc', 'px_height', 'px_width', 'ram', 'sc_h', 'sc_w', 'talk_time']
binary_features = ['blue', 'dual_sim', 'four_g', 'three_g', 'touch_screen', 'wifi']

# Create a column transformer with scaling for continuous features and one-hot encoding for binary features
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), continuous_features),
        ('cat', OneHotEncoder(drop='if_binary'), binary_features)
    ])

# Apply transformations to the training data
X_train_transformed = preprocessor.fit_transform(train_data)

# Split the dataset into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_train_transformed, train_labels, test_size=0.2, random_state=42, stratify=train_labels)

# Initialize and train the simple SVM Classifier
svm_classifier = SVC(kernel='linear', random_state=42)
svm_classifier.fit(X_train, y_train)

# Predict on the validation set and evaluate for simple SVM
val_predictions = svm_classifier.predict(X_val)
print("Simple SVM - Classification Report:")
print(classification_report(y_val, val_predictions))
print("Simple SVM - Confusion Matrix:")
print(confusion_matrix(y_val, val_predictions))

# Hyperparameter tuning for SVM
param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': ['scale', 'auto'],
    'kernel': ['linear', 'rbf', 'poly']
}

grid_search = GridSearchCV(SVC(random_state=42), param_grid, cv=5, n_jobs=-1, verbose=2)
grid_search.fit(X_train, y_train)

print("Best Parameters for Hyperparameter Tuned SVM:", grid_search.best_params_)
print("Best Score for Hyperparameter Tuned SVM:", grid_search.best_score_)

# Initialize and train the Hyperparameter SVM Classifier
svm_classifier_HP = SVC(**grid_search.best_params_, random_state=42)
svm_classifier_HP.fit(X_train, y_train)

# Predict on the validation set and evaluate for Hyperparameter Tuned SVM
val_predictions_HP = svm_classifier_HP.predict(X_val)
print("Hyperparameter Tuned SVM - Classification Report:")
print(classification_report(y_val, val_predictions_HP))
print("Hyperparameter Tuned SVM - Confusion Matrix:")
print(confusion_matrix(y_val, val_predictions_HP))

# Preprocess and predict on test data (similar to train data preprocessing)
test_data = pd.read_csv('Test_Data.csv')
test_data_transformed = preprocessor.transform(test_data)
test_predictions = svm_classifier_HP.predict(test_data_transformed)

# Save the predictions to a CSV file
pd.DataFrame(test_predictions, columns=['Predicted_Price_Range']).to_csv('Test_Predictions_SVM.csv', index=False)



