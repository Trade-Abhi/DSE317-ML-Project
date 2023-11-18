# %%
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix



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

# Initialize and train the Logistic Regression model
log_reg = LogisticRegression(max_iter=1000, random_state=42, multi_class='multinomial')
log_reg.fit(X_train, y_train)

# Predict on the validation set and evaluate
val_predictions = log_reg.predict(X_val)
print("Classification Report:")
print(classification_report(y_val, val_predictions))
print("Confusion Matrix:")
print(confusion_matrix(y_val, val_predictions))

# Load and transform the test data
test_data = pd.read_csv('Test_Data.csv')
test_data_transformed = preprocessor.transform(test_data)

# Predict on the test data
test_predictions = log_reg.predict(test_data_transformed)

# Save the predictions to a CSV file
pd.DataFrame(test_predictions, columns=['Predicted_Price_Range']).to_csv('Test_Predictions.csv', index=False)



