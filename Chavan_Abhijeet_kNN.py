# %%
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.neighbors import KNeighborsClassifier
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

# Feature Selection
num_features = 10
bestfeatures = SelectKBest(score_func=f_classif, k=num_features)
X_train_selected = bestfeatures.fit_transform(X_train, y_train)
X_val_selected = bestfeatures.transform(X_val)

# Finding the optimal number of neighbors (k) for kNN
k_range = range(1, 15)
cv_scores = []

for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, X_train_selected, y_train, cv=10, scoring='accuracy')
    cv_scores.append(scores.mean())

optimal_k = k_range[cv_scores.index(max(cv_scores))]
print(f"The optimal number of neighbors is {optimal_k}")

# Plotting cross-validation scores
plt.plot(k_range, cv_scores)
plt.xlabel('Value of K for kNN')
plt.ylabel('Cross-Validated Accuracy')
plt.show()

# Building and Evaluating the kNN Classifier
knn_classifier = KNeighborsClassifier(n_neighbors=optimal_k)
knn_classifier.fit(X_train_selected, y_train)
val_predictions = knn_classifier.predict(X_val_selected)

# Classification Report and Confusion Matrix
print("Classification Report:")
print(classification_report(y_val, val_predictions))
print("Confusion Matrix:")
print(confusion_matrix(y_val, val_predictions))

# Preprocess and predict on test data (similar to train data preprocessing)
test_data = pd.read_csv('Test_Data.csv')
test_data_transformed = preprocessor.transform(test_data)
test_data_selected = bestfeatures.transform(test_data_transformed)
test_predictions = knn_classifier.predict(test_data_selected)

# Save the predictions to a CSV file
pd.DataFrame(test_predictions, columns=['Predicted_Price_Range']).to_csv('Test_Predictions_kNN.csv', index=False)



