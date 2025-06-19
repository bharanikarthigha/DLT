import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Step 1: Generate random binary classification dataset
X, y = make_classification(n_samples=1000,     # number of data points
                           n_features=5,       # number of input features
                           n_informative=3,    # number of informative features
                           n_redundant=0,      # no redundant features
                           n_classes=2,        # binary classification
                           random_state=42)

# Step 2: Split dataset into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

# Step 3: Create logistic regression model
model = LogisticRegression()

# Step 4: Train the model
model.fit(X_train, y_train)

# Step 5: Make predictions
y_pred = model.predict(X_test)

# Step 6: Evaluate the model
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
