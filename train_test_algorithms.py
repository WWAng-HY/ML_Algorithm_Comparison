
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

# Simulating dataset creation
data = {
    "V1": ["POS", "NEG", "POS", "NEG", "POS", "NEG", "POS", "POS", "NEG", "POS"],
    "V2": [
        "Successful login with valid username and password",
        "Failed login with invalid username",
        "Positive registration case with all valid details",
        "Error: Weak password provided",
        "Valid email and username for registration",
        "No username entered for registration",
        "Test case for special character inputs",
        "Normal login case",
        "Invalid email format in registration",
        "Strong password and valid username successful case",
    ],
}
df = pd.DataFrame(data)

# Encode labels (POS -> 1, NEG -> 0)
df['V1'] = df['V1'].map({'POS': 1, 'NEG': 0})

# Splitting data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(df['V2'], df['V1'], test_size=0.3, random_state=42)

# List of algorithms
algorithms = {
    'Random Forest': RandomForestClassifier(),
    'Naive Bayes': MultinomialNB(),
    'Decision Tree': DecisionTreeClassifier(),
    'Logistic Regression': LogisticRegression(),
    'SVM': SVC(),
    'KNN': KNeighborsClassifier(n_neighbors=3)
}

# Dictionary to store results
results = {}

# Evaluate each algorithm
for name, model in algorithms.items():
    pipeline = Pipeline([
        ('vectorizer', CountVectorizer()),
        ('classifier', model)
    ])
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True)
    results[name] = report['accuracy']

# Display the results
print("Algorithm Performance Comparison:")
for name, accuracy in results.items():
    print(f"{name}: {accuracy:.2f}")
