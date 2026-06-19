import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, accuracy_score
from sklearn.neural_network import MLPClassifier

data = pd.read_csv('spam_ham_dataset.csv')
data = data[['text', 'label_num']]
X = data["text"]
y = data["label_num"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

model = Pipeline(steps=[
    ("tfidf", TfidfVectorizer(
        lowercase=True,
        stop_words="english"
    )),
    ("clf", MLPClassifier(
        hidden_layer_sizes=(32, 16),
        activation='relu',
        solver='adam',
        max_iter=100,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.2
    ))
])

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("MLP Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=["Не спам", "Спам"]))