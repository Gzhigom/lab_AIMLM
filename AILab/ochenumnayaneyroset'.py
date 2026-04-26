import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
data = pd.DataFrame({
    "text": [
        "Win a free lottery prize now",
        "Limited offer buy today",
        "Meeting scheduled at 10 am",
        "Project report attached",
        "Congratulations you won money",
        "Let's discuss the budget"
    ],
    "spam": [1, 1, 0, 0, 1, 0]
})

X = data["text"]
y = data["spam"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.33,
    random_state=42,
    stratify=y
)
model = Pipeline(steps=[
    ("tfidf", TfidfVectorizer(
        lowercase=True,
        stop_words="english"  # для русского нужен другой список/подход
    )),
    ("clf", LogisticRegression(max_iter=1000))
])

model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))