import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def train_model(df):
    df = df.copy()
    df['diagnosis'] = df['diagnosis'].map({'M': 0, 'B': 1})
    X = df.drop(['id', 'diagnosis'], axis=1)
    y = df['diagnosis']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    acc = accuracy_score(y_test, clf.predict(X_test))

    return clf, acc, X.columns.tolist()
