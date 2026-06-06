{\rtf1\ansi\ansicpg1252\cocoartf2868
\cocoatextscaling0\cocoaplatform0{\fonttbl\f0\fswiss\fcharset0 Helvetica;}
{\colortbl;\red255\green255\blue255;}
{\*\expandedcolortbl;;}
\paperw11900\paperh16840\margl1440\margr1440\vieww11520\viewh8400\viewkind0
\pard\tx720\tx1440\tx2160\tx2880\tx3600\tx4320\tx5040\tx5760\tx6480\tx7200\tx7920\tx8640\pardirnatural\partightenfactor0

\f0\fs24 \cf0 import pandas as pd\
from sklearn.feature_extraction.text import CountVectorizer\
from sklearn.model_selection import train_test_split\
from sklearn.naive_bayes import MultinomialNB\
from sklearn.metrics import accuracy_score, confusion_matrix\

data = \{\
    "text": [\
        "Win a free iPhone now", \
        "Meeting scheduled at 10 AM", \
        "Update your bank account immediately",\
        "Lunch with team tomorrow"\
    ],\
    "label": ["phishing", "safe", "phishing", "safe"]\
\}\

df = pd.DataFrame(data)\

vectorizer = CountVectorizer()\
X = vectorizer.fit_transform(df["text"])\
y = df["label"]\

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)\

model = MultinomialNB()\
model.fit(X_train, y_train)\

y_pred = model.predict(X_test)\

print("Accuracy:", accuracy_score(y_test, y_pred))\
print("Confusion Matrix:\\n", confusion_matrix(y_test, y_pred))\
}
