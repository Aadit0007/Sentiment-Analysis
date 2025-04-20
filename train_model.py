import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import pickle

# Sample dataset
data = {
    'text': [
        'I love this product',
        'This is terrible',
        'Amazing experience',
        'Worst thing ever',
        'I feel happy',
        'I am very sad',
        'Great job',
        'Not good at all',
        'It is okay',
        'Pretty bad service'
    ],
    'label': [
        'positive',
        'negative',
        'positive',
        'negative',
        'positive',
        'negative',
        'positive',
        'negative',
        'neutral',
        'negative'
    ]
}

df = pd.DataFrame(data)

# Convert text to features
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df['text'])
y = df['label']

# Train model
model = MultinomialNB()
model.fit(X, y)

# Save model and vectorizer
with open('model.pkl', 'wb') as f:
    pickle.dump((model, vectorizer), f)
