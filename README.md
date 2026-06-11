# Sentiment-Analysis
Sentiment Analysis with an Recurrent Neural Networks (RNN)

**Importing Libraries and Dataset**
import pandas as pd
import numpy as np
import re  
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense, Embedding

**Loading Dataset**

pd.read_csv() : Reads the CSV file into a Pandas DataFrame
data.columns : Accesses the column names of the DataFrame
tolist() : Converts the column names from an Index object to a regular Python list

data = pd.read_csv('swiggy.csv')
print("Columns in the dataset:")
print(data.columns.tolist())

**Text Cleaning and Sentiment Labeling**
The review text is cleaned and sentiment labels are generated from ratings before training the model.

Converts review text to lowercase
Removes special characters and punctuation
Creates sentiment labels from ratings
Removes rows with missing values

data["Review"] = data["Review"].str.lower()
data["Review"] = data["Review"].replace(r'[^a-z0-9\s]', '', regex=True)

data['sentiment'] = data['Avg Rating'].apply(lambda x: 1 if x > 3.5 else 0)
data = data.dropna()

**Tokenization and Padding**
The text data is converted into numerical sequences and padded to ensure all inputs have the same length for model training

ax_features = 5000
max_length = 200

tokenizer = Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(data["Review"])
X = pad_sequences(tokenizer.texts_to_sequences(
    data["Review"]), maxlen=max_length)
y = data['sentiment'].values

**Splitting **
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.1, random_state=42, stratify=y_train
)

**RNN**
model = Sequential([
    Embedding(input_dim=max_features, output_dim=16, input_length=max_length),
    SimpleRNN(64, activation='tanh', return_sequences=False),
    Dense(1, activation='sigmoid')
])

model.compile(
    loss='binary_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

history = model.fit(
    X_train, y_train,
    epochs=5,
    batch_size=32,
    validation_data=(X_val, y_val),
    verbose=1
)

score = model.evaluate(X_test, y_test, verbose=0)
print(f"Test accuracy: {score[1]:.2f}")

**Predicting Sentiment**
def predict_sentiment(review_text):
    text = review_text.lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)

    seq = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=max_length)

    prediction = model.predict(padded)[0][0]
    return f"{'Positive' if prediction >= 0.5 else 'Negative'} (Probability: {prediction:.2f})"


sample_review = "The food was great."
print(f"Review: {sample_review}")
print(f"Sentiment: {predict_sentiment(sample_review)}")

<img width="451" height="108" alt="image" src="https://github.com/user-attachments/assets/a388af67-f5ee-4409-a992-7ed80658032c" />


