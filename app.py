from flask import Flask, request
import pandas as pd
import numpy as np
from keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Embedding, Conv1D, MaxPooling1D, Flatten, Dense
from sklearn.model_selection import train_test_split

app = Flask(__name__)

# Load the data for training
data = pd.read_csv('m_data2.csv')
data.columns = ['url', 'label']
data['label'] = data['label'].apply(lambda x: 1 if x == 'bad' else 0)

# Preprocess the data
tokenizer = Tokenizer()
tokenizer.fit_on_texts(data['url'])
X = tokenizer.texts_to_sequences(data['url'])
max_len = max([len(seq) for seq in X])
X = pad_sequences(X, maxlen=max_len)
y = data['label'].values

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the 1D CNN model
model = Sequential()
model.add(Embedding(input_dim=len(tokenizer.word_index) + 1, output_dim=128, input_length=max_len))
model.add(Conv1D(filters=64, kernel_size=5, activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(Flatten())
model.add(Dense(units=64, activation='relu'))
model.add(Dense(units=1, activation='sigmoid'))

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=1, batch_size=64, validation_data=(X_test, y_test))

@app.route('/predict_url', methods=['POST'])
def predict_url():
    url = request.form['url']
    sequence = tokenizer.texts_to_sequences([url])
    sequence = pad_sequences(sequence, maxlen=max_len)
    prediction = model.predict(sequence)[0][0]
    if prediction >= 0.5:
        result = 'Bad URL'
    else:
        result = 'Good URL'
    return result

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000, debug=False)
