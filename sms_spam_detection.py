import pandas as pd
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import string
import pickle
from flask import Flask, request, jsonify

# Load the dataset
df = pd.read_csv(r"C:\Users\nishantsharma\Downloads\sms-spam.csv", encoding='latin-1')
df = df[['ï»¿v1', 'v2']]  # Use the correct column names
df.columns = ['label', 'message']  # Rename the columns
print(df.head())

# Data Preprocessing
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

def preprocess_text(text):
    text = text.lower()
    text = ''.join([char for char in text if char not in string.punctuation])
    text = ' '.join([word for word in text.split() if word not in stop_words])
    text = ' '.join([stemmer.stem(word) for word in text.split()])
    return text

df['processed_message'] = df['message'].apply(preprocess_text)
print(df.head())

# Feature Extraction
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['processed_message'])
y = df['label']

# Model Training
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = MultinomialNB()
model.fit(X_train, y_train)

# Predict on the test set
y_pred = model.predict(X_test)

# Model Evaluation
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, pos_label='spam')
recall = recall_score(y_test, y_pred, pos_label='spam')
f1 = f1_score(y_test, y_pred, pos_label='spam')

print(f'Accuracy: {accuracy}')
print(f'Precision: {precision}')
print(f'Recall: {recall}')
print(f'F1 Score: {f1}')

# Save the model
with open('model.pkl', 'wb') as file:
    pickle.dump(model, file)

# Save the vectorizer
with open('vectorizer.pkl', 'wb') as file:
    pickle.dump(vectorizer, file)

# Load the model and vectorizer for Flask app
with open('model.pkl', 'rb') as file:
    model = pickle.load(file)
with open('vectorizer.pkl', 'rb') as file:
    vectorizer = pickle.load(file)

# Create Flask app
app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json['message']
    processed_data = preprocess_text(data)
    vectorized_data = vectorizer.transform([processed_data])
    prediction = model.predict(vectorized_data)
    return jsonify({'prediction': prediction[0]})

if __name__ == '__main__':
    app.run(debug=True)


