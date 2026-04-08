import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
import scipy.sparse


def clean_text(text):
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', '', text)
    # Remove URLs
    text = re.sub(r'http\S+|www\S+', '', text)
    # Remove special characters and numbers
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Lowercase
    text = text.lower()
    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    return text


if __name__ == "__main__":
    dataset = pd.read_csv('data/Resume.csv')
    dataset['cleaned_resume'] = dataset['Resume_str'].apply(clean_text)

    tfidf = TfidfVectorizer(max_features=5000, stop_words='english')
    X = tfidf.fit_transform(dataset['cleaned_resume'])

    joblib.dump(tfidf, 'tfidf.pkl')
    dataset.to_csv('data/cleaned_resume.csv', index=False)
    scipy.sparse.save_npz('tfidf_matrix.npz', X)

    print("Vectorizer saved!")
    print("Cleaned dataset saved!")
    print("TF-IDF matrix saved!")
