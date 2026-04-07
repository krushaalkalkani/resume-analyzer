import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
import scipy.sparse


dataset = pd.read_csv(
    'data/Resume.csv')
# print the first rusume full text
# print(dataset['Resume_str'][0])

# Create a function called clean_text in preprocess.py. It should take one resume text as input and return the cleaned version.
# Inside the function, do these 5 things in this exact order:

# Remove HTML tags — anything that looks like <...>
# Remove URLs — anything that starts with http or www
# Remove special characters and numbers — keep only letters (a-z, A-Z) and spaces
# Lowercase the whole text
# Remove extra spaces — multiple spaces become one space, and strip spaces from start/end


def clean_text(text):
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', '', text)
    # Remove URLs
    text = re.sub(r'http\S+|www\S+', '', text)
    # Remove special characters and numbers
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Lowercase the whole text
    text = text.lower()
    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    return text


# cleaned = clean_text(dataset['Resume_str'][0])
# print(cleaned[:500])  # print first 500 characters only

# Apply the cleaning function to the ENTIRE dataset
dataset['cleaned_resume'] = dataset['Resume_str'].apply(clean_text)

# print the first 3 resume with 300 characters of the cleaned version
# for i in range(3):
#     print(f"Original Resume {i+1}:\n{dataset['Resume_str'][i][:300]}...\n")
#     print(f"Cleaned Resume {i+1}:\n{dataset['cleaned_resume'][i][:300]}...\n")
#     print("-" * 80)
# print(dataset.shape)

# Apply TF-IDF to the cleaned resumes
# Limit to top 5000 features
tfidf = TfidfVectorizer(max_features=5000, stop_words='english')
X = tfidf.fit_transform(dataset['cleaned_resume'])
# print(X.shape)
# print(tfidf.get_feature_names_out()[:20])  # print first 20 feature names

# Save the cleaned data and the TF-IDF vectorizer
joblib.dump(tfidf, 'tfidf.pkl')
dataset.to_csv('data/cleaned_resume.csv', index=False)
dataset.to_csv('data/cleaned_resume.csv', index=False)

scipy.sparse.save_npz('tfidf_matrix.npz', X)
print("Vectorizer saved!")
print("Cleaned dataset saved!")
print("TF-IDF matrix saved!")
