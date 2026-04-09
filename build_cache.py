import json
from collections import Counter

import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

dataset = pd.read_csv('data/cleaned_resume.csv')

JUNK_WORDS = {
    'years', 'experience', 'work', 'company', 'team', 'using', 'new',
    'also', 'including', 'within', 'able', 'various', 'one', 'two',
    'three', 'skills', 'ability', 'role', 'job', 'position', 'time',
    'well', 'strong', 'good', 'excellent', 'etc', 'day', 'month', 'year'
}


def get_top_words(category, n=30):
    category_data = dataset[dataset['Category'] == category]
    combined_text = ' '.join(category_data['cleaned_resume'].astype(str))
    words = combined_text.split()
    words = [word for word in words if word.lower() not in ENGLISH_STOP_WORDS]
    return Counter(words).most_common(n)


def get_skills_for_category(category, n=15):
    top_words = get_top_words(category, n * 3)
    filtered_words = [
        word for word, _count in top_words
        if word not in ENGLISH_STOP_WORDS
        and word not in JUNK_WORDS
        and len(word) >= 4
    ]
    return filtered_words[:n]


def build_cache(output_path='category_cache.json'):
    cache = {}
    for category in dataset['Category'].unique():
        cache[category] = get_skills_for_category(category)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(cache, f, indent=2)


if __name__ == '__main__':
    build_cache()
    print('Saved skill cache to category_cache.json')
