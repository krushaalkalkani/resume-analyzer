import pandas as pd
from collections import Counter
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
import re
import joblib
from preprocess import clean_text

model = joblib.load('model.pkl')
tfidf = joblib.load('tfidf.pkl')
label_encoder = joblib.load('label_encoder.pkl')

dataset = pd.read_csv('data/cleaned_resume.csv')

# create skill database

SKILL_DATABASE = {
    'INFORMATION-TECHNOLOGY': [
        'python', 'java', 'sql', 'aws', 'azure', 'docker', 'kubernetes', 'git',
        'javascript', 'cloud computing', 'machine learning', 'devops'
    ],
    'BUSINESS-DEVELOPMENT': [
        'sales strategy', 'client relations', 'market analysis', 'business planning',
        'negotiation', 'revenue growth', 'strategic partnerships', 'lead generation',
        'contract management', 'market research', 'competitive analysis', 'customer acquisition'
    ],
    'FINANCE': [
        'financial analysis', 'accounting', 'excel', 'budgeting', 'financial modeling',
        'investment analysis', 'risk management', 'tax planning', 'cost control',
        'payroll', 'auditing', 'financial reporting', 'accounts payable', 'cash flow', 'general ledger'
    ]
}

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
    top_words = Counter(words).most_common(n)
    return top_words


def get_skills_for_category(category, n=15):
    if category in SKILL_DATABASE:
        return SKILL_DATABASE[category]
    else:
        top_words = get_top_words(category, n * 3)
        top_words = [
            word for word, count in top_words
            if word not in ENGLISH_STOP_WORDS
            and word not in JUNK_WORDS
            and len(word) >= 4
        ]
        return top_words[:n]


# # filter row by category
# finance = dataset[dataset['Category'] == 'FINANCE']
# # Combine all resume text from that category into one big string
# finance_text = ' '.join(finance['cleaned_resume'])
# # To split text into words: .split() on a string gives you a list of words
# finance_words = finance_text.split()

# # Remove stop words
# finance_words = [word for word in finance_words if word.lower()
#                  not in ENGLISH_STOP_WORDS]

# # print(f"Total words in FINANCE category: {len(finance_words)}")
# # print(f"Unique words in FINANCE category: {len(set(finance_words))}")


# skill extractor
# if "financial reporting" in "this is a resume mentioning financial reporting and more":
    # True!
    # found_skills = [skill for skill in skills if skill in resume_text.lower()]
    # re.search(pattern, resume_text.lower())


def extract_skills(resume_text, category):
    skills = get_skills_for_category(category)
    found_skills = []
    for skill in skills:
        pattern = r'\b' + re.escape(skill) + r'\b'
        if re.search(pattern, resume_text.lower()):
            found_skills.append(skill)
    return found_skills


def find_missing_skills(user_skills, category):
    category_skills = get_skills_for_category(category)
    user_skill_set = {skill.lower() for skill in user_skills}
    missing_skills = [
        skill for skill in category_skills if skill not in user_skill_set]
    return missing_skills


def analyze_resume(resume_text):
    cleaned_resume = clean_text(resume_text)
    category = model.predict(tfidf.transform([cleaned_resume]))[0]
    category_name = label_encoder.inverse_transform([category])[0]
    extracted_skills = extract_skills(cleaned_resume, category_name)
    missing_skills = find_missing_skills(extracted_skills, category_name)
    return {
        'predicted_category': category_name,
        'skills_found': extracted_skills,
        'skills_missing': missing_skills
    }


# top_words = Counter(finance_words).most_common(30)
# print(top_words)
if __name__ == "__main__":
    # Test with a real finance resume
    real_resume = "Finance professional with 8 years of experience in financial analysis, accounting, and payroll management. Proficient in excel and financial reporting."
    result = analyze_resume(real_resume)
    print("Predicted category:", result['predicted_category'])
    print("Skills found:", result['skills_found'])
    print("Skills missing:", result['skills_missing'])
