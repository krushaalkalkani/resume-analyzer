import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv(
    'Resume.csv')

# print(dataset.head())
# print(dataset.nunique())
# print(dataset.value_counts())
print(dataset['Category'].value_counts())

# Add a horizontal bar chart to explore.py using matplotlib to visualize that category distribution.
category_counts = dataset['Category'].value_counts()
plt.figure(figsize=(20, 8))
category_counts.plot(kind='barh')
plt.title('Number of Resumes per Category')
plt.xlabel('Number of Resumes')
plt.ylabel('Job Category')
plt.show()
plt.tight_layout()
plt.show()
