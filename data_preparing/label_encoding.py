import os
import pandas as pd

# Load train data
df = pd.read_csv('datasets/train_transaction.csv')

# Turn strings into category values 
for label, content in df.items():
    if pd.api.types.is_string_dtype(content):
        df[label] = content.astype('category').cat.as_ordered()

# Fill missing values in numerical data
missing = {}
for label, content in df.items():
    if pd.api.types.is_numeric_dtype(content) and content.isnull().sum():
            missing[label+'_is_missing'] = pd.isnull(content)
            missing[label] = content.fillna(content.median())
            df = df.drop([label], axis=1)
df = pd.concat([df, pd.DataFrame(missing)], axis=1)

# Label encoding: Filling and turning categories into numbers
categories = {}
for label, content in df.items():
    if not pd.api.types.is_numeric_dtype(content):
        categories[label+'_is_missing'] = pd.isnull(content)
        categories[label]= pd.Categorical(content).codes + 1
        df = df.drop([label], axis=1)
df = pd.concat([df, pd.DataFrame(categories)], axis=1)

# Save prepared data into csv
directory = "manipulated_datasets"
if not os.path.exists(directory):
    os.makedirs(directory)
df.to_csv(f'{directory}/label_encoding.csv', index=False)