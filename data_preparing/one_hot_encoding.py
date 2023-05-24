import pandas as pd

# Load train data
df = pd.read_csv('dataset_sample/train_sample.csv')

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

# One hot encoding: Filling and turning categories into numbers
df = pd.get_dummies(df)
df = df.sample(frac=1)

# Save prepared data into csv
df.to_csv('manipulated_datasets/one_hot_encoding.csv', index=False)