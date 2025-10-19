import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier  # Or any model you prefer
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE


# Use the read_csv() function to load the CSV file into a DataFrame
df = pd.read_csv('NSF_Results_v3.xlsx - Gait_Variability (1).csv')
df2 = pd.read_csv('Soangra_Data_Info_Shared_V2.xlsx - Copy of Gait_Anthropometry_Test_Data.csv')
df3 = pd.read_csv('Soangra_Data_Info_Shared_V2.xlsx - Copy of Gait_Anthropometry_Train_Data.csv')
combined_df = pd.concat([df2, df3], ignore_index=True)

# Display the first few rows of the DataFrame to verify the import


# Accessing specific columns or rows:
# print(df['ColumnName']) # Access a specific column
# print(df.iloc[0]) # Access the first row
no_data_indices = df[df['GCTime_mean'] == ' -'].index.tolist()
df = df[df['GCTime_mean'] != ' -']
nan_row_indices = df[df['Faller'].isna()].index.tolist()
df = df.dropna(subset=['Faller'])
df.iloc[:, 3:58] = df.iloc[:, 3:58].astype(float)


id_numbers = df['ID']
labels = df['Faller']

df1 = pd.merge(df, combined_df, on='ID', how='outer')
df1 = df1.drop(columns=['Gender'])
df1.to_csv("combined_output.csv", index=False)

df_shuffled = df1.sample(frac=1, random_state=42).reset_index(drop=True)
split_index = int(0.8 * len(df_shuffled))
df_train = df_shuffled[:split_index]
df_test = df_shuffled[split_index:]

train_labels = df_train['Faller']
test_labels = df_test['Faller']

train_data = df_train.iloc[:, 2:]
test_data = df_test.iloc[:, 2:]

train_data_encoded = pd.get_dummies(train_data)
test_data_encoded = pd.get_dummies(test_data)

train_data_encoded, test_data_encoded = train_data_encoded.align(test_data_encoded, join='left', axis=1, fill_value=0)

smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(train_data_encoded, train_labels)

print('Training...')
model = RandomForestClassifier(random_state=42, class_weight='balanced')
model.fit(X_resampled, y_resampled)
print('Done training')

print('Testing...')
y_pred = model.predict(test_data_encoded)
accuracy = accuracy_score(test_labels, y_pred)
print('Done testing')

print(f"Accuracy: {accuracy:.2f}")
print(confusion_matrix(test_labels, y_pred))
print(classification_report(test_labels, y_pred))