import pandas as pd # For data manipulation
from sklearn.ensemble import RandomForestClassifier # Imports ML model for classification

# 1. Load data
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

# 2. Explore
print(train_df.head())
print(train_df.info())
print(train_df['Survived'].value_counts())

# 3. Feature engineering (handle missing, encode)
median_age = train_df['Age'].median()
train_df['Age'] = train_df['Age'].fillna(median_age)
train_df['Sex'] = (train_df['Sex'] == 'male').astype(int)
test_df['Sex'] = (test_df['Sex'] == 'male').astype(int)

# 4. Prepare data
X = train_df[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']]
y = train_df['Survived']

# 5. Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

# 6. Evaluate
accuracy = model.score(X, y)
print(f"Training accuracy: {accuracy:.2%}")

# 7. Predict
test_X = test_df[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']]
predictions = model.predict(test_X)

# 8. Submit
submission = pd.DataFrame({
    'PassengerId': test_df['PassengerId'],
    'Survived': predictions
})
submission.to_csv('submission.csv', index=False)