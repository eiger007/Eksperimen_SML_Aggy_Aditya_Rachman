import pandas as pd
from sklearn.preprocessing import StandardScaler

def load_data():
    # Mengambil data langsung dari raw public URL untuk automasi
    url = 'https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv'
    return pd.read_csv(url)

def preprocess_data(df):
    df_clean = df.copy()
    df_clean.drop(columns=['PassengerId', 'Name', 'Ticket', 'Cabin'], inplace=True)
    df_clean['Age'].fillna(df_clean['Age'].median(), inplace=True)
    df_clean['Embarked'].fillna(df_clean['Embarked'].mode()[0], inplace=True)
    df_clean.drop_duplicates(inplace=True)
    
    # Encoding manual sederhana untuk CI/CD agar tidak error
    df_clean['Sex'] = df_clean['Sex'].map({'male': 0, 'female': 1})
    df_clean = pd.get_dummies(df_clean, columns=['Embarked'], drop_first=True)
    
    scaler = StandardScaler()
    df_clean[['Age', 'Fare']] = scaler.fit_transform(df_clean[['Age', 'Fare']])
    return df_clean

if __name__ == "__main__":
    raw_data = load_data()
    clean_data = preprocess_data(raw_data)
    clean_data.to_csv("data_preprocessing.csv", index=False)
    print("Preprocessing otomatis selesai.")
