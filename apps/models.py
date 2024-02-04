import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os

# Function to get user data through Streamlit widgets
def get_user_data() -> pd.DataFrame:
    user_data = {}

    # Using sliders for numerical features
    user_data['Age'] = st.slider(
        label='Age:',
        min_value=0,
        max_value=100,
        value=20,
        step=1
    )

    user_data['Fare'] = st.slider(
        label='How much did your ticket cost you?:',
        min_value=0,
        max_value=300,
        value=80,
        step=1
    )

    user_data['SibSp'] = st.slider(
        label='Number of siblings and spouses aboard:',
        min_value=0,
        max_value=15,
        value=3,
        step=1
    )

    user_data['Parch'] = st.slider(
        label='Number of parents and children aboard:',
        min_value=0,
        max_value=15,
        value=3,
        step=1
    )
    # Using radio buttons for categorical features
    col1, col2, col3 = st.beta_columns(3)

    user_data['Pclass'] = col1.radio(
        label='Ticket class:',
        options=['1st', '2nd', '3rd'],
        index=0
    )

    user_data['Sex'] = col2.radio(
        label='Sex:',
        options=['Man', 'Woman'],
        index=0
    )

    user_data['Embarked'] = col3.radio(
        label='Port of Embarkation:',
        options=['Cherbourg', 'Queenstown', 'Southampton'],
        index=1
    )

    # Turn dict 'values' to a list before turning the dict into a DataFrame
    for k in user_data.keys():
        user_data[k] = [user_data[k]]

    # Convert dictionary 'values' to a DataFrame
    df = pd.DataFrame(data=user_data)

    # Some preprocessing of the raw data from the user.
    # Follow the same data structure as in the Kaggle competition
    df['Sex'] = df['Sex'].map({'Man': 'male', 'Woman': 'female'})
    df['Pclass'] = df['Pclass'].map({'1st': 1, '2nd': 2, '3rd': 3})
    df['Embarked'] = df['Embarked'].map(
        {'Cherbourg': 'C', 'Queenstown': 'Q', 'Southampton': 'S'}
    )
    df['num_relatives'] = df['SibSp'] + df['Parch']
    return df

# Function to load the machine learning model
@st.cache
def load_model(model_file_path: str):
    with st.spinner("Loading model..."):
        with open(model_file_path, 'rb') as file:
            model = pickle.load(file)
    return model

# Function to preprocess user data before making predictions
def preprocess_user_data(df):
    # Print information about missing values before processing
    print('Missing values before processing:')
    print(df.isnull().sum())

    # Preprocessing
    # Handle missing values
    # Fill missing values and perform feature engineering
    df['Age'].fillna(df['Age'].mean(), inplace=True)
    df['Fare'].fillna(df['Fare'].mean(), inplace=True)
    df['SibSp'].fillna(0, inplace=True)
    df['Parch'].fillna(0, inplace=True)

    # Feature engineering
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    df['IsAlone'] = 1
    df['IsAlone'].loc[df['FamilySize'] > 1] = 0

    # Convert categorical variables to numerical

    df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
    df['Embarked'] = df['Embarked'].map(
        {'Cherbourg': 'C', 'Queenstown': 'Q', 'Southampton': 'S'}
    )

    # One-hot encode 'Embarked' column
    df = pd.get_dummies(df, columns=['Embarked'], drop_first=True)

    # Ensure only one 'Embarked' column is set to 1
    embarked_cols = ['Embarked_C', 'Embarked_Q', 'Embarked_S']
    df[embarked_cols] = np.zeros((len(df), len(embarked_cols)))  # Set all to 0 initially

    # Find the correct 'Embarked' column based on the user input
    selected_embarked_col = df.filter(regex='Embarked_.*').columns[0]
    selected_embarked = selected_embarked_col.split('_')[1]
    df[selected_embarked_col] = 1
    print('intermediate')
    print(df)
    # Reorder columns to match the desired structure
    features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'FamilySize', 'IsAlone'] + embarked_cols
    df = df[features]

    # Print information about missing values after processing
    print('Missing values after processing:')
    print(df.isnull().sum())
    return df

# Streamlit app for machine learning modeling
def app():
    st.title("Machine Learning Modelling")

    model_name = 'logistic_regression_model.pkl'

    this_file_path = os.path.abspath(__file__)
    current_working_directory = os.getcwd()
    project_path = '/'.join(this_file_path.split('/')[:-2])

    st.header(body='Would you have survived the Titanic?🚢')

    df_user_data = get_user_data()
    df_user_data = preprocess_user_data(df_user_data)

    model = load_model(model_file_path=project_path + 'models/' + model_name)
    prob = model.predict_proba(df_user_data)[0][1]
    prob = int(prob * 100)

    emojis = ["😕", "🙃", "🙂", "😀"]
    survival_state = min(prob // 25, 3)

    st.write('')
    st.title(f'{prob}% chance to survive! {emojis[survival_state]}')

    # Map survival state for use in messages
    if survival_state == 0:
        state_message = "Oh no! Shark snacks incoming! 🚨🦈 Better start practicing your shark dance!"
    elif survival_state == 1:
        state_message = "Hold your breath! Looks like a swimming challenge awaits! 🌊🏊‍♂️"
    elif survival_state == 2:
        state_message = "Great job! You're navigating the Titanic waters like a pro! 🌟🗺️ Don't forget your compass!"
    else:
        state_message = 'Hooray! Smooth sailing ahead! 🚢🌅 You can start planning your victory dance! 💃🕺'

    st.info(state_message)
    st.image(project_path + 'images/titanic.jpg')

if __name__ == "__main__":
    app()
