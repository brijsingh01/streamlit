import matplotlib
import pandas as pd
import seaborn as sns
import streamlit as st
import matplotlib.pyplot as plt
from .utils import upload_dataset

# Set up Matplotlib
matplotlib.use("Agg")
fig, ax = plt.subplots()
matplotlib.rcParams.update({"font.size": 8})
st.set_option("deprecation.showPyplotGlobalUse", False)

# Function to get categorical columns with low unique values
def categorical_column(df, max_unique_values=15):
    categorical_column_list = []
    for column in df.columns:
        if df[column].nunique() < max_unique_values:
            categorical_column_list.append(column)
    return categorical_column_list

def display_column_description():
    st.subheader("Column Descriptions")
    st.markdown("1. **PassengerId**: A unique identifier assigned to each passenger.")
    st.markdown("2. **Survived**: Indicates whether the passenger survived (1) or did not survive (0).")
    st.markdown("3. **Pclass**: Represents the passenger class (1st, 2nd, or 3rd).")
    st.markdown("4. **Name**: The name of the passenger.")
    st.markdown("5. **Sex**: The gender of the passenger (male or female).")
    st.markdown("6. **Age**: The age of the passenger.")
    st.markdown("7. **SibSp**: Number of siblings or spouses aboard the Titanic.")
    st.markdown("8. **Parch**: Number of parents or children aboard the Titanic.")
    st.markdown("9. **Ticket**: The ticket number.")
    st.markdown("10. **Fare**: The amount of money paid for the ticket.")
    st.markdown("11. **Cabin**: Cabin number where the passenger stayed (if available).")
    st.markdown("12. **Embarked**: The port of embarkation (C = Cherbourg, Q = Queenstown, S = Southampton).")


# Separate functions for each type of visualization
def visualize_survival_distribution(df):
    # Visualize survival distribution using a pie chart
    st.subheader('Survival Distribution')
    survived_counts = df['Survived'].value_counts()
    fig_survival = plt.figure(figsize=(6, 6))
    plt.pie(survived_counts, labels=['Not Survived', 'Survived'], autopct='%1.1f%%', startangle=90, colors=['#FF9999', '#66B2FF'])
    plt.title('Survival Distribution')
    st.pyplot(fig_survival)

def visualize_age_distribution(df):
    # Visualize age distribution using a histogram
    st.subheader('Age Distribution')
    fig_age = plt.figure(figsize=(8, 6))
    sns.histplot(df['Age'], bins=20, kde=True, color='skyblue')
    plt.title('Age Distribution')
    plt.xlabel('Age')
    plt.ylabel('Frequency')
    st.pyplot(fig_age)

def visualize_passenger_class_distribution(df):
    # Visualize passenger class distribution using a bar chart
    st.subheader('Passenger Class Distribution')
    pclass_counts = df['Pclass'].value_counts().sort_index()
    fig_pclass = plt.figure(figsize=(8, 6))
    pclass_counts.plot(kind='bar', color=['#FFD700', '#C0C0C0', '#CD7F32'])
    plt.title('Passenger Class Distribution')
    plt.xlabel('Passenger Class')
    plt.ylabel('Count')
    plt.xticks(rotation=0)
    st.pyplot(fig_pclass)

def visualize_embarked_distribution(df):
    # Visualize embarked distribution using a bar chart
    st.subheader('Embarked Distribution')
    embarked_counts = df['Embarked'].value_counts().sort_index()
    fig_embarked = plt.figure(figsize=(8, 6))
    embarked_counts.plot(kind='bar', color=['#FFD700', '#C0C0C0', '#CD7F32'])
    plt.title('Embarked Distribution')
    plt.xlabel('Embarked Port')
    plt.ylabel('Count')
    plt.xticks(rotation=0)
    st.pyplot(fig_embarked)

def visualize_family_size_distribution(df):
    # Visualize family size distribution using a bar chart
    st.subheader('Family Size Distribution')
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    family_size_counts = df['FamilySize'].value_counts().sort_index()
    fig_family_size = plt.figure(figsize=(10, 6))
    family_size_counts.plot(kind='bar', color='skyblue')
    plt.title('Family Size Distribution')
    plt.xlabel('Family Size')
    plt.ylabel('Count')
    plt.xticks(rotation=0)
    st.pyplot(fig_family_size)

def eda(df):
    # Explore Dataset options
    explore_dataset_option = st.checkbox("Explore Dataset")

    if explore_dataset_option:
        with st.beta_expander("Explore Dataset Options", expanded=True):
            show_dataset_summary_option = st.checkbox("Show Datset Summary")
            if show_dataset_summary_option:
                # Display column descriptions
                display_column_description()

            show_dataset = st.checkbox("Show Dataset")
            if show_dataset:
                number = st.number_input("Number of rows to view", min_value=1, value=5)
                st.dataframe(df.head(number))

            show_columns_option = st.checkbox("Show Columns Names")
            if show_columns_option:
                st.write(df.columns)

            show_shape_option = st.checkbox("Show Shape of Dataset")
            if show_shape_option:
                st.write(df.shape)
                data_dim = st.radio("Show Dimension by ", ("Rows", "Columns"))
                if data_dim == "Columns":
                    st.text("Number of Columns")
                    st.write(df.shape[1])
                elif data_dim == "Rows":
                    st.text("Number of Rows")
                    st.write(df.shape[0])
                else:
                    st.write(df.shape)

            select_columns_option = st.checkbox("Select Column to show")
            if select_columns_option:
                all_columns = df.columns.tolist()
                selected_columns = st.multiselect("Select Columns", all_columns)
                new_df = df[selected_columns]
                st.dataframe(new_df)

            show_value_counts_option = st.checkbox("Show Value Counts")
            if show_value_counts_option:
                all_columns = df.columns.tolist()
                selected_columns = st.selectbox("Select Column", all_columns)
                st.write(df[selected_columns].value_counts())

            show_data_types_option = st.checkbox("Show Data types")
            if show_data_types_option:
                st.text("Data Types")
                st.write(df.dtypes)

            show_summary_option = st.checkbox("Show Summary")
            if show_summary_option:
                st.text("Summary")
                st.write(df.describe().T)

            show_raw_data_option = st.checkbox('Show Raw Data')
            if show_raw_data_option:
                raw_data_rows = st.number_input("Number of Rows for Raw Data", min_value=1, value=5)
                raw_data_selection = df.head(raw_data_rows)
                selected_columns = st.multiselect("Select Columns", df.columns.tolist(), default=df.columns.tolist())
                new_df = raw_data_selection[selected_columns]
                st.dataframe(new_df)

    show_visualizations_option = st.checkbox('Show Visualizations')

    if show_visualizations_option:
        with st.beta_expander("Show Visualization Options", expanded=True):
            show_distribution_option = st.checkbox("Survival Distribution")
            if show_distribution_option:
                visualize_survival_distribution(df)
            show_distribution_option = st.checkbox("Age Distribution")
            if show_distribution_option:
                visualize_age_distribution(df)
            show_distribution_option = st.checkbox("Passenger Class Distribution")
            if show_distribution_option:
                visualize_passenger_class_distribution(df)
            show_distribution_option = st.checkbox("Embarked Distribution")
            if show_distribution_option:
                visualize_embarked_distribution(df)
            show_distribution_option = st.checkbox("Family Size Distribution")
            if show_distribution_option:
                visualize_family_size_distribution(df)

def app():
    st.title("Data Explorer")
    st.subheader("Explore Dataset")
    upload_dataset(eda)

if __name__ == "__main__":
    app()
