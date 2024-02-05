# Streamlit Demo

## Streamlit Multipage Web App Example

This project demonstrates how to quickly create a multipage web app using Streamlit.

## Installation

Make sure you have Python 3.7.9 installed. You can install the required dependencies using the following:

pip install -r requirements.txt



## Usage
Explore the different pages of the web app to see how Streamlit makes it simple to create a multipage interface. The app includes pages for data exploration (EDA) and machine learning modeling.

## Key Steps
Loading the Data: The Titanic dataset is loaded into a Pandas DataFrame for analysis and modeling.

Data Preprocessing: The notebook covers essential preprocessing steps, including handling missing values, feature engineering, and converting categorical variables into numerical format.

Train-Test Split: The data is split into training and testing sets to assess the model's performance.

Feature Scaling: Standardization is applied to scale the features, ensuring consistent model training.

Logistic Regression Model: A logistic regression model is trained using scikit-learn's LogisticRegression class.

Model Evaluation: The notebook evaluates the model using metrics such as accuracy, confusion matrix, and classification report.

Model Serialization: The trained model is saved as a pickle file (logistic_regression_model.pkl) for future use or deployment.

## How to Use
Open the Jupyter Notebook (model_creation_notebook.ipynb) in your Jupyter environment.
Execute the notebook cells sequentially to perform each step.
Review the printed evaluation metrics and the saved model file.
For detailed code explanations and visualizations, refer to the notebook itself.

## File Structure
The project is organized into the following structure:
```
project-root/
│
├── apps/
│   ├── eda.py
│   ├── models.py
│   ├── utils.py
│   ├── __init__.py
│
├── datasets/
│   ├── titanic.csv
│
├── images/
│   └── titanic.jpg
│
├── models/
│   ├── logistic_regression_model.pkl
│
├── app.py
├── multiapp.py
├── create_model.ipynb
├── LICENSE
├── requirements.txt
├── README.md
```
## Dependencies
Python 3.7.9
streamlit==0.81.0
pandas==1.1.3
numpy==1.17.4
matplotlib==3.3.2
seaborn==0.11.0
scikit-learn== 0.24.2

## How to Run
Run the Streamlit app using the following command:

streamlit run app.py

Visit http://localhost:8501 in your web browser to interact with the multipage web app.

![](https://github.com/brijsingh01/streamlit/blob/main/images/titanic.gif)

## Customization
Feel free to customize and extend the app based on your specific requirements. Streamlit's simplicity allows for easy modification and adaptation.

## Contributing
Contributions are welcome! If you have ideas for improvement or additional features, please open an issue or submit a pull request.

## License
This project is licensed under the MIT License.
