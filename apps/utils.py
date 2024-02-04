import os
import streamlit as st
import pandas as pd

def upload_dataset(callback_function):
    """User can upload his own dataset.

    Args:
        callback_function (function): function should be called after dataset is uploaded.
    """
    data = st.file_uploader("Upload a Dataset", type=["csv", "txt"])
    if data is not None:
        data.seek(0)
        st.success("Uploaded Dataset Successfully")
        df = pd.read_csv(data)
        df.dropna(inplace=True)
        callback_function(df)  # Execute callback function
