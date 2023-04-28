# Imports
import streamlit as st
import itertools
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import date
from datetime import timedelta

# Page config
st.set_page_config(page_title="D2C, Dropship, Marketplace Forecasting")

st.header("Step 1: Upload datasets for forcasting.")


def upload_sales_size_data(uploaded_file) -> tuple:
    """
    Upload, format, and cleanse product sales data by size and day.

    Args:
        uploaded_file: A CSV file containing product sales data by size and day.

    Returns:
        A tuple of two DataFrames:
            1. all_sales_data: Aggregated product sales data.
            2. all_sales_size_data: Product sales data with proper typing and formatting.
    """
    with st.spinner("Uploading file..."):
        all_sales_size_data = pd.read_csv(uploaded_file,
                                          header=0,
                                          names=['product', 'size', 'ds', 'y'],
                                          dtype={
                                              'product': 'object', 'size': 'object', 'date': 'object', 'y': 'object'},
                                          parse_dates=['ds'],
                                          infer_datetime_format=True
                                          )
        all_sales_size_data = all_sales_size_data[all_sales_size_data['size'] != 'Result']
        all_sales_size_data['y'] = all_sales_size_data['y'].str.replace(
            ',', '')
        all_sales_size_data['y'] = all_sales_size_data['y'].astype('int64')
        if 'all_sales_size_data' not in st.session_state:
            st.session_state['all_sales_size_data'] = all_sales_size_data
        else:
            st.session_state['all_sales_size_data'] = all_sales_size_data
        # Group and save a version of the dataframe aggregated to the product level
        all_sales_data = all_sales_size_data.groupby(
            ['product', 'ds']).agg({'y': 'sum'})
        all_sales_data.reset_index(inplace=True)
        if 'all_sales_data' not in st.session_state:
            st.session_state['all_sales_data'] = all_sales_data
        else:
            st.session_state['all_sales_data'] = all_sales_data
    st.success("File upload complete!")
    return all_sales_data, all_sales_size_data


def upload_master_data(uploaded_md_file) -> tuple:
    """
    Upload product master data.

    Args:
        uploaded_md_file: A CSV file containing product master data. The first column should be the unique product key.

    Returns:
        A tuple of two DataFrames:
            1. master_data: The uploaded product master data.
            2. master_data_enc: The encoded product master data.
    """
    with st.spinner("Uploading file..."):
        master_data = pd.read_csv(uploaded_md_file, index_col=0)
        master_data.index.rename('product', inplace=True)
        master_data.index = master_data.index.map(str)
        # Econde the master data --> break this into a separate optional function later.
        master_data_enc = pd.get_dummies(master_data)
        master_data_enc = master_data_enc.fillna(0)
        if 'master_data' not in st.session_state:
            st.session_state['master_data'] = master_data
        else:
            st.session_state['master_data'] = master_data
        if 'master_data_enc' not in st.session_state:
            st.session_state['master_data_enc'] = master_data_enc
        else:
            st.session_state['master_data_enc'] = master_data_enc
    st.success("File upload complete!")
    return master_data, master_data_enc


def upload_events_data(uploaded_ev_file) -> pd.DataFrame:
    """
    Upload holidays and events data.

    Args:
        uploaded_ev_file: A CSV file containing holidays and events data.

    Returns:
        events_df: A DataFrame containing the uploaded holidays and events data.
    """
    with st.spinner("Uploading file..."):
        events_df = pd.read_csv(uploaded_ev_file,
                                header=0,
                                names=['event', 'ds',
                                       'lower_window', 'upper_window'],
                                dtype={'event': 'object', 'ds': 'object',
                                       'lower_window': 'int64', 'upper_window': 'int64'},
                                parse_dates=['ds'],
                                infer_datetime_format=True)
        if 'events_df' not in st.session_state:
            st.session_state['events_df'] = events_df
        else:
            st.session_state['events_df'] = events_df
    st.success("File upload complete!")
    return events_df


def upload_lag_data(uploaded_lag_file) -> tuple:
    """
    Upload and preprocess lag regressors data.

    Args:
        uploaded_lag_file: A CSV file containing lag regressors data.

    Returns:
        lag_data: A DataFrame containing the preprocessed lag regressors data.
    """
    with st.spinner("Uploading file..."):
        lag_data = pd.read_csv(uploaded_lag_file,
                               header=0,
                               parse_dates=['ds'],
                               infer_datetime_format=True
                               )
        # Set datetime interval to day and ensure timeseries complete by day
        lag_data['ds'] = pd.to_datetime(lag_data['ds'])
        lag_data = lag_data.set_index('ds')
        lag_data = lag_data.resample('1D').ffill().reset_index()
        # Remove commas and cast all other columns as float64 data type
        for col in lag_data.columns:
            if col != 'ds':
                if lag_data[col].dtype == 'object':
                    lag_data[col] = lag_data[col].str.replace(',', '')
                lag_data[col] = lag_data[col].astype('float64')
        if 'lag_data' not in st.session_state:
            st.session_state['lag_data'] = lag_data
        else:
            st.session_state['lag_data'] = lag_data
    st.success("File upload complete!")
    return lag_data


def upload_future_data(uploaded_future_file) -> tuple:
    """
    Upload and preprocess future regressors data.

    Args:
        uploaded_future_file: A CSV file containing future regressors data.

    Returns:
        future_data: A DataFrame containing the preprocessed future regressors data.
    """
    with st.spinner("Uploading file..."):
        future_data = pd.read_csv(uploaded_future_file,
                                  header=0,
                                  parse_dates=['ds'],
                                  infer_datetime_format=True
                                  )
        # set datetime interval to day and ensure timeseries completeness
        future_data['ds'] = pd.to_datetime(future_data['ds'])
        future_data = future_data.set_index('ds')
        future_data = future_data.resample('1D').ffill().reset_index()
        # Remove commas and cast all other columns as float64 data type
        for col in future_data.columns:
            if col != 'ds':
                if future_data[col].dtype == 'object':
                    future_data[col] = future_data[col].str.replace(',', '')
                future_data[col] = future_data[col].astype('float64')
        if 'future_data' not in st.session_state:
            st.session_state['future_data'] = future_data
        else:
            st.session_state['future_data'] = future_data
    st.success("File upload complete!")
    return future_data


@st.cache_resource
def convert_df(df):
    """
    Convert a DataFrame to a CSV string and encode it as utf-8 bytes.

    Args:
        df: A DataFrame to be converted.

    Returns:
        A CSV string of the DataFrame encoded as utf-8 bytes.
    """
    return df.to_csv().encode('utf-8')


# File Uploaders
st.subheader("a. Upload historical sales by product and size.")
uploaded_file = st.file_uploader(
    "Select Sales History by Product and Size for Upload")
st.write(uploaded_file)
if uploaded_file is not None:
    all_sales_data, lag_data = upload_sales_size_data(uploaded_file)
if 'all_sales_data' in st.session_state:
    st.write("You currently have uploaded the following data:")
    cola, colb = st.columns(2)
    with cola:
        st.subheader('Product Sales History')
        st.write(st.session_state['all_sales_data'])
    with colb:
        st.subheader('Product-Size Sales History')
        st.write(st.session_state['all_sales_size_data'])

st.subheader(
    "b. Upload product attributes for augmenting historicals with similar product sales.")
uploaded_md_file = st.file_uploader("Select product master data to upload.")
st.write(uploaded_md_file)
if uploaded_md_file is not None:
    master_data, master_data_enc = upload_master_data(uploaded_md_file)
if 'master_data' in st.session_state:
    st.write("You currently have uploaded the following data:")
    st.subheader('Product Master Data')
    st.write(st.session_state['master_data'])
    st.download_button(label="Download Encoded Master Data",
                       data=convert_df(st.session_state['master_data_enc']),
                       file_name='master_data_enc.csv',
                       mime='text/csv'
                       )

st.subheader("c. Upload holidays and special events.  (Optional)")
uploaded_ev_file = st.file_uploader(
    "Select holidays and events file to upload.")
st.write(uploaded_ev_file)
if uploaded_ev_file is not None:
    events_df = upload_events_data(uploaded_ev_file)
if 'events_df' in st.session_state:
    st.write("You currently have uploaded the following data:")
    st.subheader('Events')
    st.write(st.session_state['events_df'])

if 'all_sales_data' in st.session_state and 'all_sales_size_data' in st.session_state:
    st.subheader("d. Upload lag regressors.  (Optional)")
    uploaded_lag_file = st.file_uploader(
        "Select lag regressor data to upload.")
    st.write(uploaded_lag_file)
    if uploaded_lag_file is not None:
        lag_data = upload_lag_data(uploaded_lag_file)
    if 'lag_data' in st.session_state:
        st.write("You currently have uploaded the following data:")
        st.subheader('Lag Regressor Data')
        st.write(st.session_state['lag_data'])

    # st.subheader("e. Upload future regressors.")
    # uploaded_future_file = st.file_uploader("Select future regressor data to upload.")
    # st.write(uploaded_future_file)
    # if uploaded_future_file is not None:
    #    future_data = upload_future_data(uploaded_future_file)
    # if 'lag_data' in st.session_state:
    #    st.write("You currently have uploaded the following data:")
    #    st.subheader('Future Regressor Data')
    #    st.write(st.session_state['future_data'])
