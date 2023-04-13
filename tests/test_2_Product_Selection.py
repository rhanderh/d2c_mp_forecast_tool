import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from d2c_mp_sales_forecaster.pages._2_Product_Selection import (
    get_len_history,
    get_similarities,
    check_history_needed,
    match_product,
    fill_missing,
    merge_regressors,
    append_history,
    prepare_non_appended,
)

# Sample data for testing purposes
sales_data = pd.read_csv("./select_product/sales_data.csv", 
                    header = 0,
                    names = ['product','ds','y'],
                    dtype={'product':'object','date':'object', 'y':'object'},
                    parse_dates=['ds'],
                    infer_datetime_format=True
                    )
sales_data['y'] = sales_data['y'].str.replace(',','')
sales_data['y'] = sales_data['y'].astype('int64')

#sample encoded master data
master_data_enc = pd.read_csv("./select_product/master_data_enc.csv", index_col=0)
master_data_enc.index = master_data_enc.index.map(str)

#sample lag data
lag_regressors = pd.read_csv("./select_product/lag_data.csv", 
                        header = 0,
                        parse_dates=['ds'],
                        infer_datetime_format=True
                        )
#Set datetime interval to day and ensure timeseries complete by day
lag_regressors['ds'] = pd.to_datetime(lag_regressors['ds'])
lag_regressors = lag_regressors.set_index('ds')
lag_regressors = lag_regressors.resample('1D').ffill().reset_index()
#Remove commas and cast all other columns as float64 data type
for col in lag_regressors.columns:
    if col != 'ds':
        if lag_regressors[col].dtype == 'object':
            lag_regressors[col] = lag_regressors[col].str.replace(',', '')
        lag_regressors[col] = lag_regressors[col].astype('float64')
lag_data = merge_regressors(sales_data, lag_regressors)



def test_get_len_history():
    sales_minmax_dates = get_len_history(sales_data)
    assert isinstance(sales_minmax_dates, pd.DataFrame)
    assert "len_hist" in sales_minmax_dates.columns

def test_get_similarities():
    similarity_df = get_similarities(master_data_enc)
    assert isinstance(similarity_df, pd.DataFrame)
    assert len(similarity_df) == len(master_data_enc)

def test_check_history_needed():
    sales_minmax_dates = get_len_history(sales_data)
    product_fcst = sales_data["product"].iloc[0]
    min_append_hist_req, cutoff_date = check_history_needed(sales_minmax_dates, product_fcst)
    assert isinstance(min_append_hist_req, int)
    assert isinstance(cutoff_date, np.datetime64)

def test_match_product():
    similarity_df = get_similarities(master_data_enc)
    sales_minmax_dates = get_len_history(sales_data)
    product_fcst = sales_data["product"].iloc[0]
    _, cutoff_date = check_history_needed(sales_minmax_dates, product_fcst)
    matched = match_product(similarity_df, sales_minmax_dates, product_fcst, cutoff_date, 1)
    assert isinstance(matched, np.ndarray)
    assert len(matched) > 0

def test_fill_missing():
    sales_history = sales_data.loc[sales_data['product'] == sales_data["product"].iloc[0]]
    filled_history = fill_missing(sales_history)
    assert isinstance(filled_history, pd.DataFrame)
    assert not filled_history['y'].isna().any()

def test_merge_regressors():
    sales_history = sales_data.loc[sales_data['product'] == sales_data["product"].iloc[0]]
    filled_history = fill_missing(sales_history)
    merged = merge_regressors(filled_history, lag_regressors)
    assert isinstance(merged, pd.DataFrame)
    assert set(lag_regressors.columns).issubset(merged.columns)

def test_append_history():
    product_fcst = sales_data["product"].iloc[0]
    sales_minmax_dates = get_len_history(sales_data)
    similarity_df = get_similarities(master_data_enc)
    _, cutoff_date = check_history_needed(sales_minmax_dates, product_fcst)
    matched = match_product(similarity_df, sales_minmax_dates, product_fcst, cutoff_date)
    appended_history, appended_history_product = append_history(sales_data, matched, product_fcst, lag_regressors)
    assert isinstance(appended_history, pd.DataFrame)
    assert isinstance(appended_history_product, pd.DataFrame)

def test_prepare_non_appended():
    product_fcst = sales_data["product"].iloc[0]
    forecast_df, forecast_df_product = prepare_non_appended(sales_data, product_fcst, lag_data)
    assert isinstance(forecast_df, pd.DataFrame)
    assert isinstance(forecast_df_product, pd.DataFrame)
    assert set(forecast_df.columns).issubset(forecast_df_product.columns)

if __name__ == "__main__":
    pytest.main(["-v", "-k", "test_2_product_selection.py"])
