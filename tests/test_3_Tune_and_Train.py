import pytest
import pandas as pd
import streamlit as st
from datetime import datetime
from d2c_mp_sales_forecaster.pages._3_Tune_and_Train import fit_complete, cross_val_tune, is_events

@pytest.fixture
def sample_data():
    sales_data = pd.read_csv("./select_product/sales_data.csv", 
                    header = 0,
                    names = ['product','ds','y'],
                    dtype={'product':'object','date':'object', 'y':'object'},
                    parse_dates=['ds'],
                    infer_datetime_format=True
                    )
    sales_data['y'] = sales_data['y'].str.replace(',','')
    sales_data['y'] = sales_data['y'].astype('int64')
    sales_data = sales_data.loc[sales_data['product'] == "10009814"]
    sales_data.drop(columns=['product'], inplace=True)
    sales_data['y'] = sales_data['y'].astype('int64')
    sales_data.ffill(inplace=True)
    return pd.DataFrame(sales_data)

@pytest.fixture
def sample_params():
    return {
        'seasonality_mode': 'additive',
        'loss_func': 'MSE',
        'n_changepoints': 10,
        'changepoints_range': 0.5,
        'normalize': 'soft',
        'num_hidden_layers': 0,
        'n_lags': 7,
        'n_forecasts': 7,
        'drop_missing':True
    }

if 'forecast_df' not in st.session_state:
    st.session_state['forecast_df'] = pd.DataFrame()

def test_fit_complete(sample_data, sample_params):
    model = fit_complete(params=sample_params, df=sample_data)
    assert model is not None, "Failed to fit the model"

def test_cross_val_tune(sample_data, sample_params):
    param_grid = {k: [v] for k, v in sample_params.items()}
    best_metrics, best_params = cross_val_tune(params=param_grid, df=sample_data, forecast_product="10009814")
    
    assert best_metrics is not None, "Failed to get best_metrics from cross_val_tune"
    assert best_params is not None, "Failed to get best_params from cross_val_tune"

def test_is_events():
    assert is_events() is None, "Failed to return None when no events data is available"
