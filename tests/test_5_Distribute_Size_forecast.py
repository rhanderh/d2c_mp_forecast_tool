import pandas as pd
import numpy as np
from d2c_mp_sales_forecaster.pages._5_Distribute_Size_Forecast import prep_data_topdown, convert_forecast_for_reconciler, add_size_fcst_frame

def test_prep_data_topdown():
    forecast_product = 'A'
    forecast_df_product = pd.DataFrame({
        'ds': pd.date_range('2021-01-01', periods=5),
        'product': ['A'] * 5,
        'y': [0,0,0,0,0]
    })
    all_sales_size_data = pd.DataFrame({
        'ds': pd.date_range('2021-01-01', periods=5),
        'product': ['A'] * 5,
        'size': ['S', 'M', 'L', 'XL', 'XXL'],
        'y': np.random.randint(1, 100, size=5)
    })

    Y_df, S_df, tags = prep_data_topdown(forecast_product, forecast_df_product, all_sales_size_data)

    assert isinstance(Y_df, pd.DataFrame)
    assert isinstance(S_df, pd.DataFrame)
    assert isinstance(tags, list)

def test_convert_forecast_for_reconciler():
    forecast = pd.DataFrame({
        'ds': pd.date_range('2021-01-01', periods=10),
        'yhat1': np.random.rand(10),
        'yhat2': np.random.rand(10),
    })
    horizon = 3

    result = convert_forecast_for_reconciler(forecast, horizon)

    assert isinstance(result, pd.DataFrame)
    assert 'yhat' in result.columns
    assert 'unique_id' in result.columns

def test_add_size_fcst_frame():
    fcst_converted = pd.DataFrame({
        'ds': pd.date_range('2021-01-01', periods=5),
        'unique_id': ['A'] * 5,
        'yhat': np.random.rand(5)
    })
    S_df = pd.DataFrame({
        'S': np.random.rand(5),
        'M': np.random.rand(5),
        'L': np.random.rand(5),
    })

    result = add_size_fcst_frame(fcst_converted, S_df)

    assert isinstance(result, pd.DataFrame)
    assert 'ds' in result.columns
    assert 'yhat' in result.columns

