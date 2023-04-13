import pandas as pd
import numpy as np
import pytest
from datetime import datetime
from pytest_mock import MockerFixture
from neuralprophet import NeuralProphet
from d2c_mp_sales_forecaster.pages._4_Forecast_Product import convert_cols_to_int, predict, is_events

def test_convert_cols_to_int():
    df = pd.DataFrame({
        'y': [1.2, 2.3, 3.8],
        'residual1': [0.2, 0.8, 0.7],
        'step1': [4.5, 4.9, 5.1],
        'other': [1.1, 2.2, 3.3]
    })
    expected_df = pd.DataFrame({
        'y': [1, 2, 4],
        'residual1': [0, 1, 1],
        'step1': [4, 5, 5],
        'other': [1.1, 2.2, 3.3]
    })
    result_df = convert_cols_to_int(df)
    pd.testing.assert_frame_equal(result_df, expected_df)

def test_predict(mocker: MockerFixture):
    mocker.patch("d2c_mp_sales_forecaster.pages._4_Forecast_Product.st.spinner")
    mocker.patch("d2c_mp_sales_forecaster.pages._4_Forecast_Product.st.success")

    _m = NeuralProphet()
    horizon = 10
    dates = pd.date_range(start='2021-01-01', periods=100)
    y = np.random.rand(100)
    df = pd.DataFrame({"ds": dates, "y": y})
    events_df = None
    _m.fit(df=df)

    forecast = predict(_m, horizon, df, events_df)
    assert len(forecast) == 10
    assert "ds" in forecast.columns
    assert "yhat1" in forecast.columns

def test_is_events(mocker: MockerFixture):
    mocker.patch("d2c_mp_sales_forecaster.pages._4_Forecast_Product.st.session_state", {"events_df": None})
    assert is_events() is None

    test_events_df = pd.DataFrame({"ds": pd.date_range(start='2021-01-01', periods=10), "event": ["event1"] * 10})
    mocker.patch("d2c_mp_sales_forecaster.pages._4_Forecast_Product.st.session_state", {"events_df": test_events_df})
    assert is_events().equals(test_events_df)
