import pytest
import pandas as pd
import streamlit as st
from io import StringIO
from unittest.mock import MagicMock
from d2c_mp_sales_forecaster.pages._1_Load_Data import LoadData

class TestClassLoadData:

    @pytest.fixture(autouse=True)
    def set_up(self):
        with MagicMock() as st.session_state:
            yield
    
    @pytest.fixture
    def sample_sales_size_data(self):
        data = "product,size,ds,y\nA,Small,2022-01-01,100\nA,Medium,2022-01-01,200\nB,Small,2022-01-01,150\nB,Large,2022-01-01,300\nA,Small,2022-01-02,120\nA,Medium,2022-01-02,180\nB,Small,2022-01-02,140\nB,Large,2022-01-02,310"
        return StringIO(data)

    @pytest.fixture
    def sample_master_data(self):
        data = "product,attribute\nA,Red\nB,Blue"
        return StringIO(data)

    @pytest.fixture
    def sample_events_data(self):
        data = "event,ds,lower_window,upper_window\nNew_Year,2022-01-01,-1,1\nEaster,2022-04-10,-1,1"
        return StringIO(data)

    @pytest.fixture
    def sample_lag_data(self):
        data = "ds,lag1\n2022-01-01,10\n2022-01-02,12"
        return StringIO(data)

    def test_upload_sales_size_data(self, sample_sales_size_data):
        ld = LoadData()
        all_sales_data, all_sales_size_data = ld.upload_sales_size_data_func(sample_sales_size_data)
        assert isinstance(all_sales_data, pd.DataFrame)
        assert isinstance(all_sales_size_data, pd.DataFrame)

    def test_upload_master_data(self, sample_master_data):
        ld = LoadData()
        master_data, master_data_enc = ld.upload_master_data_func(sample_master_data)
        assert isinstance(master_data, pd.DataFrame)
        assert isinstance(master_data_enc, pd.DataFrame)

    def test_upload_events_data(self, sample_events_data):
        ld = LoadData()
        events_df = ld.upload_events_data_func(sample_events_data)
        assert isinstance(events_df, pd.DataFrame)

    def test_upload_lag_data(self, sample_lag_data):
        ld = LoadData()
        lag_data = ld.upload_lag_data_func(sample_lag_data)
        assert isinstance(lag_data, pd.DataFrame)