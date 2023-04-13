import pytest
import pandas as pd
from io import StringIO
from d2c_mp_sales_forecaster.pages._1_Load_Data import (
    upload_sales_size_data,
    upload_master_data,
    upload_events_data,
    upload_lag_data,
    upload_future_data,
)

class TestClassLoadData:
    
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

    @pytest.fixture
    def sample_future_data(self):
        data = "ds,future1\n2022-01-03,15\n2022-01-04,18"
        return StringIO(data)

    def test_upload_sales_size_data(self, sample_sales_size_data):
        all_sales_data, all_sales_size_data = upload_sales_size_data(sample_sales_size_data)
        assert isinstance(all_sales_data, pd.DataFrame)
        assert isinstance(all_sales_size_data, pd.DataFrame)

    def test_upload_master_data(self, sample_master_data):
        master_data, master_data_enc = upload_master_data(sample_master_data)
        assert isinstance(master_data, pd.DataFrame)
        assert isinstance(master_data_enc, pd.DataFrame)

    def test_upload_events_data(self, sample_events_data):
        events_df = upload_events_data(sample_events_data)
        assert isinstance(events_df, pd.DataFrame)

    def test_upload_lag_data(self, sample_lag_data):
        lag_data = upload_lag_data(sample_lag_data)
        assert isinstance(lag_data, pd.DataFrame)

    def test_upload_future_data(self, sample_future_data):
        future_data = upload_future_data(sample_future_data)
        assert isinstance(future_data, pd.DataFrame)
