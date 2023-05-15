import streamlit as st
import itertools
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import date
from datetime import timedelta
from neuralprophet import NeuralProphet, set_log_level, set_random_seed, utils
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics.pairwise import cosine_similarity
from hierarchicalforecast.core import HierarchicalReconciliation
from hierarchicalforecast.methods import TopDown
from hierarchicalforecast.evaluation import HierarchicalEvaluation
from hierarchicalforecast.utils import aggregate
from d2c_mp_sales_forecaster.utils.db_manager import PostgreSQLManager


class DistributeSizeForecast:

    def __init__(self):
        # Initialize state-dependent streamlit variables
        # initialize session_state vars
        if 'forecast' not in st.session_state:
            st.session_state['forecast'] = pd.DataFrame
        if 'fcst_session_name' not in st.session_state:
            st.session_state['fcst_session_name'] = ""
        if 'fcst_session_id' not in st.session_state:
            st.session_state['fcst_session_id'] = None
        if 'user_tag' not in st.session_state:
            st.session_state['user_tag'] = ""
        if 'Y_hat_sizes' not in st.session_state:
            st.session_state['Y_hat_sizes'] = pd.DataFrame
        if 'Y_df' not in st.session_state:
            st.session_state['Y_df'] = pd.DataFrame
        if 'forecast_product' not in st.session_state:
            st.session_state['forecast_product'] = None
        if 'forecast_df_product' not in st.session_state:
            st.session_state['forecast_df_product'] = pd.DataFrame
        if 'all_sales_size_data' not in st.session_state:
            st.session_state['all_sales_size_data'] = pd.DataFrame()
        if 'horizon' not in st.session_state:
            st.session_state['horizon'] = int
        # Layout page
        st.set_page_config(page_title="D2C, Dropship, Marketplace Forecasting")
        st.header("Step 5: Distribute forecast to product sizes.")
        if st.session_state['forecast'].empty:
            st.write("Please use the preceding pages to upload data, train a model, and form a product-level prediction before proceeding to size distribution.")
        else:
            method = st.selectbox("Choose method of calculating size curve from historical data:", [
                'average_proportions', 'proportion_averages'], index=0)
            size_dist_check = st.checkbox("Distribute your forecast by size?",
                                          help="Distribute forecast to a size-curve using top-down average historical proportions.")
            if size_dist_check:
                # Execute the reconciliation
                st.session_state['Y_hat_sizes'], st.session_state['Y_df'] = self.size_dist_recon(st.session_state['forecast_product'],
                                                                                                 st.session_state['forecast_df_product'],
                                                                                                 st.session_state['all_sales_size_data'],
                                                                                                 st.session_state['forecast'],
                                                                                                 st.session_state['horizon'],
                                                                                                 method)
                st.write(st.session_state['Y_hat_sizes'])
                st.download_button(label="Download Forecast CSV",
                                   data=self.convert_df(
                                       st.session_state['Y_hat_sizes']),
                                   file_name='forecast_bysize_df_' +
                                   st.session_state['forecast_product']+'.csv',
                                   mime='text/csv')
                with st.expander("Size Distribution Visualizations"):
                    # Check chart 1
                    size_grp = st.session_state['Y_hat_sizes'].loc[:, ('ds', 'unique_id', 'fcst_qty')].groupby(
                        'unique_id').sum().sort_values(by='fcst_qty')
                    st.subheader("Forecast Quantity by Size")
                    st.pyplot(size_grp.plot(kind='barh').figure)
                    # Check chart 2
                    Y_df_comp = st.session_state['Y_df'].reset_index()
                    Y_df_comp = Y_df_comp[Y_df_comp['unique_id']
                                          != st.session_state['forecast_product']]
                    Y_df_comp.rename(columns={"y": "qty"}, inplace=True)
                    st.subheader("Training Set Quantity Distribution")
                    st.pyplot(Y_df_comp.loc[:, ('ds', 'unique_id', 'qty')].groupby(
                        'unique_id').sum().sort_values(by='qty').plot(kind='barh').figure)
                with st.expander("Save Results to Database"):
                    if st.session_state['fcst_session_id'] is not None:
                        self.save_size_fcst_db(st.session_state['Y_hat_sizes'],
                                               st.session_state['fcst_session_id'],
                                               st.session_state['fcst_session_name'], st.session_state['user_tag'])
                    else:
                        st.write(
                            'Please save matching product-level forecast to DB first for referential integrity.')

    @st.cache_data
    def prep_data_topdown(_self, forecast_product: str, forecast_df_product: pd.DataFrame, all_sales_size_data: pd.DataFrame) -> tuple:
        """
        Prepare data for top-down hierarchical forecasting.

        Args:
            forecast_product: The target product for forecasting.
            forecast_df_product: A DataFrame containing the sales history of the target product.
            all_sales_size_data: A DataFrame containing sales data for all products and sizes.

        Returns:
            A tuple containing the aggregate sales data, disaggregate sales data, and unique identifier tags.
        """
        hiers = [
            ['product'],
            ['product', 'size']
        ]
        dfd = all_sales_size_data.merge(forecast_df_product[['ds', 'product']], on=[
                                        'ds', 'product'], how='inner')
        dfd.loc[:, 'product'] = forecast_product
        dfd.reset_index(inplace=True, drop=True)
        # use hierarchicalforecast package aggregate function to prep needed dataframes
        Y_df, S_df, tags = aggregate(dfd, hiers)
        # setup historical 'fit' column since using NeuralProphet to create forecast
        Y_df.loc[:, 'yhat'] = Y_df['y']
        return Y_df, S_df, tags

    @st.cache_data
    def convert_forecast_for_reconciler(_self, forecast: pd.DataFrame, horizon: int) -> pd.DataFrame:
        """
        Convert forecast data to a format suitable for the reconciler.

        Args:
            forecast: A DataFrame containing the forecast data.
            horizon: The forecast horizon in days.

        Returns:
            A DataFrame containing the reformatted forecast data.
        """
        fcst = forecast.iloc[horizon:, :(horizon+2)].copy()
        fcst['yhat'] = fcst.filter(regex='^yhat').sum(axis=1)
        fcst['unique_id'] = st.session_state['forecast_product']
        fcst = fcst.loc[:, ('ds', 'yhat', 'unique_id')]
        return fcst

    @st.cache_data
    def add_size_fcst_frame(_self, fcst_converted: pd.DataFrame, S: pd.DataFrame) -> pd.DataFrame:
        """
        Create size-level forecast frame and add zero forecast as base forecast for size-level to the forecast data.

        Args:
            fcst_converted: A DataFrame containing the converted forecast data.
            S: A DataFrame containing the disaggregate sales data.

        Returns:
            A DataFrame containing the forecast data with added size-level forecasts.
        """
        # create template to add zero forecast as base forecast for size-level
        dates = fcst_converted['ds'].to_list()
        sizes = S.columns.to_list()
        # Create a list of tuples with all combinations of dates and sizes
        date_size_combinations = list(itertools.product(dates, sizes))
        # Create the DataFrame with 'date' and 'size' columns
        sizes_fcst_frame = pd.DataFrame(
            date_size_combinations, columns=['ds', 'unique_id'])
        sizes_fcst_frame.loc[:, 'yhat'] = 0
        # Concat with product level predictions for the full forecast frame for reconciler
        full_fcst_frame = pd.concat(
            [fcst_converted, sizes_fcst_frame], ignore_index=True)
        full_fcst_frame.index = full_fcst_frame['unique_id']
        full_fcst_frame.drop(columns=['unique_id'], inplace=True)
        full_fcst_frame.columns = ['ds', 'yhat']
        return full_fcst_frame

    @st.cache_resource
    def convert_df(_self, df):
        """
        Convert a DataFrame to a CSV string and encode it as utf-8 bytes.

        Args:
            df: A DataFrame to be converted.

        Returns:
            A CSV string of the DataFrame encoded as utf-8 bytes.
        """
        # IMPORTANT: Cache the conversion to prevent computation on every rerun
        return df.to_csv().encode('utf-8')

    def save_size_fcst_db(self, forecast_df: pd.DataFrame, fcst_session_id: int, fcst_session_name: str, user_tag: str = ""):
        """
        Save the size forecast to the database using the provided inputs.

        Args:
            forecast_df (pd.DataFrame): The size forecast dataframe.
            fcst_session_id (int): The forecast session ID.
            fcst_session_name (str): The forecast session name.
            user_tag (Optional[str], optional): The user tag. Defaults to an empty string.
        """
        pg_manager = PostgreSQLManager()
        with st.form("size_fcst_session_form"):
            st.write("UNIQUE Forecast Session Name : " + fcst_session_name)
            user_tag = st.text_input("User Tag (optional)", value=user_tag)
            date_input = st.date_input("Date")

            submit_button = st.form_submit_button(label="Save Size Forecast")

            if submit_button:
                st.session_state['user_tag'] = user_tag
                st.session_state['date_input'] = date_input.strftime(
                    "%Y-%m-%d")

                # add session id to dataframe for db entry with reference to session
                forecast_df.columns = ['unique_id', 'ds', 'yhat', 'fcst_qty']
                forecast_df['fcst_session_id'] = fcst_session_id
                size_save_success = pg_manager.insert_dataframe(
                    forecast_df, 'streamlit.product_size_fcst')

                if size_save_success:
                    st.success("Size forecast saved!")
                else:
                    st.error("Failed to save size forecast.")
        pg_manager.close_pool()

    def size_dist_recon(self, forecast_product: str, forecast_df_product: pd.DataFrame, all_sales_size_data: pd.DataFrame, forecast: pd.DataFrame, horizon: int, method: str) -> tuple:
        """
        Reconciles size distribution forecasts using a top-down approach.

        Args:
            forecast_product (str): The name of the forecasted product.
            forecast_df_product (pd.DataFrame): The forecast data for the product.
            all_sales_size_data (pd.DataFrame): The sales size data for all products.
            forecast (pd.DataFrame): The original forecast data.
            horizon (int): The forecast horizon.
            method (str): The reconciliation method.

        Returns:
            tuple: A tuple containing Y_hat_sizes and Y_df.
        """
        Y_df, S_df, tags = self.prep_data_topdown(
            forecast_product, forecast_df_product, all_sales_size_data)
        fcst = self.convert_forecast_for_reconciler(forecast, horizon)
        Y_hat_df = self.add_size_fcst_frame(fcst, S_df)
        # Reconcile the base predictions
        reconcilers = [TopDown(method)]
        hrec = HierarchicalReconciliation(reconcilers=reconcilers)
        Y_hat_rec = hrec.reconcile(
            Y_hat_df=Y_hat_df, S=S_df, tags=tags, Y_df=Y_df)
        # convert forecast to int for qty sold
        Y_hat_rec.loc[:, 'fcst_qty'] = Y_hat_rec['yhat/TopDown_method-'+method].round()
        Y_hat_rec.reset_index(inplace=True)
        Y_hat_sizes = Y_hat_rec[Y_hat_rec['unique_id']
                                != forecast_product].iloc[:, [0, 1, 3, 4]]
        return Y_hat_sizes, Y_df
    
if __name__ == "__main__":
    dsf = DistributeSizeForecast()