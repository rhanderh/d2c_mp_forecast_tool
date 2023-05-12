import streamlit as st
import itertools
import pandas as pd
import numpy as np
import base64
import matplotlib.pyplot as plt
from datetime import date
from datetime import timedelta
from cycler import cycler
from typing import Union, Tuple
from neuralprophet import NeuralProphet, set_log_level, set_random_seed, utils, plot_forecast
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics.pairwise import cosine_similarity
from d2c_mp_sales_forecaster.utils.db_manager import PostgreSQLManager

# Page config
st.set_page_config(page_title="D2C, Dropship, Marketplace Forecasting")

@st.cache_data
def convert_cols_to_int(df):
    """
    Convert specified columns of a DataFrame to integers.

    Args:
        df: A DataFrame containing the columns to be converted.

    Returns:
        The modified DataFrame with specified columns converted to integers.
    """
    for col in df.columns:
        if col.startswith('y') or col.startswith('residual') or col.startswith('step'):
            df[col].fillna(0, inplace=True)
            df[col] = df[col].round(0).astype(int)
    return df


@st.cache_data
def predict(_m: NeuralProphet, horizon: int, df: pd.DataFrame, events_df: pd.DataFrame = None) -> tuple:
    """
    Make predictions using a trained NeuralProphet model.

    Args:
        _m: A trained NeuralProphet model.
        horizon: The forecast horizon in days.
        df: A DataFrame containing the data for prediction.
        events_df: A DataFrame containing events data (optional).

    Returns:
        A tuple containing the forecast DataFrame.
    """
    with st.spinner("Predicting with model... "):
        if events_df is not None:
            data_ev = _m.create_df_with_events(df, events_df)
            future = _m.make_future_dataframe(
                df=data_ev, events_df=events_df, periods=horizon)
        else:
            future = _m.make_future_dataframe(df=df, periods=horizon)
        # convert predict output to integers as this application is focused on forecasting QTY which cannot be decimal
        forecast = convert_cols_to_int(_m.predict(future))
        # raw_forecast = convert_cols_to_int(_m.predict(future, raw=True))
    st.success("Prediction complete", icon="✅")
    return forecast  # , raw_forecast


def is_events():
    """
    Check if events data is available in the session state.

    Returns:
        The events data if available, otherwise None.
    """
    if 'events_df' in st.session_state:
        return st.session_state['events_df']
    else:
        return None


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


def custom_plot_np_components(model, forecast, components=None):
    """
    Create custom component plots for a NeuralProphet model's forecast.

    Args:
        model: A trained NeuralProphet model.
        forecast: A DataFrame containing the model's forecast.
        components: A list of components to be plotted (default: None, which includes 'trend', 'season_yearly', 'season_weekly', and 'AR').

    Returns:
        A matplotlib figure containing the component plots.
    """
    if components is None:
        components = ['trend', 'season_yearly', 'season_weekly', 'AR']

    n_plots = len(components)
    fig, axes = plt.subplots(n_plots, 1, figsize=(10, 4 * n_plots))
    if n_plots == 1:
        axes = [axes]

    for idx, component in enumerate(components):
        ax = axes[idx]
        if component == 'AR':
            ar_data = model.model.ar_weights.detach().numpy()
            mean_ar_weights = ar_data.mean(axis=1).flatten()
            time_steps = list(range(1, len(mean_ar_weights) + 1))
            ax.bar(time_steps, mean_ar_weights, label=component)
        elif component == 'season_weekly':
            forecast['day_of_week'] = forecast['ds'].dt.dayofweek
            weekly_data = forecast.groupby('day_of_week')[
                'season_weekly'].agg(np.mean)
            ax.plot(weekly_data.index, weekly_data.values, label=component)
            ax.set_xticks(list(range(7)))
            ax.set_xticklabels(
                ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])
        elif component == 'season_yearly':
            forecast['day_of_year'] = forecast['ds'].dt.dayofyear
            yearly_data = forecast.groupby('day_of_year')[
                'season_yearly'].agg(np.mean)
            ax.plot(yearly_data.index, yearly_data.values, label=component)
        else:
            data = forecast.loc[:, ['ds', component]]
            ax.plot(data['ds'], data[component], label=component)
        # Customize the plot
        ax.set_title(f"{component.capitalize().replace('_', ' ')} Component")
        ax.set_xlabel("Day of Week" if component ==
                      "season_weekly" else "Lag" if component == "AR" else "Date")
        ax.set_ylabel(component.capitalize().replace('_', ' '))
        ax.legend()
    plt.tight_layout()
    return fig

@st.cache_data
def condense_fcst_df(forecast_df: pd.DataFrame, forecast_product: str, forecast_session_id: int) -> pd.DataFrame:
    """
    Condense a forecast dataframe by calculating the sum of the yhat columns and adding the specified columns.

    Args:
        forecast_df (pd.DataFrame): The original forecast dataframe.
        forecast_product (str): The forecast product name.
        forecast_session_id (int): The forecast session ID.

    Returns:
        pd.DataFrame: The condensed forecast dataframe.
    """
    forecast_df_dense = forecast_df.copy(deep=True)
    forecast_df_dense['product'] = forecast_product
    forecast_df_dense['fcst_session_id'] = forecast_session_id
    forecast_df_dense['yhat'] = forecast_df_dense.filter(
        regex='^yhat').sum(axis=1)
    # round to int for qty prediction
    forecast_df_dense['fcst_qty'] = forecast_df_dense['yhat'].round()
    forecast_df_dense = forecast_df_dense.loc[:, [
        'fcst_session_id', 'product', 'ds', 'y', 'yhat', 'fcst_qty']]

    return forecast_df_dense

def save_fcst_db(forecast_df: pd.DataFrame, forecast_product: str, fcst_session_name: str, user_tag: str = None):
    """
    Save the forecast to the database using the provided inputs.

    Args:
        forecast_df (pd.DataFrame): The forecast dataframe.
        forecast_product (str): The forecast product name.
        fcst_session_name (str): The forecast session name.
        user_tag (Optional[str], optional): The user tag. Defaults to None.
    """

    pg_manager = PostgreSQLManager()

    with st.form("fcst_session_form"):
        fcst_session_name = st.text_input(
            "Choose a UNIQUE Forecast Session Name", value=fcst_session_name)
        user_tag = st.text_input("User Tag (optional)", value=user_tag)
        date_input = st.date_input("Date")

        submit_button = st.form_submit_button(label="Save Forecast")

        # Save values to st.session_state and insert values into the database
        if submit_button:
            st.session_state['fcst_session_name'] = fcst_session_name
            st.session_state['user_tag'] = user_tag
            st.session_state['date_input'] = date_input.strftime("%Y-%m-%d")

            query_a = "INSERT INTO streamlit.fcst_sessions (fcst_session_name, fcst_date, user_tag) VALUES (%s, %s, %s)"
            params = (fcst_session_name, date_input, user_tag)
            a_success = pg_manager.insert_data(query_a, params)

            # QUERY DB TO GET FCST_SESSION_ID for entered values
            query_id = "SELECT fcst_session_id FROM streamlit.fcst_sessions WHERE fcst_session_name = '" + \
                st.session_state['fcst_session_name'] + "'"
            fcst_session_res = pg_manager.execute_query(query_id)
            fcst_session_id = fcst_session_res[0]['fcst_session_id']
            st.session_state['fcst_session_id'] = fcst_session_id

            dense_df = condense_fcst_df(
                forecast_df, forecast_product, fcst_session_id)
            b_success = pg_manager.insert_dataframe(
                dense_df, 'streamlit.product_fcst')

            if a_success and b_success:
                st.success("Forecast saved!")
            else:
                st.error("Failed to create forecast session.")
            pg_manager.close_pool()



#initiliaze session_state variables
if 'forecast' not in st.session_state:
    st.session_state['forecast'] = pd.DataFrame

if 'horizon' not in st.session_state:
    st.session_state['horizon'] = int

if 'fit_flag' not in st.session_state:
    st.session_state['fit_flag'] = False

if 'fcst_session_name' not in st.session_state:
    st.session_state['fcst_session_name'] = ""

if 'fcst_session_id' not in st.session_state:
    st.session_state['fcst_session_id'] = None

if 'user_tag' not in st.session_state:
    st.session_state['user_tag'] = ""


# Section for prediction and visualization
st.header("4. Predict")

if 'fit_flag' in st.session_state:
    if st.session_state['fit_flag'] == False:
        st.write("Please first load your data and select or train a model in the \"Tune and Train\" page to proceed with forecast prediction.")
    else:

        st.session_state['horizon'] = st.number_input(
            'Please enter the desired forecast horizon.', min_value=1, max_value=len(st.session_state['forecast_df']), value=30)

        predict_check = st.checkbox(
            "Predict with model.", help="Use fitted NeuralProphet model to predict chosen forecast horizon.")

        if predict_check:
            st.session_state['forecast'] = predict(st.session_state['m'],
                                                   st.session_state['horizon'],
                                                   st.session_state['forecast_df'],
                                                   is_events())

            with st.expander("View and Download Forecast"):
                st.subheader("NeuralProphet Forecast Dataframe")
                st.write(
                    "Please note that all forecast results (yhat) and residuals are rounded to nearest integer for representing whole quantities of product.")
                st.dataframe(data=st.session_state['forecast'])
                st.download_button(label="Download Forecast CSV",
                                   data=convert_df(
                                       st.session_state['forecast']),
                                   file_name='forecast_df_' +
                                   st.session_state['forecast_product']+'.csv',
                                   mime='text/csv'
                                   )

            with st.expander("Visualize Results"):
                st.subheader("Forecasted values for " +
                             st.session_state['forecast_product'])
                fig, ax = plt.subplots()
                for label in ax.get_xticklabels(which='major'):
                    label.set(rotation=30, horizontalalignment='right')
                fig = st.session_state['m'].plot_latest_forecast(st.session_state['forecast'],
                                                                 df_name=st.session_state['forecast_df'],
                                                                 ax=ax,
                                                                 xlabel="Date",
                                                                 ylabel="Quantity"
                                                                 )
                st.pyplot(fig)

                st.subheader("Forecast components for " +
                             st.session_state['forecast_product'])
                fig2 = st.session_state['m'].plot_components(
                    fcst=st.session_state['forecast'], plotting_backend='matplotlib')
                st.pyplot(fig2)

                st.subheader("Forecast parameters for " +
                             st.session_state['forecast_product'])
                fig3 = st.session_state['m'].plot_parameters(
                    plotting_backend='matplotlib')
                st.pyplot(fig3)

            with st.expander("Save Results to Database"):
                save_fcst_db(st.session_state['forecast'],
                             st.session_state['forecast_product'],
                             st.session_state['fcst_session_name'],
                             st.session_state['user_tag'])
