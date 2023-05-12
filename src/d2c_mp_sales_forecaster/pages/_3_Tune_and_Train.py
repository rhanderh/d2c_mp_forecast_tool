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


class TuneTrain:
    def __init__(self):
        # Initialize state-dependent streamlit variables
        if 'hyperparam_options' not in st.session_state:
            st.session_state['hyperparam_options'] = {}
        if 'selected_options' not in st.session_state:
            st.session_state['selected_options'] = {}
        if 'iter_selections' not in st.session_state:
            st.session_state['iter_selections'] = None
        if 'best_params' not in st.session_state:
            st.session_state["best_params"] = {}
        if 'metrics_test' not in st.session_state:
            st.session_state["metrics_test"] = pd.DataFrame()
        if 'm' not in st.session_state:
            st.session_state['m'] = None
        if 'forecast_df' not in st.session_state:
            st.session_state['forecast_df'] = pd.DataFrame()
        if 'forecast_product' not in st.session_state:
            st.session_state['forecast_product'] = None
        if 'fit_flag' not in st.session_state:
            st.session_state['fit_flag'] = False
        if 'events_df' not in st.session_state:
            st.session_state['events_df'] = None
        normalize_list = ['soft', 'minmax', 'soft1', 'standardize', 'off']
        # Layout page
        st.set_page_config(page_title="D2C, Dropship, Marketplace Forecasting")
        st.header("3. Select, tune, and train the forecasting model.")
        if st.session_state['forecast_df'].empty:
            st.write(
                "Please first load sales data and select a product to forecast to be able to setup and train a forecasting model.")
        else:
            fit_spinner_text = "Fitting the model for product " + \
                st.session_state['forecast_product'] + \
                " with selected hyperparameters..."
            tab_auto, tab_select = st.tabs(
                ["Automatic Hyperparameter Tuning", "Manual Hyperparameter Selection"])
            with tab_select:
                st.subheader("Select hyperparamters to tune a forecasting model for the selected product: ",
                             st.session_state['forecast_product'])
                self.tab_select_form(normalize_list)
                # Print selected options from form
                st.write("Selected options:")
                st.write(st.session_state['selected_options'])
                if st.session_state['selected_options']:
                    # Cross-Validation Option
                    with st.expander("Cross-Validation"):
                        if st.button("Cross-Validate"):
                            with st.spinner("Performing cross-validation... (Please do not refresh or close this tab while running to complete this task.)"):
                                st.session_state['metrics_test'], st.session_state['selected_options'] = self.cross_validate(
                                    st.session_state['forecast_df'],
                                    st.session_state['forecast_product'],
                                    st.session_state['selected_options'],
                                    st.session_state['events_df'])
                            st.success("Cross-validation complete!", icon="✅")
                        else:
                            st.write(
                                "Push \"Cross-Validate\" to perform 5-fold backtesting cross-validation with your selected hyperparameters.")
                    # Fit complete dataset (required step for forecasting)
                    with st.expander("Fit Model to Complete Dataset"):
                        st.write("Current Identified Best Parameters: ")
                        st.write(st.session_state['selected_options'])
                        if st.button("Fit Model", key='fit_select'):
                            with st.spinner(fit_spinner_text):
                                st.session_state["m"] = self.fit_complete(params=st.session_state['selected_options'],
                                                                          df=st.session_state['forecast_df'],
                                                                          events_df=st.session_state['events_df'])
                            st.success("Model fit complete!", icon="✅")
                        else:
                            st.write(
                                "Fit the model with the full data-set for the selected product.  This is required for moving to forecast prediction.")
            with tab_auto:
                st.subheader("Select options and ranges for an automated grid-search to find the best hyperparameters: ",
                             st.session_state['forecast_product'])
                self.tab_auto_form(normalize_list)
                st.write("Selected options:")
                st.write(st.session_state['hyperparam_options'])
                if st.session_state['hyperparam_options']:
                    # Grid Search for Hyperparameter option
                    with st.expander("Hyperparameter Search"):
                        if st.button("Grid Search"):
                            with st.spinner("Performing cross-validation... (Please do not refresh or close this tab while running to complete this task.)"):
                                st.session_state['metrics_test'], st.session_state['best_params'] = self.cross_validate(
                                    st.session_state['forecast_df'],
                                    st.session_state['forecast_product'],
                                    st.session_state['hyperparam_options'],
                                    st.session_state['events_df'])
                            st.success("Cross-validation complete!", icon="✅")
                        else:
                            st.write(
                                "Push \"Cross-Validate\" to Grid Search for best hyperparameters using 5-fold backtesting cross-validation.")
                    if st.session_state['best_params']:
                        # Fit complete model after hyperparameter best are identified
                        with st.expander("Fit Model to Complete Dataset"):
                            st.write("Current Identified Best Parameters: ")
                            st.write(st.session_state['best_params'])
                            if st.button("Fit Model", key='fit_auto'):
                                with st.spinner(fit_spinner_text):
                                    st.session_state["m"] = self.fit_complete(params=st.session_state['best_params'],
                                                                              df=st.session_state['forecast_df'],
                                                                              events_df=st.session_state['events_df'])
                                st.success("Model fit complete!", icon="✅")
                            else:
                                st.write("Fit the model with the full data-set and best hyperparameters from grid search for the selected product.  \n",
                                         "This is required for moving to forecast prediction.")

    @st.cache_resource(show_spinner=False)
    def fit_complete(_self, params: dict, df: pd.DataFrame, events_df: pd.DataFrame = None) -> NeuralProphet:
        """
        Fit a NeuralProphet model to the complete dataset using the selected params or the best parameters obtained from cross-validation.

        Args:
            params: A dictionary containing the best hyperparameters.
            df: A DataFrame containing the dataset to be fit.
            events_df: A DataFrame containing events data (optional).

        Returns:
            A fitted NeuralProphet model.
        """
        # Fit the whole training set with best params
        m = NeuralProphet(**params)
        for col in df.columns:
            if col != 'y' and col != 'ds':
                m = m.add_lagged_regressor(names=col)
        if events_df is not None:
            # Need to update this so it is flexible to different holiday names
            m.add_events(['Cyber Week'], mode='additive')
            data_ev = m.create_df_with_events(df, events_df)
            fit_metrics = m.fit(df=data_ev, freq="D")
        else:
            fit_metrics = m.fit(df=df, freq="D")
        st.write("Metrics from model fit:", fit_metrics)
        st.session_state['fit_flag'] = True
        return m

    @st.cache_resource(show_spinner=False)
    def cross_val_tune(_self, params: dict, df: pd.DataFrame, forecast_product: str, events_df: pd.DataFrame = None, horizon: int = 30) -> tuple:
        """
        Perform hyperparameter tuning using cross-validation for a given forecast product.

        Args:
            params: A dictionary containing the parameters to be tuned.
            df: A DataFrame containing the dataset.
            forecast_product: The product to be forecasted.
            events_df: A DataFrame containing events data (optional).
            horizon: The forecast horizon in days (default: 30).

        Returns:
            A tuple containing:
                1. best_metrics: A DataFrame containing the best test metrics from cross-validation.
                2. best_params: A dictionary containing the best hyperparameters.
        """
        METRICS = ["MAE", "RMSE"]
        fold_pct = horizon/len(df)
        st.write(
            "Applying five-fold cross-validation to the forecast model for " + forecast_product)
        param_grid = params
        # Generate all combinations of parameters
        all_params = [dict(zip(param_grid.keys(), v))
                      for v in itertools.product(*param_grid.values())]
        cv_test_loss = []
        test_metrics_list = []
        best_metrics = []
        # Use cross validation to evaluate all parameters
        for i, params in enumerate(all_params):
            folds = NeuralProphet(**params).crossvalidation_split_df(df,
                                                                     freq="D", k=5, fold_pct=fold_pct, fold_overlap_pct=0)
            metrics_train = pd.DataFrame(columns=METRICS)
            metrics_test = pd.DataFrame(columns=METRICS)
            for df_train, df_test in folds:
                m = NeuralProphet(**params)
                for col in df.columns:
                    if col != 'y' and col != 'ds':
                        m = m.add_lagged_regressor(names=col)
                if events_df is not None:
                    # Need to update this so that it is flexible to different holidays
                    m.add_events(['Cyber Week'], mode='additive')
                    df_train_ev = m.create_df_with_events(df_train, events_df)
                    df_test_ev = m.create_df_with_events(df_test, events_df)
                    train = m.fit(df=df_train_ev, freq="D")
                    test = m.test(df=df_test_ev)
                else:
                    train = m.fit(df=df_train, freq="D")
                    test = m.test(df=df_test)
                metrics_train = metrics_train.append(train.iloc[-1])
                metrics_test = metrics_test.append(test.iloc[-1])
            cv_test_loss.append(metrics_test['MSELoss'].mean())
            test_metrics_list.append(metrics_test)
        best_params = all_params[np.argmin(cv_test_loss)]
        best_metrics = test_metrics_list[np.argmin(cv_test_loss)]
        st.write("Best model test metrics from cross-validation:")
        st.write(best_metrics)
        st.write("Best model test metrics mean across 5 folds: ")
        st.write(best_metrics.mean())
        st.write("Resulting hyperparmeters from cross-validation:")
        st.write(best_params)
        return best_metrics, best_params

    @st.cache_resource(show_spinner=False)
    def selections_to_lists(_self, selected_options: dict) -> dict:
        """
        Convert selected_options dictionary values to lists.

        Args:
            selected_options (dict): A dictionary containing selected options.

        Returns:
            dict: A dictionary with values converted to lists.
    """
        return {k: [v] for k, v in selected_options.items()}

    @st.cache_resource(show_spinner=False)
    def cross_validate(_self, forecast_df: pd.DataFrame, forecast_product: str, selected_options: dict, events_df: pd.DataFrame = None) -> tuple:
        """
        Perform cross-validation for the NeuralProphet model.

        Args:
            forecast_df (pd.DataFrame): DataFrame containing forecast data.
            forecast_product (str): The product for forecasting.
            selected_options (dict): A dictionary containing selected options.

        Returns:
            tuple: A tuple containing the test metrics and updated selected options.
        """
        if isinstance(selected_options['n_changepoints'], list):
            iter_selections = selected_options
        else:
            iter_selections = _self.selections_to_lists(selected_options)
        metrics_test, selected_options = _self.cross_val_tune(
            params=iter_selections,
            df=forecast_df,
            forecast_product=forecast_product,
            events_df=events_df,
            horizon=int(iter_selections['n_forecasts'][0]))
        return metrics_test, selected_options

    def tab_select_form(self, normalize_list: list):
        """
        Create a form to select options for param_grid.
        """
        with st.form("param_grid_form"):
            # Add a title to the form
            st.write("Select options for param_grid")
            # Define the options for each parameter
            seasonality_mode_options = st.selectbox(
                'Select seasonality_mode', ['additive', 'multiplicative'])
            loss_func_options = st.selectbox(
                'Select loss_func', ['MSE', 'MAE'])
            n_changepoints = 10
            n_changepoints_options = st.slider(
                'Select n_changepoints', min_value=1, max_value=20, value=n_changepoints)
            changepoints_range = 10.0
            changepoints_range_options = st.slider(
                'Select changepoints_range', min_value=0.01, max_value=20.0, value=changepoints_range)
            normalize_options = st.selectbox(
                'Select normalization method', normalize_list)
            n_lags_options = st.slider(
                'Select n_lags', min_value=7, max_value=90, value=30)
            n_forecasts_options = st.slider(
                'Select n_forecasts', min_value=7, max_value=90, value=30)
            num_hidden_layers_options = st.selectbox(
                'Select num_hidden_layers', [0, 1, 2])
            # Add a submit button to the form
            submitted = st.form_submit_button("Submit")
            if submitted:
                st.session_state['selected_options'] = {
                    'seasonality_mode': seasonality_mode_options,
                    'loss_func': loss_func_options,
                    'n_changepoints': n_changepoints_options,
                    'changepoints_range': changepoints_range_options,
                    'normalize': normalize_options,
                    'num_hidden_layers': num_hidden_layers_options,
                    'n_lags': n_lags_options,
                    'n_forecasts': n_forecasts_options
                }

    def tab_auto_form(self, normalize_list: list):
        """
        Create a form to select options for param_grid.
        """
        with st.form("hyper_grid_form"):
            # Add a title to the form
            st.write("Select options for hyperparameter search.")
            # Define the options for each parameter
            seasonality_mode_options = st.multiselect(
                'Select seasonality_mode', ['additive', 'multiplicative'], default='additive')
            loss_func_options = st.multiselect(
                'Select loss_func', ['MSE', 'MAE'], default='MSE')
            n_cp_min, n_cp_max = st.slider(
                'Select n_changepoints (min=1/max=20)', min_value=1, max_value=20, value=(5, 10))
            n_cp_steps = st.number_input(
                'Select step-size for n_changepoints (min=1/max=10)', min_value=1, max_value=10, value=5)
            cp_range_min, cp_range_max = st.slider(
                'Select changepoints_range (min=0.01/max=20.0)', min_value=0.01, max_value=20.0, value=(1.0, 10.0))
            cp_range_steps = st.number_input(
                'Select step-size for n_changepoints (min=0.01/max=10.0)', min_value=0.01, max_value=10.0, value=9.0)
            normalize_options = st.multiselect(
                'Select normalization method', normalize_list, default='soft')
            n_lags_options = st.slider(
                'Select n_lags', min_value=7, max_value=90, value=30)
            n_forecasts_options = st.slider(
                'Select n_forecasts', min_value=7, max_value=90, value=30)
            num_hidden_layers_options = st.multiselect(
                'Select num_hidden_layers', [0, 1, 2], default=0)
            # Add a submit button to the form
            submitted = st.form_submit_button("Submit")
            # If the user submitted the form, print the selected options
            if submitted:
                st.session_state['hyperparam_options'] = {
                    'seasonality_mode': seasonality_mode_options,
                    'loss_func': loss_func_options,
                    'n_changepoints': np.arange(n_cp_min, n_cp_max+1, n_cp_steps).tolist(),
                    'changepoints_range': np.arange(cp_range_min, cp_range_max+0.01, cp_range_steps).tolist(),
                    'normalize': normalize_options,
                    'num_hidden_layers': num_hidden_layers_options,
                    'n_lags': [n_lags_options],
                    'n_forecasts': [n_forecasts_options]
                }


if __name__ == '__main__':
    tt = TuneTrain()
