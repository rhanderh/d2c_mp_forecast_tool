import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from datetime import date
from datetime import timedelta

# Page config
st.set_page_config(page_title="D2C, Dropship, Marketplace Forecasting")

def get_len_history(all_sales_data: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate the length of sales history for each product.
    
    Args:
        all_sales_data: A DataFrame containing aggregated product sales data.
        
    Returns:
        sales_minmax_dates: A DataFrame containing the length of sales history for each product.
    """
    sorted_sales = all_sales_data.sort_values(by=['product','ds'], ascending = True)
    sales_minmax_dates = sorted_sales.groupby('product').agg({'ds':['min','max'], 'y':'sum'})
    sales_minmax_dates['len_hist'] = (sales_minmax_dates.loc[:,('ds','max')] - sales_minmax_dates.loc[:,('ds','min')]) / np.timedelta64(1, 'D')
    return sales_minmax_dates

@st.cache_data
def get_similarities(master_data_enc: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate cosine similarity between products based on encoded master data.
    
    Args:
        master_data_enc: A DataFrame containing encoded product master data.
        
    Returns:
        similarity_df: A DataFrame containing the cosine similarity scores between products.
    """
    similarity = cosine_similarity(master_data_enc)
    similarity_df = pd.DataFrame(similarity, 
                             index=master_data_enc.index.values,
                             columns=master_data_enc.index.values)
    return similarity_df

def check_history_needed(sales_minmax_dates: pd.DataFrame, product_fcst: str, min_req_hist: float = 366.0):
    """
    Check for the minimum history needed for forecasting a specific product.
    
    Args:
        sales_minmax_dates: A DataFrame containing the length of sales history for each product.
        product_fcst: The product to be forecasted.
        min_req_hist: The minimum required history for forecasting (default: 366 days).
        
    Returns:
        min_append_hist_req: The minimum required history to append.
        cutoff_date: The cutoff date for adding history.
    """
    #lookup length of history for that product
    len_avail_hist = sales_minmax_dates.loc[sales_minmax_dates.index == product_fcst]['len_hist'].values[0]
    #difference with minimum required history to identify days gap aka minimum required length of history to append
    min_append_hist_req = int(min_req_hist - len_avail_hist)
    #identify the cutoff to add the length of history to
    cutoff_date = sales_minmax_dates.loc[sales_minmax_dates.index == product_fcst][('ds','min')] - timedelta(min_append_hist_req)
    return min_append_hist_req, cutoff_date.values[0]


def match_product(similarity_df: pd.DataFrame, sales_minmax_dates: pd.DataFrame, product_fcst: str, cutoff_date: np.datetime64,  top_n: int = 1):
    """
    Match a product to similar products based on a similarity DataFrame.
    
    Args:
        similarity_df: A DataFrame containing the cosine similarity scores between products.
        sales_minmax_dates: A DataFrame containing the length of sales history for each product.
        product_fcst: The product to be forecasted.
        cutoff_date: The cutoff date for adding history.
        top_n: The number of top similar products desired (default: 1).
        
    Returns:
        matched: A list of matched similar products.
    """
    # product list of only products that meet minimum length requirements in addition to available history
    product_list = sales_minmax_dates.loc[sales_minmax_dates[('ds','min')] < cutoff_date].index.values
    # slice similarity dataframe by the product being forecasted on row-index, and the product list on the column-index
    product_sim = similarity_df.loc[similarity_df.index == product_fcst, product_list].values[0]
    # get products sorted by similarity
    sorted_products = np.argsort(product_sim)[::-1]
    # get matched product id
    matched = product_list[sorted_products[1:top_n+1]]
    return matched

def fill_missing(sales_history: pd.DataFrame) -> pd.DataFrame:
    """
    Fill missing values in a sales history DataFrame.
    
    Args:
        sales_history: A DataFrame containing sales history data.
        
    Returns:
        sales_history: A DataFrame with missing values filled.
    """
    sales_history = sales_history.sort_values(by='ds')
    sales_history.index = pd.to_datetime(sales_history['ds'])
    sales_history.drop(columns=['ds'], inplace=True)
    sales_history = sales_history.asfreq('D')
    #fill all non quantities with forward-fill
    for col in sales_history.columns:
                if col != 'y':
                    sales_history[col].fillna(method='ffill',inplace=True) 
    # assume a 0 means no sale for that day
    sales_history['y'].fillna(0, inplace=True) 
    sales_history.reset_index(inplace=True)
    return sales_history

def merge_regressors(sales_data, lag_data, future_data=None) -> pd.DataFrame:
    """
    Merge sales data with lag and future regressor data.
    
    Args:
        sales_data: A DataFrame containing sales data.
        lag_data: A DataFrame containing lag regressor data.
        future_data: A DataFrame containing future regressor data (optional).
        
    Returns:
        all_sales_data_lag: A DataFrame with sales data merged with lag and future regressor data.
    """
    all_sales_data_lag = pd.merge(sales_data, lag_data, on='ds', how='inner')
    return all_sales_data_lag

def append_history(all_sales_data: pd.DataFrame, matched: list, product_fcst: str, lag_data: pd.DataFrame = None) -> pd.DataFrame:
    """
    Append sales history for a product with the history of a similar product.
    
    Args:
        all_sales_data: A DataFrame containing aggregated product sales data.
        matched: A list of matched similar products.
        product_fcst: The product to be forecasted.
        lag_data: A DataFrame containing lag regressor data (optional).
        
    Returns:
        A tuple of two DataFrames:
            1. appended_history: The appended sales history.
            2. appended_history_product: The appended sales history with the product column.
    """
    product_history = all_sales_data.loc[all_sales_data['product'] == product_fcst]
    min_date = product_history['ds'].min()
    extra_history = all_sales_data.loc[all_sales_data['product'] == matched[0]]
    extra_history = extra_history.loc[extra_history['ds'] < min_date]
    appended_history_product = fill_missing(pd.concat([product_history, extra_history]))
    if lag_data is not None:
            appended_history_product = merge_regressors(appended_history_product, lag_data)
    appended_history = appended_history_product.drop(columns='product').copy(deep=True)
    return appended_history, appended_history_product


def prepare_non_appended(sales_history: pd.DataFrame, forecast_product: str, all_sales_regressors: pd.DataFrame = None):
    """
    Prepare sales history for a product without appending additional history from similar products.
    
    Args:
        sales_history: A DataFrame containing sales history data.
        forecast_product: The product to be forecasted.
        all_sales_regressors: A DataFrame containing sales data with additional regressors (optional).
        
    Returns:
        A tuple of two DataFrames:
            1. forecast_df: The prepared sales history without the product column.
            2. forecast_df_product: The prepared sales history with the product column.
    """
    if all_sales_regressors is None:
        forecast_df_product = fill_missing(sales_history.loc[sales_history['product'] == forecast_product])
        forecast_df = forecast_df_product.loc[:,['ds','y']].copy(deep=True)
    else:
        forecast_df_product = fill_missing(all_sales_regressors.loc[all_sales_regressors['product'] == forecast_product])
        forecast_df = forecast_df_product.drop(columns='product').copy(deep=True)
    return forecast_df, forecast_df_product

def clear_state(key: str):
    """
    Clear the session state for a specified key.
    
    Args:
        key: The key to be cleared from the session state.
    """
    if key in st.session_state:
        st.session_state[key] = None


if 'forecast_df' not in st.session_state:
    st.session_state['forecast_df'] = pd.DataFrame

if 'forecast_df_product' not in st.session_state:
    st.session_state['forecast_df_product'] = pd.DataFrame

if 'matched' not in st.session_state:
    st.session_state['matched'] = None

if 'forecast_product' not in st.session_state:
    st.session_state['forecast_product'] = None


#Title
st.title("D2C, Dropship, Marketplace Forecasting")

#Select the product that will be used for forecasting.
st.header("2. Select product for forecasting.")

if 'master_data' not in st.session_state:
    st.write("Please go to Load Data and upload a product attribute file to be able to choose a product for forecasting.")
else:
    if st.session_state['forecast_product'] is not None:
       st.session_state['forecast_product'] = forecast_product = st.table(st.session_state['master_data']).selectbox('Select a product ID', 
                                                                              st.session_state['master_data'].index, 
                                                                              index=st.session_state['master_data'].index.get_loc(st.session_state['forecast_product']),
                                                                              on_change=clear_state('matched'))
    else:
        st.session_state['forecast_product'] = forecast_product = st.table(st.session_state['master_data']).selectbox('Select a product ID', 
                                                                               st.session_state['master_data'].index, 
                                                                               index=0,
                                                                               on_change=clear_state('matched'))
    cola, colb = st.columns(2)
    with cola:
        st.write("You selected:", st.session_state['forecast_product'])
        st.write(st.session_state['master_data'].loc[st.session_state['forecast_product']])
    with colb:
        if 'all_sales_data' in st.session_state:
            if 'lag_data' in st.session_state:
                st.session_state['all_sales_regressors'] = pd.DataFrame
                st.session_state['all_sales_regressors'] = merge_regressors(st.session_state['all_sales_data'], st.session_state['lag_data'])
                st.write("Sales history:") 
                st.write(st.session_state['all_sales_regressors'].loc[st.session_state['all_sales_regressors']['product'] == st.session_state['forecast_product'],:]
                         .sort_values(by='ds'))
            else:
                st.write("Sales history:") 
                st.write(st.session_state['all_sales_data'].loc[st.session_state['all_sales_data']['product'] == st.session_state['forecast_product'],:])
        else:
            st.write("Please load sales history data for the product ", st.session_state['forecast_product'], " before proceeding further.")

            

#Section for identifying similar products
st.subheader("Optional - Match most similar product for sufficient historical data for forecasting. ")


st.write("It is necessary to have at minimum one year of sales history for the NeuralPropeht algorithm to pick up on seasonal affects effectively.  For products with not enough sales history, or for new products, we will use Cosine Similarity on the encoded product master to identify a product that is most-similar, and append that to the desired product history to be able to come up with a better forecast. \
        For reference on the meaning of Cosine Similarity: https://www.geeksforgeeks.org/cosine-similarity/")
st.write("If you loaded additional regressors on the previous page, these will be automatically inner-joined to the forecast product dataset.")

# Select if we want to use cosine similarity to extend product history?
use_similarity = st.checkbox(
    "Identify most-similar product to extend historical data for forecasting?", help="Uses cosine similarity to match most-similar product and appends similar product sales to forecast product sales data to be at least one year in length. "
)


colc, cold = st.columns(2)
if use_similarity:
    if 'all_sales_data' in st.session_state:
        #This sets up base values for calculating and appending history based on the uploaded data
        sales_minmax_dates = get_len_history(st.session_state['all_sales_data'])
        top_n = 1 #Only find one most-similar product to matach and append history
        similarity_df = get_similarities(st.session_state['master_data_enc'])
        min_append_hist_req, cutoff_date = check_history_needed(sales_minmax_dates, st.session_state['forecast_product'])

        #Determine if history appendage is required, and if so, append history of the most-similar identified product
        if min_append_hist_req > 0:
            st.session_state['matched'] = match_product(similarity_df, sales_minmax_dates, st.session_state['forecast_product'], cutoff_date, top_n)
            if 'lag_data' in st.session_state:
                st.session_state['forecast_df'], st.session_state['forecast_df_product']  = append_history(st.session_state['all_sales_data'], 
                                                                                                           st.session_state['matched'], 
                                                                                                           st.session_state['forecast_product'],
                                                                                                           st.session_state['lag_data'])
            else:
                st.session_state['forecast_df'], st.session_state['forecast_df_product']  = append_history(st.session_state['all_sales_data'], 
                                                                                                           st.session_state['matched'], 
                                                                                                           st.session_state['forecast_product']
                                                                                                           )        
        else:
            if 'all_sales_regressors' in st.session_state:
                st.session_state['forecast_df'], st.session_state['forecast_df_product']  = prepare_non_appended(st.session_state['all_sales_data'], 
                                                                                                         st.session_state['forecast_product'],
                                                                                                         st.session_state['all_sales_regressors'])
            else:
                st.session_state['forecast_df'], st.session_state['forecast_df_product']  = prepare_non_appended(st.session_state['all_sales_data'], 
                                                                                                         st.session_state['forecast_product'])
        
        if 'forecast_df_product' in st.session_state:
            with colc:
                st.write("This is the data-set that will be used for forecasting:")
                st.write(st.session_state['forecast_df_product'])
            with cold:
                if st.session_state['matched'] is not None:
                    st.write('Identified most-similar product based on uploaded attributes using cosine similarity:')
                    st.write(st.session_state['master_data'].loc[st.session_state['matched']])
                else:
                    st.write('Sufficient historical data identified, there is no need to append similar product history.')
    else:
        st.write("Please upload sales history from the Load Data page to proceed.")

else:
    #If the user does not select to append historicals, set up the forecast dataframe with just the chosen product
    
    if 'all_sales_data' in st.session_state:
        if 'all_sales_regressors' in st.session_state:
            st.session_state['forecast_df'], st.session_state['forecast_df_product']  = prepare_non_appended(st.session_state['all_sales_data'], 
                                                                                                         st.session_state['forecast_product'],
                                                                                                         st.session_state['all_sales_regressors'])
        else:
            st.session_state['forecast_df'], st.session_state['forecast_df_product']  = prepare_non_appended(st.session_state['all_sales_data'], 
                                                                                                         st.session_state['forecast_product'])
        with colc:
            st.write("This is the data-set that will be used for forecasting:")
            st.write(st.session_state['forecast_df_product'])
    else:
        st.write("Please upload sales history from the Load Data page to proceed.")



