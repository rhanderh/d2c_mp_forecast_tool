
import streamlit as st
import itertools
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
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
#from d2c_mp_sales_forecast_streamlit.libraries.similar_products import *

# Page config
st.set_page_config(page_title="D2C, Dropship, Marketplace Time-Series Forecasting")

#Title
st.title("D2C, Dropship, Marketplace Sales Time-Series Forecasting")





