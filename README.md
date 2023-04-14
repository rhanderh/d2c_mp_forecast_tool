# D2C Marketplace Sales Forecaster

Welcome to the D2C Marketplace Sales Forecaster! This Streamlit application utilizes the Neural Prophet model to predict sales for various products. Designed with non-technical users in mind, this app guides you through the entire process of loading data, selecting products, tuning and training the model, generating forecasts, and distributing size forecasts.

## Table of Contents
- [Overview](#overview)
- [Getting Started](#getting-started)
- [Application Pages](#application-pages)
    - [1. Load Data](#1-load-data)
    - [2. Product Selection](#2-product-selection)
    - [3. Tune and Train](#3-tune-and-train)
    - [4. Forecast Product](#4-forecast-product)
    - [5. Distribute Size Forecasts](#5-distribute-size-forecasts)
- [Contributing](#contributing)
- [License](#license)

## Overview

  The D2C Marketplace Sales Forecaster app is designed to help you predict sales for products in a direct-to-consumer (D2C) marketplace. With an easy-to-use interface and step-by-step guidance, you'll be able to generate accurate sales forecasts for your products.  The application is built using [Streamlit](https://streamlit.io/) and leverages the [NeuralProphet] (https://neuralprophet.com/) algorithm developed by Meta research to provide flexible forecasting options and interpretable results.  

  Developed as a capstone for the UCSD Professional Extension, Machine Learning Engineering certification.


1. What is the problem?

Forecast product-level sales by day for footwear and apparel digital marketplace sellers.

2. Why solve for it?

Online sellers face challenges in tracking inventory across multiple platforms, which can lead to overselling and increased inventory costs. They are limited in how many times per day each platform can be updated with available inventory balances.  By more accurately predicting sales, sellers can allocate inventory more efficiently, maximize revenue, and minimize costs.  As an additional benefit, the by-day forecasts can assist in more accurately planning warehouse labor and save logistics operational costs for the seller as well. 

3. Unique problems for D2C Digital Sales Forecasting

E-commerce sales forecasting faces specific challenges that this application aims to address leveraging similar-product alignment and hierarchical planning methods.

-Strong seasonality and trend to the sales
-Cold Start problem for new products that have no sales history
-Insufficient  data at the lowest level of granularity such as  a EAN/UPC or size-level  to be able to train machine learning models reliably


## Getting Started

To begin using the application, follow these steps:

1. Clone this repository to your local machine:

git clone https://github.com/rhanderh/d2c_mp_sales_forecaster.git


2. Change to the cloned directory:

cd d2c_mp_sales_forecast_streamlit


3. Install the required dependencies:

pip install -r requirements.txt


4. Start the Streamlit application by running the following command:

streamlit run src/d2c_mp_sales_forecaster/D2CMPForecaster.py


5. The application should now be running in your browser. Follow the instructions in each page to perform sales forecasting.

## Application Pages

The application consists of five main pages that guide you through the sales forecasting process.

### 1. Load Data

In this first step, you'll upload your sales data to the application. The data should be in CSV format with columns representing dates, product identifiers, and sales quantities.

### 2. Product Selection

After loading the data, you'll be presented with a list of products from the dataset. Choose the product(s) you want to forecast sales for by selecting them from the list.

### 3. Tune and Train

In this step, you'll have the option to fine-tune the Neural Prophet model's hyperparameters to improve the sales forecast accuracy. You can either use the default settings or adjust the parameters as needed. Once you're satisfied with the configuration, click the "Train Model" button to train the model using your selected product data.

### 4. Forecast Product

With the trained model, you can now generate sales forecasts for the selected products. Choose a forecast horizon (e.g., next 30 days) and click the "Generate Forecast" button. The application will display the forecasted sales along with confidence intervals for the selected period.

### 5. Distribute Size Forecasts

In the final step, you can allocate the forecasted sales quantities across various product sizes based on historical data or custom distribution rules. You can input the percentage distribution for each size and click "Distribute Sizes" to generate the final size-wise sales forecast.

## Contributing

Contributions are welcome to improve the D2C Marketplace Sales Forecaster. If you have any suggestions, bug reports, or feature requests, please open an issue or submit





