# For more information, please refer to https://aka.ms/vscode-docker-python
FROM continuumio/miniconda3

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    git \
    && rm -rf /var/lib/apt/lists/*

RUN git clone https://github.com/rhanderh/d2c_mp_forecast_tool.git .

# Create the environment from environment.yml 
ADD environment.yml /tmp/environment.yml
RUN conda env create -f /tmp/environment.yml
RUN echo "source activate d2c_mp_sales_fcst" > ~/.bashrc
ENV PATH /opt/conda/envs/d2c_mp_sales_fcst/bin:$PATH

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

# Define the entrypoint to activate the conda environment before running the command
ENTRYPOINT ["/opt/conda/envs/d2c_mp_sales_fcst/bin/streamlit", "run", "src/d2c_mp_sales_forecaster/D2CMPForecaster.py", "--server.port=8501", "--server.address=0.0.0.0"]
