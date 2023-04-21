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

# Create the environment from environment.yml and make sure conda is in the PATH
RUN conda env create -f environment.yml && \
    echo "source activate $(head -1 environment.yml | cut -d' ' -f2)" > ~/.bashrc && \
    /bin/bash -c "source ~/.bashrc"

EXPOSE 8501

# Define the entrypoint to activate the conda environment before running the command
ENTRYPOINT [ "/bin/bash", "-c", "source activate $(head -1 environment.yml | cut -d' ' -f2) && exec" ]

# Run streamlit when the container launches
CMD ["streamlit", "run", "src/d2c_mp_sales_forecaster/D2CMPForecaster.py"]
