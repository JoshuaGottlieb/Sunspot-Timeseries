# Sunspot-Timeseries

## Project Description and Repository Usage
This project analyzes sunspot time series data from the [Solar Influences Data Analysis Center](https://www.sidc.be/SILSO/datafiles). There are three datasets which are used: one with daily sunspot predictions, one with mean-aggregated monthly sunspot predictions, and one with mean-aggregated yearly predictions. The purpose of this project is to use the [Prophet](https://facebook.github.io/prophet/docs/quick_start.html#python-api) Python package to create a time series model to predict future sunspots and to determine which sampling frequency of data produces the most robust models.

The models were tuned by adding custom seasonalities, testing different growth types, and by adjusting the changepoint prior scale and number of changepoints. A description of the effects of each these parameters can be found on the [Prophet](https://facebook.github.io/prophet/docs/quick_start.html) documentation page but can be broadly summarized as allowing the time series model to become more flexible and able to adapt to large seasonalities or changes in trend. To test the effectiveness of each model, the mean average error (MAE), mean average percentage error (MAPE) and R-squared scores were calculated to determine the size of the model errors and ability of the model to capture the variance in the data. Each model was also set to forecast into the future past the original datasets. For the daily data, forecasts were predicted 100/200/365 days into the future; for the monthly data, forecasts were predicted 1/6/9 months into the future; and, for the yearly data, forecasts were predicted 1/10/20 years into the future.

There are four main notebooks in this repository. The bulk of the work and explanations can be found in the [Modeling-Daily.ipnyb](https://github.com/JoshuaGottlieb/Sunspot-Timeseries/blob/main/src/Modeling-Daily.ipynb) notebook. Within this notebook, the tuning process is walked through in-depth at each step of the process, with a final summary of the model performance on the daily sampled data. The [Modeling-Monthly.ipynb](https://github.com/JoshuaGottlieb/Sunspot-Timeseries/blob/main/src/Modeling-Monthly.ipynb) and [Modeling-Yearly.ipynb](https://github.com/JoshuaGottlieb/Sunspot-Timeseries/blob/main/src/Modeling-Yearly.ipynb) notebooks cover the tuning process of models using the monthly and yearly aggregated data, respectively. These two notebooks have fewer explanations and are more concise than the daily notebook, as the tuning process is effectively identical to the process used in the daily notebook. The final notebook, [Model-Comparisons.ipynb](https://github.com/JoshuaGottlieb/Sunspot-Timeseries/blob/main/src/Model-Comparisons.ipynb), compares the base and best models from each of the other three notebooks in order to determine which model is the most robust and which sampling frequency for the data is best for predicting sunspots.

All of the models were saved as pickled objects under [models](https://github.com/JoshuaGottlieb/Sunspot-Timeseries/tree/main/models) for ease of loading without having to retrain the models. Forecasts require significant time to execute, as the model must construct uncertainty intervals for each future period, so the forecasts were also saved as pickled objects under [forecasts](https://github.com/JoshuaGottlieb/Sunspot-Timeseries/tree/main/forecasts) for ease of loading without having to repeatedly recreate forecasts. In order to keep the notebooks easy to read and to reduce repetition of code, many functions were moved into project [modules](https://github.com/JoshuaGottlieb/Sunspot-Timeseries/tree/main/src/modules), separated by core function use. For more information, the modules contain full documentation, and module summaries are shown below under the repository structure.

## Repository Structure
```
.
|── data/                            # Datasets
|   ├── SN_d_tot_V2.0.csv            # Daily sunspot predictions
|   ├── SN_m_tot_V2.0.csv            # Monthly average sunspot predictions
|   └── SN_y_tot_V2.0.csv            # Yearly average sunspot predictions
|
|── models/                          # Pickled trained Prophet models
|   ├── daily/                       # Models fit on daily frequency sunspot predictions
|   ├── monthly/                     # Models fit on monthly averaged frequency sunspot predictions
|   └── yearly/                      # Models fit on yearly averaged frequency sunspot predictions
|
|── forecasts/                       # Pickled predictions from trained Prophet models
|   ├── daily/                       # Predictions from models fit on daily frequency sunspot predictions
|   ├── monthly/                     # Predictions from models fit on monthly averaged frequency sunspot predictions
|   └── yearly/                      # Predicitons from models fit on yearly averaged frequency sunspot predictions
|
|── src/                             # Notebooks go here
|   ├── Modeling-Daily.ipynb         # Model tuning and explanations on daily frequency sunspot predictions
|   ├── Modeling-Monthly.ipynb       # Model tuning and explanations on monthly frequency sunspot predictions
|   ├── Modeling-Yearly.ipynb        # Model tuning and explanations on yearly frequency sunspot predictions
|   ├── Model-Comparisons.ipynb      # Comparisons and evaluation of different models trained and tuned at different sampling frequencies
|   └── modules/                     # Modules used by notebooks go here
|   	├── __init__.py
|   	├── evaluation.py            # Functions to evaluate model performance
|   	├── preprocessing.py         # Functions to preprocess datasets for use by Prophet
|   	└── modeling.py              # Functions to fit models and create forecasts
```


## Libraries and Versions
```
# Minimal versions of libraries and Python needed
Python 3.8.10
matplotlib==3.7.5
pandas==2.0.3
prophet==1.1.6
scikit_learn==1.3.2
seaborn==0.13.2
```
