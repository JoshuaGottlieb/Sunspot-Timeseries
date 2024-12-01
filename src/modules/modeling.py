from prophet import Prophet
from . import evaluation as ev
import pickle
import os

def forecast_future(df, model, future_periods, periodicity = 'D', cap = None):
    '''
    Creates forecasts using a trained Prophet model and splits the data along current and future data.
    
    args:
        df: Pandas dataframe containing raw periodic data to use for forecasting
        model: Trained Prophet model to use for forecasting
        future_periods: int, number of periods to forecast into the future
        periodicity: str, indicating periodicity of the data for creature future data
        cap: None, int, or array-like, whether to add a carry capacity to the forecasting dataframe,
            used for logistic growth
        
    returns: Two Pandas dataframes, one containing forecasts on observed data and
        one containing forecasts on future data
    
    '''
    # Make future time periods based on given number of periods and periodicity
    future = model.make_future_dataframe(periods = future_periods, freq = periodicity)
    
    # If a carrying capacity is required, create a new column equal to the capacity
    if cap is not None:
        future['cap'] = cap
    
    # Create predictions
    forecast = model.predict(future)
    
    # Subset data
    current_fc = forecast.iloc[:len(df.index)]
    future_fc = forecast.iloc[len(df.index):]
    
    return current_fc, future_fc

def fit_model(df, func, func_params, seasonality_params = {},
              save = False, save_path = '../models/temp_model.pickle',
              future_periods = None, periodicity = 'D', cap = None):
    '''
    Fits a Prophet model using given function parameters and seasonality parameters.
        Optionally saves the model to a pickled object and creates forecasts using the model which are also
        saved to a pickled object for ease of future loading.
        
    args:
        df: Pandas dataframe containing raw periodic data to use for forecasting
        func: Prophet function signature to use for modeling
        func_params: dict of kwargs to pass to func
        seasonality_params: dict of dicts, custom seasonalities to add to the model
            keys are used for name of the custom seasonality and the inner dictionary is a dict of kwargs to pass
            to model.add_seasonality()
        save: bool, whether to save the model after training
        save_path: str, path to use for saving the model
        future_periods: None or int, number of periods to predict in the future for creating forecasts,
            has no effect if save is False. If not None, forecasts will be saved to pickled objects.
        periodicity: str to pass to model.make_future_dataframe() for creating future periods for forecasting
        cap: None, int, or array-like to use as carry capacity for forecasting, used for logistic growth
        
    returns fitted Prophet model
    '''
    
    # Instantiate the model
    model = func(**func_params)
    
    # Add custom seasonalities
    if len(seasonality_params) != 0:
        for s in seasonality_params:
            model.add_seasonality(name = s, **seasonality_params[s])
            
    # Fit the model
    model.fit(df)
    
    if save:
        # Check if the subdirectory path exists, if it does not exist, create the subdirectories recursively
        model_root = os.path.join(*save_path.split(os.path.sep)[:-1])
        if not os.path.exists(model_root):
            os.makedirs(model_root)
        
        # Save the model to the save path
        with open(save_path, 'wb') as f:
            pickle.dump(model, f)
            
        print(f'Model saved to {save_path} successfully.')
        
        # Create and save forecasts
        if future_periods is not None:
            print(f'Creating model forecasts.')

            # Create forcasts
            forecast = forecast_future(df, model, future_periods, periodicity, cap)
            
            # Define forecast save path to use the same convention as model save path
            fc_path = save_path.split(os.path.sep)
            fc_path = os.path.join(fc_path[0], 'forecasts', *fc_path[2:])
            
            # Check if the subdirectory path exists, if it does not exist,
            # create the subdirectories recursively
            fc_root = os.path.join(*fc_path.split(os.path.sep)[:-1])           
            if not os.path.exists(fc_root):
                os.makedirs(fc_root)
            
            # Save the forecasts to the designated path
            with open(fc_path, 'wb') as f:
                pickle.dump(forecast, f)
                
            print(f'Forecasts created and saved to {fc_path} successfully.')
        
    return model