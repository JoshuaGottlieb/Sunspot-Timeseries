import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, r2_score

def load_pickle(path):
    '''
    Loads a pickled object.
    
    arg:
        path: str to use as path to pickled object
        
    returns: unpickled object
    '''
    with open(path, 'rb') as f:
        obj = pickle.load(f)
        
    print(f'Pickled object at {path} loaded successfully.')
        
    return obj

def set_text_params(ax, title = '', xlabel = '', ylabel = '', tick_labels = None,
                    titlesize = 20, labelsize = 16, ticksize = 12, rotation = 0):
    '''
    Sets the title, x- and y-labels, and fontsizes for a Matplotlib Axes object.
    
    params:
        ax: Matplotlib Axes object to modify
        title: str to use as title for plot
        xlabel: str to use as x-label for plot
        ylabel: str to use as y-label for plot
        tick_labels: list of str to use as tick labels for plot
        titlesize: int to use as fontsize for title
        labelsize: int to use as fontsize for x- and y-labels
        ticksize: int to use as fontsize for tick labels
        rotation: int to use for rotating x-tick labels
    '''  
    
    ax.set_title(title, fontsize = titlesize);
    ax.set_xlabel(xlabel, fontsize = labelsize);
    ax.set_ylabel(ylabel, fontsize = labelsize);
    if tick_labels is not None:
        ax.set_xticklabels(tick_labels)
    
    ax.tick_params(axis = 'both', labelsize = ticksize)
    if rotation != 0:
        ax.tick_params(axis = 'x', rotation = rotation)
    
    return

def format_future_artists(ax, n_periods, prediction_periods, periodicity):
    '''
    Formats the artists on a Prophet plot graph so that each prediction period has a different color.
    
    args:
        ax: Matplotlib Axes object that contains the results of a Prophet model.plot() function
        n_periods: int, number of partitions for future predictions
        prediction_periods: list of int, placement of partitions for future predictions
        periodicity: str, 'D', 'MS', or 'YS', denoting the periodicity of the data
    '''
    
    # Extract the artists from the axis
    children = ax.get_children()
    
    # Extract the first three artists: the observed data and the forecast and uncertainty on current data
    lines = children[0:3]
    
    # Set the color of the forecast and uncertainty on current data to green
    children[1].set_color('green')
    children[2].set_color('green')
    
    # Define future colors for further forecasts of prediction periods
    colors = sns.color_palette('magma', n_periods)

    # Every 3 children follow the form of: observed data, forecast, uncertainty
    for i, c in enumerate(children[3:(3 + 1) * n_periods]):
        # Remove the observed data points and delete them from the axis
        if i % 3 == 0 and i != (3 * n_periods):
            c.remove();
            continue
        
        # Otherwise, set the forecast and uncertainty colors to the next color
        # and add to the list of lines for adding artists to the legend
        c.set_color(colors[i // 3])
        lines.append(c)

    # Extract the appropriate label based on periodicity of model
    period_map = {'D': 'Days', 'MS': 'Months', 'YS': 'Years'}
    period_label = period_map[periodicity]
    
    # Create labels
    labels = ['Observed Data Points', 'Forecast, Current', 'Uncertainty, Current']
    labels += [f'Forecast, {prediction_periods[int(i / 2)]} {period_label}' if i % 2 == 0
               else f'Uncertainty, {prediction_periods[int(i / 2)]} {period_label}' for i in range(len(lines[3:]))]

    # Add to legend
    ax.legend(handles = lines, labels = labels, loc = 1, ncols = 2, fontsize = 12);
    
    return

def plot_future(model, current_fc, future_fc, prior_periods = None, prediction_periods = None,
                ax = None, uncertainties = True, periodicity = 'D', title = ''):
    '''
    Plots forecasts of a Prophet model, with the optional ability to restrict the plotted timeframe,
        create partitions of future predictions, and the choice of whether to plot uncertainty intervals.
        
    args:
        model: Trained Prophet model to use for invoking model.plot()
        current_fc: Pandas DataFrame containing forecasts on current data
        future_fc: Pandas DataFrame containing forecasts on future data
        prior_periods: None or int, number of periods prior to future forecasts
            to include in the plot
        prediction_periods: None or list of int, partition of future periods to use for future forecasts,
            must not exceed length of future_fc. If None, forecast on future data will not be partitioned.
        ax: None or Matplotlib Axes object to plot on, if None, a new axis is created using plt.subplots()
        uncertainties: bool, whether to plot uncertainty intervals for each forecast
        periodicity: 'D', 'MS', or 'YS', periodicity of the data for labeling future forecasts
        title: str to use as title of resultant plot
    
    returns Matplotlib Axes object containing plotted data
    '''
    # Create figure and axis, if needed
    if ax is None:
        fig, ax = plt.subplots(figsize = (12, 6))
    
    # Define model kwargs for future brevity
    model_kwargs = {'ax': ax, 'include_legend': False, 'uncertainty': uncertainties, 'plot_cap': False}
    
    # Invoke Prophet's plot function using the forecasts on current data
    model.plot(current_fc, **model_kwargs)
    
    # If there are any prediction periods, plot forecasts on future data
    if prediction_periods is not None:
        # Number of partitions of future data
        n = len(prediction_periods)
        
        # Plot predictions on last data point of current forecast up to first partition of future forecasts
        model.plot(pd.concat([current_fc.iloc[-1:], future_fc.iloc[0:prediction_periods[0]]]), **model_kwargs);
        
        # For the remaining partitions, plot the predictions on that partition
        for i, p in enumerate(prediction_periods[:-1]):
            if i == n - 1:
                model.plot(future_fc.iloc[p - 1:], **model_kwargs);
            else:
                model.plot(future_fc.iloc[p - 1:prediction_periods[i + 1]], **model_kwargs);
    # If there are no prediction periods, just plot the forecasts on the future data
    else:
        n = 1
        model.plot(future_fc, **model_kwargs);
    
    # Format the partitions of the future data and create a legend, if applicable
    if uncertainties:
        format_future_artists(ax = ax, n_periods = n,
                              prediction_periods = prediction_periods,
                              periodicity = periodicity);
        
    # Set axis and title labels
    set_text_params(ax, title = title, xlabel = '', ylabel = 'Number of Sunspots');
    
    # If the number of prior periods is specified, subset the plot
    if prior_periods is not None:
        ax.set_xlim([current_fc.tail(prior_periods).ds.values[0], future_fc.ds.tail(1).values[0]]);

    return ax

def plot_multi_timeframes(model, current_fc, future_fc, prior_periods, prediction_periods = None,
                          uncertainties = True, periodicity = 'D', titles = None, ax = None):
    '''
    Plots forecasts of a Prophet model with different levels of periods prior to future forecasts included, effectively
        showing the graph zoomed in at different levels. There is an optional ability to create partitions of future
        predictions, and the choice of whether to plot uncertainty intervals.
        
    args:
        model: Trained Prophet model to use for invoking model.plot()
        current_fc: Pandas DataFrame containing forecasts on current data
        future_fc: Pandas DataFrame containing forecasts on future data
        prior_periods: list of int, number of periods prior to future forecasts to include in each subplot,
            the different "zoom" levels for each subplot
        prediction_periods: None or list of int, partition of future periods to use for future forecasts,
            must not exceed length of future_fc. If None, forecast on future data will not be partitioned.
        uncertainties: bool, whether to plot uncertainty intervals for each forecast
        periodicity: 'D', 'MS', or 'YS', periodicity of the data for labeling future forecasts
        titles: list of str to use as titles of resultant subplots
        ax: None or Matplotlib Axes object to plot on, if None, a new axis is created using plt.subplots()
    
    returns Matplotlib Axes object containing plotted data    
    '''
    
    # Define the number of prior periods (the "zoom") to plot
    n = len(prior_periods)
    
    # If axis is not defined, create a new axis
    if ax is None:    
        fig, ax = plt.subplots(n, 1, figsize = (12, n * 6), sharey = True)
    
    # If titles are not defined, create a list of empty strings
    if titles is None:
        titles = [''] * n
    
    # For each prior period, plot Prophet forecasts
    for i, a in enumerate(ax):
        a = plot_future(model, current_fc, future_fc, prior_periods[i],
                        prediction_periods = prediction_periods,
                        uncertainties = uncertainties, periodicity = periodicity,
                        title = titles[i], ax = a)
    
    plt.tight_layout()
    
    return ax

def format_artists_model_comparisons(ax, n_colors, model_names, n_future):
    '''
    Formats lines from a shared Prophet plot without uncertainty intervals
        to distinguish between different models using color.
    
    args:
        ax: Matplotlib Axes object containing plotted data
        n_colors: int, number of colors to use for creating the Seaborn hls_palette()
        model_names: list of str, labels to use for each model for drawing the legend
        n_future: int, number of partitions of future forecast data
    '''
    # Get the lines from the plot
    lines = ax.get_lines()

    # Forecasts are the odd-numbered lines, observed data points are the even-numbered lined
    fcasts = lines[1::2]
    points = lines[::2]
    
    # Define the color palette to use for lines
    colors = sns.hls_palette(n_colors = n_colors, h = 0.5, l = 0.4, s = 1)

    # For each forecast, set the appropriate color
    # Forecasts are of the form current_fc, future_fc_1, ..., future_fc_n_future
    # Need to color each of the forecasts but only need to label the first forecast for each model
    for i, fcast in enumerate(fcasts):
        fcast.set_color(colors[i // (n_future + 1)])
        
        # If this forecast is the first for the model, set the label for the artist to the model name
        if i % (n_future + 1) == 0:
            fcast.set_label(model_names[i // (n_future + 1)])
        # Otherwise, set the label for the artist to empty so that it does not appear in the legend
        else:
            fcast.set_label('')

    # Delete all of the observed data points except for the first artist
    # For the first artist, give it a label so that it appears in the legend
    # and lower the opacity so that the overlaid forecasts are more visible
    for i, point in enumerate(points):
        if i != 0:
            point.remove()
        else:
            point.set_color('black')
            point.set_alpha(0.15)
            point.set_label('Observed Data Points')

    ax.legend();
    
    return

def plot_model_comparisons(models, model_names, forecasts, prior_periods, prediction_periods, titles):
    '''
    Plots multiple Prophet models on the same graph for comparison, excluding uncertainty intervals for clarity.
    
    args:
        models: fitted Prophet models to use for plotting
        model_names: list of str, labels to use for each model for drawing the legend
        forecasts: list of tuples of Pandas dataframes, where each tuple are the forecasts of a model on
            current data and the forecasts on future data
        prior_periods: None or list of int, number of periods prior to future forecasts to include in each subplot,
            the different "zoom" levels for each subplot
        prediction_periods: None or list of int, partition of future periods to use for future forecasts,
            must not exceed length of future_fc. If None, forecast on future data will not be partitioned.
        titles: list of str to use as titles of resultant subplots
        
    returns Matplotlib Axes object with plotted data
    '''
    # Define number of prior periods
    if prior_periods is None:
        p = 1
    else:
        p = len(prior_periods)
    # Define number of partitions for future periods
    if prediction_periods is None:
        f = 1
    else:
        f = len(prediction_periods)
    
    # Create subplots for the number of prior periods
    fig, ax = plt.subplots(p, 1, figsize = (12, 12));

    # For each model, plot the model on the appropriate axis.
    n = len(models)
    
    for i in range(n):
        if p == 1:
            ax = plot_future(models[i], forecasts[i][0], forecasts[i][1],
                             prior_periods = prior_periods,
                             prediction_periods = prediction_periods,
                             title = titles[0], ax = ax, uncertainties = False);
        else:
            ax = plot_multi_timeframes(models[i], forecasts[i][0], forecasts[i][1],
                                       prior_periods = prior_periods,
                                       prediction_periods = prediction_periods,
                                       titles = titles, ax = ax, uncertainties = False);

    # Format legend
    if p == 1:
        format_artists_model_comparisons(ax, n, model_names, f)
    else:
        for a in ax:
            format_artists_model_comparisons(a, n, model_names, f)
        
    return ax
    
def calculate_errors(obs, pred, model_name = None):
    '''
    Calculates the Mean Average Error (MAE), Mean Average Percentage Error (MAPE) and R^2 Score for a model.
    
    args:
        obs: Pandas dataframe with observed values
        pred: Pandas dataframe with predicted values
        model_name: None or str to use for labeling the model
    
    returns Pandas dataframe with metrics as columns and model as row
    '''
    if model_name is None:
        model_name = 'Model Errors'
        
    mae = mean_absolute_error(obs.y, pred.yhat)
    r2 = r2_score(obs.y, pred.yhat)
    
    # Subset the data where y > 0  to exclude zero and missing data
    # as very small values inflate the MAPE
    positive_idx = obs[obs.y > 0].index.tolist()
    reduced_obs = obs.loc[positive_idx]
    reduced_pred = pred.loc[positive_idx]
    
    mape = mean_absolute_percentage_error(reduced_obs.y, reduced_pred.yhat)
    
    error_frame = pd.DataFrame(data = {model_name: [mae, mape, r2]}, index = ['mae', 'mape', 'r2']).T
    
    return error_frame

def multi_model_errors(obs_preds_names):
    '''
    Calculates the MAE, MAPE, and R^2 Scores for multiple models and wraps the results in a dataframe.
    
    args:
        obs_preds_names: list of tuples of form (observation dataframe, prediction dataframe, model_name)
    
    returns Pandas dataframe with metrics as columns and models as rows
    '''
    error_frames = []
    for obs, pred, name in obs_preds_names:
        error_frames.append(calculate_errors(obs, pred, name))
                            
    return pd.concat(error_frames)