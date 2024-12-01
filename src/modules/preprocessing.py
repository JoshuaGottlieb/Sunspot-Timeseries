import pandas as pd

def load_dataset(path):
    '''
    Loads a sunspot dataset and subsets the data on relevant columns.
    
    args:
        path: str, path to csv file
    
    returns: A Pandas dataframe subset on relevant columns.
    '''
    if path.split('/')[-1] == 'SN_y_tot_V2.0.csv':
        cols = [0, 1]
        column_names = ['year', 'predicted_sunspots']
    elif path.split('/')[-1] == 'SN_m_tot_V2.0.csv':
        cols = [0, 1, 3]
        column_names = ['year', 'month', 'predicted_sunspots']
    elif path.split('/')[-1] == 'SN_d_tot_V2.0.csv':
        cols = [0, 1, 2, 4]
        column_names = ['year', 'month', 'day', 'predicted_sunspots']
    else:
        print('Wrong file selected. File name should be SN_{d/m/y}_tot_V2.0.csv')
        return
    
    df = pd.read_csv(path, sep = ';', header = None, usecols = cols, names = column_names)
    
    return df    

def determine_periodicity(df):
    '''
    Determines the periodicity of a dataframe using value_counts() function.
    
    args:
        df: Pandas dataframe containing data
        
    returns str denoting periodicity of dataframe
    '''
    # If each year only has one value, then the data is yearly
    if df['year'].value_counts().max() == 1:
        return 'YS'
    
    # If each month only has one value, then the data is monthly
    if df[['year', 'month']].value_counts().max() == 1:
        return 'MS'
    
    # If each day only has one value, then the data is daily
    if df[['year', 'month', 'day']].value_counts().max() == 1:
        return 'D'
    
    # If none of the above, then the periodicity cannot be determined
    print('Unable to determine periodicity of dataframe')
    
    return

def format_date(x, periodicity):
    '''
    Creates a formatted date using columns from a row slice of a dataframe.
    
    args:
        x: Pandas dataframe row
        periodicity: 'D', 'MS', or 'YS' to determine periodicity of the data for formatting data
    '''
    if periodicity == 'D':
        return f'{int(x.year):04d}-{int(x.month):02d}-{int(x.day):02d}'
    
    if periodicity == 'MS':
        return f'{int(x.year):04d}-{int(x.month):02d}-01'
    
    if periodicity == 'YS':
        return f'{int(x.year):04d}-01-01'
    
    return

def create_periodic_dataframe(df, target_column = 'predicted_sunspots'):
    '''
    Creates a dataframe formatted for use by Prophet models (time column 'ds', target column 'y').
    
    args:
        df: Pandas dataframe containing data
        target_column: str, column to use as target
        
    returns tuple of formatted dataframe and str denoting periodicity of the data
    '''
    # Determine the periodicity
    periodicity = determine_periodicity(df)
    
    # If the periodicity was None, do not continue
    if periodicity is None:
        return
    
    # Create a new dataframe, extract date and target information
    periodic_df = pd.DataFrame()
    periodic_df['ds'] = pd.to_datetime(df.apply(lambda x: format_date(x, periodicity), axis = 1))
    periodic_df['y'] = df[target_column]
    
    return periodic_df, periodicity