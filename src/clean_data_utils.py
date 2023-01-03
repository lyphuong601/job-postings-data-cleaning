import os
import pandas as pd
import numpy as np

pd.set_option('display.max_rows', None)
pd.set_option('display.expand_frame_repr', False)

##########################################################
# Data cleaning functions
##########################################################
def load_data(filename: str) -> pd.DataFrame:
    """Read data from a filename and output it as a dataframe"""
    df = pd.read_csv(filename)
    return df


def check_missing_data(data: pd.DataFrame) -> pd.Series:
    """Check for missing data in the df (display in descending order)"""
    result = ((data.isnull().sum() * 100)/ len(data)).sort_values(ascending=False)
    return result


def fill_na_data(data: pd.DataFrame, column_names: list, fill_value="NA") -> None:
    """Fill out null data with appropriate value, default value as `NA`"""
    for column in column_names:
        data[column].fillna(fill_value, inplace=True)
    

def drop_columns(data: pd.DataFrame, column_list: list, inplace=True) -> None:
    """Drop a list of columns from dataset"""
    data.drop(column_list, axis=1, inplace=inplace)


def split_string(data: pd.Series, split_by: str) -> pd.Series:
    """Split data in each row by a parameter"""
    return [row.split(split_by)[0].strip() for row in data]


def replace_string(data: pd.Series, old_value: str, new_value: str) -> pd.Series:
    """Replace data in each row with a new String value"""
    return [row.replace(old_value, new_value) for row in data]


def remove_character(data: pd.Series, to_replace=' &#.*', replace_value ='') -> None:
    """Remove all the characters in a string in a dataframe column default to remove &#.*"""
    data.replace(to_replace, replace_value, regex=True, inplace=True)


def create_bins(data: pd.Series, no_bin: int, group_name: list) -> pd.Series:
    """Cut data into different bins and assign names for each bin"""
    bins = np.linspace(data.min(), data.max(), no_bin)
    data = pd.cut(data, bins, labels=group_name, include_lowest=True)
    return data


def change_dtypes(data: pd.DataFrame, column_names: list, to_type='int32') -> pd.DataFrame:
    """Change data type of columns, default type is set to `int32`"""
    data[column_names] = data[column_names].astype(to_type)
    return data


def filter_data(data: pd.DataFrame, column_name: str, column_to_exclude: list) -> pd.DataFrame:
    """Create mask and filter Dataframe based on the specified conditions
    by excluding a list of columns from the resulting dataset"""
    mask = ~data[column_name].isin(column_to_exclude)
    return data[mask]


def impute_outlier_data(data:pd.Series, old_value: list, new_value=np.NAN) -> pd.Series:
    """Fill out specified outliers data with approriate values, default value as `NA`"""
    data = np.where(data.isin(old_value), new_value, data)
    return data

def get_job_title(title: str) -> str:
    """Categorize each entry into different job titles based on keywords in each row"""
    if 'data scientist' in title.casefold():
        return 'data scientist'
    elif 'data engineer' in title.casefold():
        return 'data engineer'
    elif 'analyst' in title.casefold():
        return 'analyst'
    elif 'machine learning' in title.casefold():
        return 'mle'
    elif 'manager' in title.casefold():
        return 'manager'
    elif 'director' in title.casefold():
        return 'director'
    else:
        return 'na'
    
    
def seniority(title: str) -> str:
    """Categorize each entry into different seniority levels based on keywords in each row"""
    if 'jr.' in title.casefold():
        return 'junior'
    elif 'sr' in title.casefold() or 'senior' in title.casefold() or 'sr.' in title.casefold():
        return 'senior'
    elif 'lead' in title.casefold() or 'principal' in title.casefold() or 'manager' in title.casefold() or 'director' in title.casefold():
        return 'manager'
    else:
        return 'staff'


##########################################################
# Cleaning routine (Combined)
##########################################################
def process_data(df: pd.DataFrame) -> pd.DataFrame:
    """Process dataset using functions created above"""
    drop_columns(df, ['index'])
    df['Job Title'] = df['Job Title'].apply(lambda x: x.casefold())
    remove_character(df['Job Description'],'\n', ' ')
    
    df['Salary Estimate'] = split_string(df['Salary Estimate'], '(')
    df['Salary Estimate'] = replace_string(df['Salary Estimate'], "K", "000")
    df['Salary Estimate'] = replace_string(df['Salary Estimate'], "$", "")

    df['Company Name'] = split_string(df['Company Name'], "\n")
    df['Min Salary'] = df['Salary Estimate'].apply(lambda x: x.split('-')[0]).astype(int)
    df['Max Salary'] = df['Salary Estimate'].apply(lambda x: x.split('-')[1]).astype(int)
    df['Avg Salary'] = (df['Min Salary'] + df['Max Salary'])//2

    df['Rating'] = impute_outlier_data(df['Rating'], [-1.0], 0)
    df['Headquarters'] = impute_outlier_data(df['Headquarters'], ['-1'], 'Unknown')
    df['Size'] = impute_outlier_data(df['Size'], ['-1', 'Unknown'], 'Unknown')
    df['Founed'] = impute_outlier_data(df['Founded'], [-1], np.NAN)
    df['Type of ownership'] = impute_outlier_data(df['Type of ownership'], ['-1'], 'Unknown')
    df['Industry'] = impute_outlier_data(df['Industry'], ['-1'], 'Unknown')
    df['Sector'] = impute_outlier_data(df['Sector'], ['-1'], 'Unknown')
    df['Revenue'] = impute_outlier_data(df['Revenue'], ['-1'], 'Unknown / Non-Applicable')
    df['Competitors'] = impute_outlier_data(df['Competitors'], ['-1'], 'Unknown')
    
    df['Company Age'] = df['Founded'].apply(lambda x: x if x < 1 else 2021 - x)
    df['Company Age'] = impute_outlier_data(df['Company Age'], [-1.0], 0)

    df['Location State'] = df['Location'].apply(lambda x: x.split(',')[-1])
    df['Same Location As HQ'] = df.apply(lambda x: 1 if x['Location'] == x['Headquarters'] else 0, axis=1)

    df['Position'] = df['Job Title'].apply(get_job_title)
    df['Seniority'] = df['Job Title'].apply(seniority)
    return df
