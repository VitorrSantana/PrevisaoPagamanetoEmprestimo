import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.impute import SimpleImputer

def label_encoder(base,col): 
    le = preprocessing.LabelEncoder()
    array_transform = le.fit_transform(base[col].fillna('-1'))
    return array_transform

def set_col_nan(base,col,text):
    return base[col].replace(text,np.nan)

def set_nan_negative(base,col):
    return base[col].fillna(-1)

def replace_col(base,col,text_actual:str,text_new):
    return base[col].replace(text_actual,text_new)

def calc_days_between_date(base,date1,date2):
    return (pd.to_datetime(base[date1]) - pd.to_datetime(base[date2])).dt.days

def create_flag(condicao):
    return np.where(condicao,1,0)
