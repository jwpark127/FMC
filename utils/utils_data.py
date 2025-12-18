import os
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler, MinMaxScaler

def normalizefea(X, norm):
    """
    Normalize
    """
    standard_scaler = StandardScaler()
    scaled_X = standard_scaler.fit_transform(X)
    if norm == 'L2':
        feanorm = np.maximum(1e-14,np.sum(scaled_X**2,axis=1))
        X_out = scaled_X/(feanorm[:,None]**0.5)
        return X_out
    if norm == 'standard':
        return scaled_X

def read_dataset(data_dir, dataset, n_sensitive, normalize='L2'):
    assert dataset in ['adult', 'bank', 'credit', 'census']
    if dataset == 'adult':
        _path = 'adult.data'
        data_path = os.path.join(data_dir,_path)

        df = pd.read_csv(data_path, sep=',', header=None)

        n = df.shape[0]
        sens_attr = 9
        sex = df.iloc[:, sens_attr]
        sens_attributes =[' Male', ' Female']
        df_no_sensitive = df.drop(columns=[sens_attr])
        sens_numpy = np.zeros(n, dtype=int)
        sens_numpy[sex.astype(str).values == sens_attributes[1]] = 1

        cont_cols = [0,2,4,10,12]
        cate_cols = [1,3,5,6,7,8,13]
        clss_cols = [14]

        for col in df_no_sensitive.select_dtypes(include='int64').columns:
            df_no_sensitive[col] = df_no_sensitive[col].astype('float')

        for col in df_no_sensitive.select_dtypes(include='object').columns:
            df_no_sensitive[col] = df_no_sensitive[col].astype('category')
            df_no_sensitive[col] = df_no_sensitive[col].cat.codes

    elif dataset == 'bank':
        _path = 'bank-additional-full.csv'
        data_path = os.path.join(data_dir,_path)
        pd_bank_data = pd.read_csv(data_path, sep = ';')
        n = pd_bank_data.shape[0]
        if n_sensitive == 2:
            bank_data = pd_bank_data.loc[~pd_bank_data['marital'].isin(['unknown'])]
            bank_data.loc[bank_data['marital'] == 'divorced', 'marital'] = 'single'
        elif n_sensitive == 3:
            bank_data = pd_bank_data.loc[~pd_bank_data['marital'].isin(['unknown'])]
        elif n_sensitive == 4:
            bank_data = pd_bank_data
        marriage = pd.DataFrame(bank_data.loc[:,'marital'])
        marriage = (marriage.astype('category')['marital'].cat.codes)
        df_no_sensitive = bank_data.drop(columns=['marital'])

        sens_numpy = marriage.to_numpy()

        cont_cols = 'age, duration, euribor3m, nr.employed, cons.price.idx, campaign'.split(', ')
        cate_cols = 'job, education, default, housing, loan, contact, month, day_of_week, poutcome, y'.split(', ')
        for col in cont_cols:
            df_no_sensitive[col] = df_no_sensitive[col].astype('float')

        for col in cate_cols:
            df_no_sensitive[col] = df_no_sensitive[col].astype('category')
            df_no_sensitive[col] = df_no_sensitive[col].cat.codes
    
    elif dataset == 'credit':
        _path = 'creditcard.csv'
        data_path = os.path.join(data_dir,_path)
        pd_credit_data_origin = pd.read_csv(data_path, sep=',',header=0)

        pd_credit_data = pd_credit_data_origin.drop(columns=['ID','default payment next month'])

        n = pd_credit_data.shape[0]
        if n_sensitive == 2:
            pd_credit_data.loc[pd_credit_data['EDUCATION'].isin([0,3,5,4,6]), 'EDUCATION'] = 1
        elif n_sensitive == 3:
            pd_credit_data.loc[pd_credit_data['EDUCATION'].isin([0,3,5,4,6]), 'EDUCATION'] = 0
        elif n_sensitive == 4:
            pd_credit_data.loc[pd_credit_data['EDUCATION'].isin([0,5,4,6]), 'EDUCATION'] = 0
        education = pd.DataFrame(pd_credit_data.loc[:,'EDUCATION'])
        education = (education.astype('category')['EDUCATION'].cat.codes)
        df_no_sensitive = pd_credit_data.drop(columns = ['EDUCATION'])

        sens_numpy = education.to_numpy()

        cont_cols = 'LIMIT_BAL,AGE,BILL_AMT1,PAY_AMT1'.split(',')
        cate_cols = 'SEX,MARRIAGE,PAY_0,PAY_2,PAY_3,PAY_4,PAY_5,PAY_6'.split(',')
        for col in cont_cols:
            df_no_sensitive[col] = df_no_sensitive[col].astype('float')

        for col in cate_cols:
            df_no_sensitive[col] = df_no_sensitive[col].astype('category')
            df_no_sensitive[col] = df_no_sensitive[col].cat.codes
    
    elif dataset == 'census':
        _path = 'USCensus1990raw.data.txt'
        data_path = os.path.join(data_dir,_path)
        df_no_sensitive = pd.read_csv(data_path, sep='\t', header = None)
        sens_numpy = df_no_sensitive.iloc[:,112].astype(int).values
        cont_cols = [12,35,36,47,53,54,55,58,60,63,64,65,73,80,81,93,94,95,96,101,109,116,118,122,124]
        cate_cols = [0,1,2,3,4,5,6,7,8,9]


    df_cont = df_no_sensitive.loc[:,cont_cols]
    df_cate = df_no_sensitive.loc[:,cate_cols]
    data_cont = np.array(df_cont, dtype = float)
    data_cate = np.array(df_cate, dtype = np.int8)

    normalized_cont = normalizefea(data_cont, normalize)
    cont_data = torch.tensor(normalized_cont)
    cate_data = torch.tensor(data_cate)
    sens_data = torch.tensor(sens_numpy)

    if dataset == 'adult':
        df_clss = df_no_sensitive.loc[:,clss_cols]
        data_clss = np.array(df_clss, dtype = np.int8)
        clss_data = torch.tensor(data_clss)

        return cont_data, cate_data, sens_data, clss_data

    else:
        return cont_data, cate_data, sens_data