import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def count_duplicated(df):
    print('Количество дубликатов -', df.duplicated().sum())


def count_pass(df):
    print('Количество пропусков по столбцам:\n', df.isnull().sum())
    fig = plt.figure(figsize=(10,8))
    sns.heatmap(data=df.isnull())
    plt.show()


def drop_not_information_columns(df, *columns):
    df = df.drop(columns=list(columns))
    return df


def correct_valute(df):
    data_currency = pd.read_csv('/Users/muzaladinovdzamalladinruslanovic/Desktop/PROJECT CIAN/data/additional data/exchange_rate.csv') # чтение файла с курсами валют
    data_currency = data_currency.rename(columns={'Unnamed: 0': 'valute', '0': 'rate'})

    df = pd.merge(df, data_currency, how='left', left_on='currency', right_on='valute') # объединение по валюте
    df['price_rub'] = df['rate'] * df['price']
    df = df.drop(columns=['price', 'currency', 'valute', 'rate'])
    return df


def fillna_pass(df, **columns_val):
    df = df.fillna(columns_val)
    return df


def dropnan(df, *columns):
    df = df.dropna(subset=columns)
    return df