import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def nunique_val(df, *columns):
    for col in columns:
        print(f'Количество уникальных значений в столбце {col} - {df[col].nunique()}')


def picture_rooms_price(df):
    '''Выводим график зависимости цены от количества комнат'''

    df = df.copy()
    rooms_price_group = df.groupby('rooms')['price_rub'].agg(['mean', 'median']).reset_index()

    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(15,5))

    sns.barplot(
        data=rooms_price_group,
        x='rooms',
        y='mean',
        hue='rooms',
        ax=ax[0],
        legend=False
    )
    ax[0].set_title('Средняя стоимость недвижимости взависимости от комнат')
    ax[0].set_ylabel('Средняя стоимость')
    ax[0].set_xlabel('Количество комнат')

    sns.barplot(
        data=rooms_price_group,
        x='rooms',
        y='median',
        hue='rooms',
        ax=ax[1],
        legend=False
    )
    ax[1].set_title('Медианная стоимость недвижимости взависимости от комнат')
    ax[1].set_ylabel('Медианная стоимость')
    ax[1].set_xlabel('Количество комнат')


def count_rooms(df):
    print('Общее количество количество квартир по числу комнат')
    display(df.groupby('rooms').count().T.iloc[0].to_frame().T.rename({'area': 'count'}))


def picture_area_price(df):
    '''Выводим график зависимости цены от площади (в разрезах)'''

    df = df.copy()
    bins=[0, 50, 100, 150, 200, 250, 300, 350, 400, df['area'].max()+1]
    df['area_bin'] = pd.cut(df['area'], bins=bins, right=False)
    print('Количество комнат в диапозонах:')
    display(df.groupby('area_bin', observed=True).count().T.iloc[0].to_frame().T)

    bins_area_group = df.groupby('area_bin', observed=True)['price_rub'].agg(['mean', 'median']).astype(int)
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(15,5))

    sns.barplot(
        data=bins_area_group,
        x='area_bin',
        y='mean',
        hue='area_bin',
        ax=ax[0],
        legend=False
    )
    ax[0].set_title('Средняя стоимость недвижимости взависимости от площади')
    ax[0].set_ylabel('Средняя стоимость')
    ax[0].set_xlabel('Диапозон площади')
    ax[0].tick_params(rotation=45)

    sns.barplot(
        data=bins_area_group,
        x='area_bin',
        y='median',
        hue='area_bin',
        ax=ax[1],
        legend=False
    )
    ax[1].set_title('Медианная стоимость недвижимости взависимости от площади')
    ax[1].set_ylabel('Медианная стоимость')
    ax[1].set_xlabel('Диапозон площади')
    ax[1].tick_params(rotation=45)


def picture_metro_price(df):
    '''Выводим график зависимости цены от станции метро'''

    df = df.copy()
    df = df[df['metro'].isin(df['metro'].value_counts().iloc[:15].index)]
    metro_price_group = df.groupby('metro', observed=True)['price_rub'].agg(['mean', 'median'])
    display(metro_price_group.T.map(lambda x: f'{x:,.0f}'))

    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(15,5))

    sns.barplot(
        data=metro_price_group.sort_values(by='mean'),
        x='metro',
        y='mean',
        hue='metro',
        ax=ax[0],
        legend=False
    )
    ax[0].set_title('Средняя стоимость недвижимости взависимости от станции метро')
    ax[0].set_ylabel('Средняя стоимость')
    ax[0].set_xlabel('Наименование станций метро')
    ax[0].tick_params(rotation=87)

    sns.barplot(
        data=metro_price_group.sort_values(by='median'),
        x='metro',
        y='median',
        hue='metro',
        ax=ax[1],
        legend=False
    )
    ax[1].set_title('Медианная стоимость недвижимости взависимости от станции метро')
    ax[1].set_ylabel('Медианная стоимость')
    ax[1].set_xlabel('Наименование станций метро')
    ax[1].tick_params(rotation=87)


def picture_district_price(df):
    '''Выводим график зависимости цены от района'''

    df = df.copy()
    df = df[df['district'].isin(df['district'].value_counts().iloc[:15].index)]
    district_price_group = df.groupby('district', observed=True)['price_rub'].agg(['mean', 'median'])
    display(district_price_group.T.map(lambda x: f'{x:,.0f}'))

    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(15,5))
    
    sns.barplot(
        data=district_price_group.sort_values(by='mean'),
        x='district',
        y='mean',
        hue='district',
        ax=ax[0],
        legend=False
    )
    ax[0].set_title('Средняя стоимость недвижимости взависимости от района')
    ax[0].set_ylabel('Средняя стоимость')
    ax[0].set_xlabel('Наименование района')
    ax[0].tick_params(rotation=87)

    sns.barplot(
        data=district_price_group.sort_values(by='median'),
        x='district',
        y='median',
        hue='district',
        ax=ax[1],
        legend=False
    )
    ax[1].set_title('Медианная стоимость недвижимости взависимости от района')
    ax[1].set_ylabel('Медианная стоимость')
    ax[1].set_xlabel('Наименование района')
    ax[1].tick_params(rotation=87)


def picture_district_price_ascending(df):
    '''Выводим график самых дорогостоющих районов'''

    df = df.copy()
    district_price = df.groupby('district')['price_rub'].mean().sort_values(ascending=False).iloc[:30].to_frame().sort_values(by='price_rub', ascending=True)
    display(district_price.T.map(lambda x: f'{x:,.0f}'))
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(20,5))

    sns.barplot(
        data=district_price,
        x='district',
        y='price_rub',
        hue='district',
        ax=ax
    )
    ax.set_title('Стоимость недвижимости в районах (самые дорогостоющие районы)')
    ax.set_ylabel('Цена недвижимости')
    ax.set_xlabel('Самые дорогостоющие районы')
    ax.tick_params(rotation=87)
    
    
