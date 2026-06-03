import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns


def train_test(df):
    X = df.drop(columns=['price_rub'])
    y = df['price_rub']
    return train_test_split(X,y, test_size=0.3,random_state=42)


class OutsideInsideMetro(BaseEstimator, TransformerMixin):
    '''Данный класс создает булевый признак is_moscow, где:
        1 - станция метро находится внутри МКАД
        0 - станция метро находится за МКАДом'''
    
    def __init__(self):
        self.outside_metro = [
                            "Аэропорт Внуково", "Бачуринская", "Боровское шоссе", "Бульвар Адмирала Ушакова",
                            "Бульвар Дмитрия Донского", "Бунинская аллея", "Волоколамская", "Говорово",
                            "Жулебино", "Коммунарка", "Корниловская", "Косино", "Котельники", "Лианозово",
                            "Лухмановская", "Мамыри", "Митино", "Мякинино", "Некрасовка", "Новокосино",
                            "Новомосковская", "Новопеределкино", "Ольховая", "Прокшино", "Пыхтино",
                            "Пятницкое шоссе", "Рассказовка", "Румянцево", "Саларьево", "Солнцево",
                            "Тютчевская", "Улица Горчакова", "Улица Дмитриевского", "Улица Скобелевская",
                            "Улица Старокачаловская", "Филатов Луг", "Физтех", "Ховрино"
                            ]
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        df = X.copy()
        df['in_moscow'] = (~df['metro'].isin(self.outside_metro)).astype(int)
        return df
    

class CoffForMetroMinutes(BaseEstimator, TransformerMixin):
    '''Данный класс изменяет столбец metro_minutes.
       Если metro_transport='transport' умножаем
       metro_minutes на коэффициент 2.5'''
    
    def __init__(self, coff=2.5):
        self.coff = coff

    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        df = X.copy()
        df['metro_minutes'] = np.where(
                                        df['metro_transport'] == 'transport',
                                        df['metro_minutes'] * self.coff,
                                        df['metro_minutes']
                                        )
        df = df.drop(columns=['metro_transport'])
        return df
    

class DistrictKMeansEncoder(BaseEstimator, TransformerMixin):
    '''Кластеризует районы по статистикам и заменяет
       название района на номер кластера'''
    
    def __init__(self, n_clusters=5, random_state=42):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.kmeans_ = None
        self.scaler_ = StandardScaler()
        self.district_clusters_ = {}
        self.default_cluster_ = None

    def fit(self, X, y):
        df_x = X.copy()
        df_y = y.copy()

        df = pd.concat([df_x, df_y], axis=1)

        district_group = df.groupby('district').agg(
                                                    avg_minutes = ('metro_minutes', 'mean'),
                                                    walk_share = ('metro_transport', lambda x: (x == 'walk').mean()),
                                                    in_moscow = ('in_moscow', 'mean'),
                                                    price_med = ('price_rub', 'median'),
                                                    price_mean = ('price_rub', 'mean')
                                                    ).reset_index()
        
        district_group_drop = district_group.drop(columns='district')
        
        district_group_scaler = self.scaler_.fit_transform(district_group_drop)

        self.kmeans_ = KMeans(n_clusters=self.n_clusters, random_state=self.random_state)
        labels = self.kmeans_.fit_predict(district_group_scaler)

        self.district_clusters_ = dict(zip(district_group['district'], labels))
        self.default_cluster_ = int(pd.Series(labels).mode()[0])

        return self
    
    def transform(self, X):
        df = X.copy()
        df['district_cluster'] = df['district'].map(self.district_clusters_).fillna(self.default_cluster_).astype(int)
        df = df.drop(columns=['district'])
        return df
        

class MetroTargetEncoder(BaseEstimator, TransformerMixin):

    def __init__(self):
        self.metro_map_ = None
        self.default_ = None

    def fit(self, X, y):
        df_x = X.copy()
        df_y = y.copy()

        df = pd.concat([df_x, df_y], axis=1)

        self.metro_map_ = df.groupby('metro')['price_rub'].mean().to_dict()
        self.default_ = df_y.mean()

        return self
    
    def transform(self, X):
        df = X.copy()
        df['metro_encoded'] = df['metro'].map(self.metro_map_).fillna(self.default_)
        df = df.drop(columns=['metro'])
        return df
    

class SellerTypeDrop(BaseEstimator, TransformerMixin):

    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        df = X.copy()
        df = df.drop(columns=['seller_type'])
        return df

#####################################################################################################

def picture_cluster(df):
    '''Вычисляем оптимальное количество кластеров'''

    is_moscow = OutsideInsideMetro()
    df = is_moscow.fit_transform(df)

    district_group = df.groupby('district').agg(
                                                avg_minutes = ('metro_minutes', 'mean'),
                                                walk_share = ('metro_transport', lambda x: (x == 'walk').mean()),
                                                in_moscow = ('in_moscow', 'mean'),
                                                price_med = ('price_rub', 'median'),
                                                price_mean = ('price_rub', 'mean')
                                                ).reset_index().drop(columns=['district'])
    
    scaler = StandardScaler()
    district_scaler = scaler.fit_transform(district_group)
    
    tsne = TSNE(n_components=2, random_state=42)
    X_2d = tsne.fit_transform(district_scaler)

    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(13,5))
    sns.scatterplot(x=X_2d[:, 0], y=X_2d[:, 1], ax=ax[0])
    ax[0].set_title('Проекция TSNE')

    inertias = []
    for k in range(2, 12):
        km = KMeans(n_clusters=k, random_state=42)
        km.fit(district_scaler)
        inertias.append(km.inertia_)

    sns.lineplot(x=range(2, 12), y=inertias, marker='o', ax=ax[1])
    ax[1].set_title('Elbow Method')  
    ax[1].set_xlabel('n_clusters')
    ax[1].set_ylabel('Inertia')

    for k in range(2, 12):
        km = KMeans(n_clusters=k, random_state=42)
        labels = km.fit_predict(district_scaler)
        score = silhouette_score(district_scaler, labels)
        print(f"n_clusters={k}: silhouette={score:.4f}")


def picture_hue_clusters(df, n_clusters=5):
    '''Визуализируем кластеризацию с оптимальным количеством кластеров'''
    is_moscow = OutsideInsideMetro()
    df = is_moscow.fit_transform(df)

    district_group = df.groupby('district').agg(
                                                    avg_minutes = ('metro_minutes', 'mean'),
                                                    walk_share = ('metro_transport', lambda x: (x == 'walk').mean()),
                                                    in_moscow = ('in_moscow', 'mean'),
                                                    price_med = ('price_rub', 'median'),
                                                    price_mean = ('price_rub', 'mean')
                                                    ).reset_index().drop(columns=['district'])
    
    scaler = StandardScaler()
    district_scaler = scaler.fit_transform(district_group)

    tsne = TSNE(n_components=2, random_state=42)
    X_2d = tsne.fit_transform(district_scaler)

    km = KMeans(n_clusters=n_clusters, random_state=42)
    labels = km.fit_predict(district_scaler)

    sns.scatterplot(x=X_2d[:, 0], y=X_2d[:, 1], hue=labels, palette='tab10')


class DropColumns(BaseEstimator, TransformerMixin):
    
    def __init__(self, columns):
        self.columns = columns
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        return X.drop(columns=self.columns)