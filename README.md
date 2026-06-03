# 🏠 Прогнозирование цен на квартиры в Москве

Проект по предсказанию стоимости квартир на основе данных с сайта ЦИАН.  
Охватывает полный цикл: сбор данных → EDA → Feature Engineering → Моделирование.

---

## 📊 Результаты модели

| Метрика | Train | Test |
|---|---|---|
| MAE | 5,833,865 ₽ | 12,692,936 ₽ |
| RMSE | 10,041,049 ₽ | 34,802,736 ₽ |
| R² | 0.9921 | 0.8901 |

**Финальная модель:** CatBoost с подбором гиперпараметров через Optuna (CV=5)

---

## 📁 Структура проекта

```
PROJECT CIAN/
├── Parsers/
│   ├── parser_cian.py                 # Парсер объявлений с ЦИАН
│   └── parser_exchange_rates.py       # Парсер курсов валют с ЦБ РФ
│
├── Работа с данными/
│   ├── data_preprocessing.py          # Очистка и предобработка данных
│   ├── EDA.py                         # Функции для разведочного анализа
│   ├── SHOW_data_preprocessing.ipynb  # Ноутбук предобработки
│   └── SHOW_EDA.ipynb                 # Ноутбук EDA
│
├── ML/
│   ├── feature_engineering.py         # Кастомные трансформеры для пайплайна
│   └── SHOW_feature_engineering.ipynb # Ноутбук ML
│
├── data/
│   ├── full data/
│   │   └── cian_all_moscow_full.csv   # Собранные данные (25k объявлений)
│   └── additional data/
│       └── exchange_rate.csv          # Курсы валют
│
├── .gitignore
└── README.md
```

---

## 🔄 Пайплайн

```python
Pipeline([
    ('mkad',     OutsideInsideMetro()),       # Признак: внутри/за МКАД
    ('district', DistrictKMeansEncoder()),    # KMeans кластеризация районов
    ('minutes',  CoffForMetroMinutes()),      # Корректировка времени до метро
    ('metro',    MetroTargetEncoder()),       # Target encoding станций метро
    ('seller',   SellerTypeDrop()),           # Удаление малозначимого признака
    ('drop',     DropColumns(['in_moscow'])), # Удаление промежуточного признака
    ('cb',       CatBoostRegressor(...))      # Финальная модель
])
```

---

## 📈 Сравнение моделей

| Модель | MAE test | RMSE test | R² test |
|---|---|---|---|
| LinearRegression | 27,577,960 ₽ | 55,754,023 ₽ | 0.72 |
| XGBoost | 14,248,333 ₽ | 39,051,719 ₽ | 0.86 |
| CatBoost baseline | 14,228,158 ₽ | 36,521,350 ₽ | 0.88 |
| **CatBoost + Optuna** | **12,692,936 ₽** | **34,802,736 ₽** | **0.89** |

---

## 🔍 Feature Engineering

| Трансформер | Описание |
|---|---|
| `OutsideInsideMetro` | Бинарный признак — станция внутри/за МКАД |
| `DistrictKMeansEncoder` | Замена района на кластер (n=5, silhouette=0.51) |
| `CoffForMetroMinutes` | Умножение времени на 2.5 если transport |
| `MetroTargetEncoder` | Замена станции на среднюю цену по ней |
| `SellerTypeDrop` | Удаление seller_type (не влияет на цену) |

---

## 🔬 EDA — ключевые выводы

- **Площадь** — самый важный признак (SHAP)
- **Станция метро** — второй по важности признак
- **Тип транспорта до метро** — квартиры у метро пешком дороже на 10-24% (подтверждено тестом Манна-Уитни)
- **За МКАД** — квартиры кратно дешевле чем внутри МКАД
- **Тип продавца** — не влияет на цену, удалён из модели

---

## ⚠️ Ограничения модели

- Квартиры дороже 300 млн (~4% выборки) вносят ~40% общей ошибки
- Отсутствуют признаки: год постройки, тип дома, состояние ремонта
- Данные актуальны на момент сбора (2026 год)

---

## 🛠 Установка

```bash
git clone https://github.com/lopa8in/PROJECT-CIAN.git
cd PROJECT-CIAN
python -m venv venv
source venv/bin/activate  # Mac/Linux
pip install -r requirements.txt
```

---

## 📦 Основные библиотеки

| Библиотека | Назначение |
|---|---|
| `pandas`, `numpy` | Обработка данных |
| `scikit-learn` | Пайплайн и метрики |
| `catboost` | Финальная модель |
| `xgboost` | Сравнение моделей |
| `optuna` | Подбор гиперпараметров |
| `shap` | Интерпретация модели |
| `mlflow` | Логирование экспериментов |
| `requests` | Парсинг данных |
| `geopy` | Геокодирование адресов |
| `scipy` | Статистические тесты |
| `seaborn`, `matplotlib` | Визуализация |

---

## 🚀 Воспроизведение результатов

1. Запустить `Parsers/parser_cian.py` — сбор данных с ЦИАН
2. Запустить `Parsers/parser_exchange_rates.py` — курсы валют
3. Открыть `Работа с данными/SHOW_data_preprocessing.ipynb` — предобработка
4. Открыть `Работа с данными/SHOW_EDA.ipynb` — разведочный анализ
5. Открыть `ML/SHOW_feature_engineering.ipynb` — обучение модели

> **Примечание:** в ноутбуке представлена только финальная модель.  
> Сравнение с LinearRegression и XGBoost приведено в таблице выше.
---

## 📋 Генерация requirements.txt

```bash
pip freeze > requirements.txt
```