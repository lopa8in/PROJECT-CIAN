import requests
import xml.etree.ElementTree as ET
import pandas as pd

r = requests.get("https://www.cbr.ru/scripts/XML_daily.asp", timeout=10)
r.encoding = "windows-1251"
root = ET.fromstring(r.text)

rates = {"rur": 1.0}
for valute in root.findall("Valute"):
    char_code = valute.find("CharCode").text.lower()
    nominal = int(valute.find("Nominal").text)
    value = float(valute.find("Value").text.replace(",", "."))
    rates[char_code] = value / nominal

exchange_rate = pd.Series(rates)
exchange_rate.to_csv('exchange_rate.csv')