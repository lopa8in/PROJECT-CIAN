import requests
import time
import random
import csv
import json
import os

OUTPUT_FILE = "cian_all_moscow_full.csv"
PROGRESS_FILE = "cian_progress_full.json"
DELAY_NORMAL = (3, 6)
DELAY_BLOCKED = (60, 120)
MAX_RETRIES = 5

# Дробим по цене — так обходим лимит в 54 страницы
PRICE_RANGES = [
    (0, 5_000_000),
    (5_000_000, 8_000_000),
    (8_000_000, 11_000_000),
    (11_000_000, 15_000_000),
    (15_000_000, 20_000_000),
    (20_000_000, 30_000_000),
    (30_000_000, 50_000_000),
    (50_000_000, 100_000_000),
    (100_000_000, 999_000_000),
]

ROOM_TYPES = [1, 2, 3, 4, 5, 6, 9]
room_labels = {1: "1-комн", 2: "2-комн", 3: "3-комн",
               4: "4-комн", 5: "5-комн", 6: "6-комн", 9: "Студия"}

FIELDNAMES = [
    "id", "url", "price", "price_per_m2", "currency",
    "rooms", "area", "floor", "floors_total",
    "address", "metro", "metro_minutes", "metro_transport",
    "district", "description", "seller_type",
    "published", "room_type_query",
]

def make_session():
    session = requests.Session()
    session.headers.update({
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36",
        "Accept": "application/json",
        "Accept-Language": "ru-RU,ru;q=0.9",
        "Content-Type": "application/json",
        "Referer": "https://cian.ru/",
    })
    session.get("https://cian.ru/", timeout=10)
    time.sleep(random.uniform(2, 4))
    return session

def fetch_page(session, room_type, page_num, price_from, price_to):
    url = "https://api.cian.ru/search-offers/v2/search-offers-desktop/"
    payload = {
        "jsonQuery": {
            "_type": "flatsale",
            "engine_version": {"type": "term", "value": 2},
            "region": {"type": "terms", "value": [1]},
            "room": {"type": "terms", "value": [room_type]},
            "page": {"type": "term", "value": page_num},
            "price": {
                "type": "range",
                "value": {"gte": price_from, "lte": price_to}
            },
        }
    }
    return session.post(url, json=payload, timeout=20)

def parse_offers(raw_offers, room_type):
    results = []
    for o in raw_offers:
        undergrounds = o.get("geo", {}).get("undergrounds") or []
        metro = undergrounds[0] if undergrounds else {}
        districts = o.get("geo", {}).get("districts") or []
        district_name = districts[0].get("name") if districts else None
        price = o.get("bargainTerms", {}).get("price") or 0
        area = o.get("totalArea")
        try:
            price_per_m2 = round(price / float(area)) if area else None
        except:
            price_per_m2 = None

        results.append({
            "id": o.get("id"),
            "url": o.get("fullUrl"),
            "price": price,
            "price_per_m2": price_per_m2,
            "currency": o.get("bargainTerms", {}).get("currency"),
            "rooms": o.get("roomsCount"),
            "area": area,
            "floor": o.get("floorNumber"),
            "floors_total": o.get("building", {}).get("floorsCount"),
            "address": o.get("geo", {}).get("userInput"),
            "metro": metro.get("name"),
            "metro_minutes": metro.get("time"),
            "metro_transport": metro.get("transportType"),
            "district": district_name,
            "description": o.get("formattedFullInfo"),
            "seller_type": o.get("user", {}).get("accountType"),
            "published": o.get("humanizedTimedelta"),
            "room_type_query": room_type,
        })
    return results

def load_progress():
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE) as f:
            return json.load(f)
    return {"done": [], "seen_ids": []}

def save_progress(progress):
    with open(PROGRESS_FILE, "w") as f:
        json.dump(progress, f)

def append_csv(rows):
    file_exists = os.path.exists(OUTPUT_FILE)
    with open(OUTPUT_FILE, "a", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        if not file_exists:
            writer.writeheader()
        writer.writerows(rows)

# ─── Главный цикл ─────────────────────────────────────────────
progress = load_progress()
seen_ids = set(progress.get("seen_ids", []))
session = make_session()
total_saved = 0

for room_type in ROOM_TYPES:
    for price_from, price_to in PRICE_RANGES:
        segment = f"{room_labels[room_type]} {price_from//1_000_000}-{price_to//1_000_000}М"
        print(f"\n{'='*55}")
        print(f"🏠 {segment}")
        print(f"{'='*55}")

        page = 1
        retries = 0

        while True:
            key = f"{room_type}:{price_from}:{price_to}:{page}"

            if key in progress["done"]:
                page += 1
                continue

            print(f"  📄 Страница {page}...", end=" ", flush=True)

            try:
                r = fetch_page(session, room_type, page, price_from, price_to)
            except Exception as e:
                print(f"⚠️ Ошибка: {e}")
                retries += 1
                if retries >= MAX_RETRIES:
                    print("❌ Пропускаем сегмент.")
                    break
                time.sleep(random.uniform(*DELAY_BLOCKED))
                session = make_session()
                continue

            if r.status_code in (403, 429) or "captcha" in r.text.lower():
                wait = random.uniform(*DELAY_BLOCKED)
                print(f"🚫 Блокировка! Пауза {wait:.0f}с...")
                retries += 1
                if retries >= MAX_RETRIES:
                    print("❌ Пропускаем сегмент.")
                    break
                time.sleep(wait)
                session = make_session()
                continue

            if r.status_code != 200:
                print(f"❌ Статус {r.status_code}")
                page += 1
                continue

            raw_offers = r.json().get("data", {}).get("offersSerialized", [])

            if not raw_offers:
                print("✅ Конец сегмента.")
                break

            offers = parse_offers(raw_offers, room_type)
            new_offers = [o for o in offers if o["id"] not in seen_ids]
            for o in new_offers:
                seen_ids.add(o["id"])

            append_csv(new_offers)
            total_saved += len(new_offers)
            progress["done"].append(key)
            progress["seen_ids"] = list(seen_ids)
            save_progress(progress)
            retries = 0

            print(f"+{len(new_offers)} | всего: {total_saved}")

            page += 1
            time.sleep(random.uniform(*DELAY_NORMAL))

print(f"\n✅ Готово! Сохранено {total_saved} объявлений в {OUTPUT_FILE}")
