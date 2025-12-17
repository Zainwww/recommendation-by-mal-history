from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.firefox.service import Service
from webdriver_manager.firefox import GeckoDriverManager
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By

import pandas as pd
import time
from collections import OrderedDict
from datetime import datetime
import os
import re


def scrape_data(username):
    start_time = time.time()

    options = webdriver.FirefoxOptions()
    options.add_argument("--headless")
    options.add_argument("--disable-blink-features=AutomationControlled")

    service = Service(GeckoDriverManager().install())
    driver = webdriver.Firefox(service=service, options=options)

    driver.get(f"https://myanimelist.net/animelist/{username}")

    try:
        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.CLASS_NAME, "list-item"))
        )
    except:
        driver.quit()
        return {"error": "Failed to load the page or username not found."}

    page = driver.page_source
    soup = BeautifulSoup(page, 'html.parser')

    list_items = soup.find_all('tbody', class_='list-item')

    data = []
    for j in list_items:
        data.append({
            'id': j.find('td', class_='image').find('a')['href'].split('/')[2],
            'image': j.find('td', class_='image').find('img')['src'].strip().replace('\n', ''),
            'title': j.find('td', class_='clearfix').find('a').text.strip().replace('\n', ''),
            'score': j.find('td', class_='score').find('a').text.strip().replace('\n', ''),
            'type': j.find('td', class_='type').text.strip().replace('\n', ''),
            'Progress': j.find('td', class_='progress').text.strip().replace('\n', '').replace(' ','')
        })

    df = pd.DataFrame(data)
    df['score'] = df['score'].replace('-', '0').astype(int)
    df = df.sort_values('score', ascending=False)


    driver.quit()

    elapsed_time = time.time() - start_time

    return OrderedDict({
        "username": username,
        "total_items": len(df),
        "elapsed_time_sec": round(elapsed_time, 2),
        "data": df.to_dict(orient='records')
    })

def main():
    while True:
        username = input("Username: ").strip()

        if username:
            scrape_data(username)
        
        print("Waiting for next input...")
        time.sleep(5)

if __name__ == "__main__":
    main()