import pandas as pd
import numpy as np
import time
from app.services.scraper_service import scrape_data
from app.utils.text_extraction import extract_tfidf_keywords, extract_bert_keywords, extract_yake_keywords, extract_rake_keywords

def analysis_anime():
    return 

def main():
    while True:
        username = input("Username: ").strip()

        if username:
            analysis_anime(username)
        
        print("Waiting for next input...")
        time.sleep(5)

if __name__ == "__main__":
    main()