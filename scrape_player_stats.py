"""
Script to scrape all NBA player statistics for MVP prediction model.
Scrapes per-game statistics from basketball-reference.com for years 1992-2024.
"""

import requests
import time
from bs4 import BeautifulSoup
import pandas as pd
from io import StringIO
import os

def scrape_player_stats(years):
    """
    Scrape player statistics for given years.
    
    Args:
        years: List of years to scrape
    """
    player_stats_url = "https://www.basketball-reference.com/leagues/NBA_{}_per_game.html"
    
    # Create player directory if it doesn't exist
    os.makedirs("player", exist_ok=True)
    
    for year in years:
        url = player_stats_url.format(year)
        print(f"Scraping {year}...")
        
        try:
            data = requests.get(url)
            data.raise_for_status()
            
            with open(f"player/{year}.html", "w+", encoding='utf-8') as file:
                file.write(data.text)
            
            # Be respectful with rate limiting
            time.sleep(1)
            
        except Exception as e:
            print(f"Error scraping {year}: {e}")
            continue
    
    print("Player stats scraping complete!")

if __name__ == "__main__":
    # Scrape 30 years of NBA MVP candidate statistics (1992-2022)
    years = list(range(1992, 2023))
    scrape_player_stats(years)
