#!/usr/bin/env python3
"""
Data Downloader for Portfolio Strategy
Downloads and caches historical price data for all assets
"""

import pandas as pd
import numpy as np
import requests
import yfinance as yf
from datetime import datetime
import os
from typing import Dict, Optional
import warnings

warnings.filterwarnings('ignore')

class DataDownloader:
    def __init__(self):
        # API Keys
        self.eodhd_api_key = '647f18a6ead3f0.56528805'

        # Asset portfolio
        self.assets = {
            'VFITX': {'name': 'Vanguard Intermediate-Term Treasury Index Fund', 'category': 'Fixed Income', 'start_year': 1991},
            'NEM': {'name': 'Newmont Corporation', 'category': 'Gold', 'start_year': 1940},
            'XOM': {'name': 'ExxonMobil Corporation', 'category': 'Energy/Commodities', 'start_year': 1970},
            'FSUTX': {'name': 'Fidelity Short-Term Treasury Bond Index Fund', 'category': 'Short-Term Treasury', 'start_year': 2003},
            'SPHQ': {'name': 'Invesco S&P 500 Quality', 'category': 'Quality Factor', 'start_year': 2005}
        }

        # Cache settings
        self.cache_dir = 'data/cache'
        self.combined_cache_file = 'data/cache/combined_data.parquet'

        # Ensure cache directory exists
        os.makedirs(self.cache_dir, exist_ok=True)

    def fetch_data_eodhd(self, symbol: str, start_date: str = '1990-01-01') -> pd.DataFrame:
        """Fetch historical data from EODHD API"""
        url = f"https://eodhd.com/api/eod/{symbol}.US"
        params = {
            'api_token': self.eodhd_api_key,
            'from': start_date,
            'period': 'd',
            'fmt': 'json'
        }

        try:
            response = requests.get(url, params=params, timeout=30)
            if response.status_code == 200:
                data = response.json()
                if data:
                    df = pd.DataFrame(data)
                    df['date'] = pd.to_datetime(df['date'])
                    df.set_index('date', inplace=True)
                    df = df[['adjusted_close']].rename(columns={'adjusted_close': symbol})
                    return df
        except Exception as e:
            print(f"EODHD failed for {symbol}: {e}")

        return pd.DataFrame()

    def fetch_data_yfinance(self, symbol: str, start_date: str = '1990-01-01') -> pd.DataFrame:
        """Fallback: fetch data from yfinance"""
        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(start=start_date, end=datetime.now().strftime('%Y-%m-%d'))
            if not df.empty:
                # Use Adj Close for total returns (includes dividends and splits)
                df = df[['Adj Close']].rename(columns={'Adj Close': symbol})
                return df
        except Exception as e:
            print(f"yfinance failed for {symbol}: {e}")

        return pd.DataFrame()

    def load_cached_data(self, symbol: str) -> Optional[pd.DataFrame]:
        """Load cached data for a symbol if it exists and is recent"""
        cache_file = f"{self.cache_dir}/{symbol}.parquet"

        if os.path.exists(cache_file):
            try:
                df = pd.read_parquet(cache_file)
                if not df.empty:
                    # Check if data is recent (within last 7 days)
                    last_date = df.index.max()
                    days_old = (datetime.now() - last_date).days

                    if days_old <= 7:
                        print(f"Using cached data for {symbol} (last updated: {last_date.date()})")
                        return df
                    else:
                        print(f"Cached data for {symbol} is {days_old} days old, will refresh")
            except Exception as e:
                print(f"Error reading cached data for {symbol}: {e}")

        return None

    def save_cached_data(self, symbol: str, data: pd.DataFrame):
        """Save data to cache"""
        if not data.empty:
            cache_file = f"{self.cache_dir}/{symbol}.parquet"
            try:
                data.to_parquet(cache_file)
                print(f"Cached data saved for {symbol}")
            except Exception as e:
                print(f"Error saving cached data for {symbol}: {e}")

    def fetch_single_asset(self, symbol: str, info: Dict) -> pd.DataFrame:
        """Fetch data for a single asset with caching"""
        print(f"Processing {symbol} ({info['name']})...")

        # Try to load from cache first
        cached_data = self.load_cached_data(symbol)
        if cached_data is not None:
            return cached_data

        # Fetch fresh data
        start_date = f"{info['start_year']}-01-01"

        # Try EODHD first
        df = self.fetch_data_eodhd(symbol, start_date)

        # Fallback to yfinance
        if df.empty:
            df = self.fetch_data_yfinance(symbol, start_date)

        if not df.empty:
            print(f"Downloaded {symbol}: {len(df)} data points from {df.index.min().date()} to {df.index.max().date()}")
            # Save to cache
            self.save_cached_data(symbol, df)
            return df
        else:
            print(f"FAILED to fetch data for {symbol}")
            return pd.DataFrame()

    def download_all_data(self, force_refresh: bool = False) -> Dict[str, pd.DataFrame]:
        """Download historical data for all assets"""
        print("Downloading historical data for all assets...")
        print("=" * 50)

        if force_refresh:
            print("Force refresh enabled - ignoring cached data")

        all_data = {}

        for symbol, info in self.assets.items():
            if force_refresh:
                # Delete cached file to force refresh
                cache_file = f"{self.cache_dir}/{symbol}.parquet"
                if os.path.exists(cache_file):
                    os.remove(cache_file)
                    print(f"Removed cached data for {symbol}")

            df = self.fetch_single_asset(symbol, info)

            if not df.empty:
                all_data[symbol] = df

        return all_data

    def combine_and_save_data(self, data_dict: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Combine all asset data and save to cache"""
        if not data_dict:
            print("No data to combine")
            return pd.DataFrame()

        print("\nCombining all asset data...")

        # Find common date range
        start_dates = [df.index.min() for df in data_dict.values()]
        end_dates = [df.index.max() for df in data_dict.values()]

        common_start = max(start_dates)
        common_end = min(end_dates)

        print(f"Common data period: {common_start.date()} to {common_end.date()}")

        # Combine data
        combined_df = pd.DataFrame()
        for symbol, df in data_dict.items():
            df_filtered = df[(df.index >= common_start) & (df.index <= common_end)]
            if combined_df.empty:
                combined_df = df_filtered.copy()
            else:
                combined_df = combined_df.join(df_filtered, how='inner')

        # Fill missing values
        combined_df = combined_df.fillna(method='ffill').dropna()

        print(f"Final combined dataset: {len(combined_df)} rows, {len(combined_df.columns)} assets")

        # Save combined data to cache
        try:
            combined_df.to_parquet(self.combined_cache_file)
            print(f"Combined data saved to: {self.combined_cache_file}")
        except Exception as e:
            print(f"Error saving combined data: {e}")

        return combined_df

    def load_combined_data(self) -> Optional[pd.DataFrame]:
        """Load combined data from cache if available and recent"""
        if os.path.exists(self.combined_cache_file):
            try:
                df = pd.read_parquet(self.combined_cache_file)
                if not df.empty:
                    last_date = df.index.max()
                    days_old = (datetime.now() - last_date).days

                    if days_old <= 7:
                        print(f"Using cached combined data (last updated: {last_date.date()})")
                        return df
                    else:
                        print(f"Combined cache is {days_old} days old, will refresh")
            except Exception as e:
                print(f"Error reading combined cache: {e}")

        return None

    def get_data(self, force_refresh: bool = False) -> pd.DataFrame:
        """Main method to get all data (from cache or download)"""
        if not force_refresh:
            # Try to load combined data from cache
            cached_combined = self.load_combined_data()
            if cached_combined is not None:
                return cached_combined

        # Download fresh data
        all_data = self.download_all_data(force_refresh)

        # Combine and save
        combined_data = self.combine_and_save_data(all_data)

        return combined_data

def main():
    """Main function to download and cache data"""
    import argparse

    parser = argparse.ArgumentParser(description='Download and cache portfolio data')
    parser.add_argument('--force-refresh', action='store_true',
                       help='Force refresh all data (ignore cache)')
    parser.add_argument('--show-info', action='store_true',
                       help='Show asset information and exit')

    args = parser.parse_args()

    downloader = DataDownloader()

    if args.show_info:
        print("Asset Information:")
        print("=" * 50)
        for symbol, info in downloader.assets.items():
            print(f"{symbol}: {info['name']} ({info['category']}) - Since {info['start_year']}")
        return

    # Change to project root if running from src/
    if os.path.basename(os.getcwd()) == 'src':
        os.chdir('..')

    print("Portfolio Data Downloader")
    print("=" * 50)

    try:
        data = downloader.get_data(force_refresh=args.force_refresh)

        if not data.empty:
            print(f"\nData summary:")
            print(f"Period: {data.index.min().date()} to {data.index.max().date()}")
            print(f"Total days: {len(data)}")
            print(f"Assets: {list(data.columns)}")
            print(f"Missing values: {data.isnull().sum().sum()}")
            print("\nData download completed successfully!")
        else:
            print("ERROR: No data was downloaded")

    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()