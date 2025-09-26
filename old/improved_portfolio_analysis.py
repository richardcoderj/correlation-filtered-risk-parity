#!/usr/bin/env python3
"""
Improved Portfolio Analysis with Long Historical Data
Focus on assets with longest available history and proper risk calculations
"""

import pandas as pd
import numpy as np
import requests
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Set pandas display options
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 100)
pd.set_option('display.width', None)

class ImprovedPortfolioAnalyzer:
    def __init__(self, api_key):
        self.api_key = api_key
        self.base_url = "https://eodhd.com/api/eod"
        
        # Focus on assets with longest historical data
        self.assets = {
            'NEM': 'Newmont Corporation (Gold Mining)',
            'XOM': 'ExxonMobil (Energy/Oil)',
            'FGOVX': 'Fidelity Government Income Fund',
            'VUSTX': 'Vanguard Long-Term Treasury Fund',
            'VWO': 'Vanguard Emerging Markets ETF',
            'EEM': 'iShares MSCI Emerging Markets ETF'
        }
        
        print("Initialized Improved Portfolio Analyzer")
        print("Target assets with long historical data:")
        for symbol, name in self.assets.items():
            print(f"  {symbol}: {name}")
    
    def fetch_historical_data(self, symbol, start_date="1990-01-01", end_date="2024-12-31"):
        """Fetch historical data with extensive date range"""
        url = f"{self.base_url}/{symbol}.US"
        params = {
            'api_token': self.api_key,
            'from': start_date,
            'to': end_date,
            'period': 'd',
            'fmt': 'json'
        }
        
        try:
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            if not data:
                print(f"No data found for {symbol}")
                return None
                
            df = pd.DataFrame(data)
            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)
            df = df[['adjusted_close']].rename(columns={'adjusted_close': symbol})
            df[symbol] = pd.to_numeric(df[symbol], errors='coerce')
            
            print(f"{symbol}: {len(df)} data points from {df.index.min().strftime('%Y-%m-%d')} to {df.index.max().strftime('%Y-%m-%d')}")
            return df
            
        except Exception as e:
            print(f"Error fetching {symbol}: {str(e)}")
            return None
    
    def explore_data_availability(self):
        """Explore data availability for all target assets"""
        print("\n=== DATA AVAILABILITY EXPLORATION ===")
        print("Checking historical data availability for each asset...")
        
        all_data = {}
        data_info = []
        
        for symbol in self.assets.keys():
            print(f"\nFetching data for {symbol}...")
            data = self.fetch_historical_data(symbol)
            
            if data is not None:
                all_data[symbol] = data
                
                # Calculate basic statistics
                start_date = data.index.min()
                end_date = data.index.max()
                total_days = len(data)
                
                data_info.append({
                    'Symbol': symbol,
                    'Name': self.assets[symbol],
                    'Start_Date': start_date.strftime('%Y-%m-%d'),
                    'End_Date': end_date.strftime('%Y-%m-%d'),
                    'Total_Days': total_days,
                    'Years': round((end_date - start_date).days / 365.25, 1)
                })
        
        # Create summary table
        summary_df = pd.DataFrame(data_info)
        summary_df = summary_df.sort_values('Start_Date')
        
        print("\n=== DATA AVAILABILITY SUMMARY ===")
        print(summary_df.to_string(index=False))
        
        return all_data, summary_df
    
    def calculate_returns_and_risk_metrics(self, price_data):
        """Calculate returns and proper risk metrics"""
        print("\n=== CALCULATING RETURNS AND RISK METRICS ===")
        
        # Calculate daily returns
        returns = price_data.pct_change().dropna()
        
        # Calculate annualized metrics (252 trading days per year)
        annual_returns = returns.mean() * 252
        annual_volatility = returns.std() * np.sqrt(252)
        
        # Calculate Sharpe ratio (assuming 2% risk-free rate)
        risk_free_rate = 0.02
        sharpe_ratio = (annual_returns - risk_free_rate) / annual_volatility
        
        # Calculate correlation matrix
        correlation_matrix = returns.corr()
        
        # Create summary
        risk_metrics = pd.DataFrame({
            'Annual_Return': annual_returns,
            'Annual_Volatility': annual_volatility,
            'Sharpe_Ratio': sharpe_ratio
        }).round(4)
        
        print("Risk Metrics Summary:")
        print(risk_metrics.to_string())
        
        print(f"\nCorrelation Matrix:")
        print(correlation_matrix.round(3).to_string())
        
        return returns, risk_metrics, correlation_matrix
    
    def calculate_risk_parity_weights(self, returns, lookback_days=252):
        """Calculate risk parity weights based on inverse volatility"""
        print(f"\n=== CALCULATING RISK PARITY WEIGHTS ===")
        
        # Calculate rolling volatility
        rolling_vol = returns.rolling(window=lookback_days).std() * np.sqrt(252)
        
        # Calculate inverse volatility weights
        inv_vol = 1 / rolling_vol
        weights = inv_vol.div(inv_vol.sum(axis=1), axis=0)
        
        print("Sample Risk Parity Weights (last 10 observations):")
        print(weights.tail(10).round(4).to_string())
        
        return weights
    
    def analyze_correlation_regimes(self, returns, window=63):  # ~3 months
        """Analyze rolling correlation regimes"""
        print(f"\n=== ANALYZING CORRELATION REGIMES ===")
        
        rolling_corr = {}
        
        # Calculate pairwise rolling correlations
        symbols = returns.columns.tolist()
        for i, symbol1 in enumerate(symbols):
            for symbol2 in symbols[i+1:]:
                pair = f"{symbol1}-{symbol2}"
                rolling_corr[pair] = returns[symbol1].rolling(window=window).corr(returns[symbol2])
        
        # Calculate average correlation
        avg_correlation = pd.DataFrame(rolling_corr).mean(axis=1)
        
        print("Correlation regime analysis:")
        print(f"Average correlation over time - Mean: {avg_correlation.mean():.3f}, Std: {avg_correlation.std():.3f}")
        print(f"Max average correlation: {avg_correlation.max():.3f}")
        print(f"Min average correlation: {avg_correlation.min():.3f}")
        
        return rolling_corr, avg_correlation

def main():
    """Main analysis function"""
    print("="*60)
    print("IMPROVED PORTFOLIO ANALYSIS - LONG HISTORICAL DATA")
    print("="*60)
    
    # Initialize analyzer
    api_key = '647f18a6ead3f0.56528805'  # From env.txt
    analyzer = ImprovedPortfolioAnalyzer(api_key)
    
    # Step 1: Explore data availability
    all_data, data_summary = analyzer.explore_data_availability()
    
    if not all_data:
        print("No data available for analysis!")
        return
    
    # Step 2: Combine data and align dates
    print(f"\n=== COMBINING AND ALIGNING DATA ===")
    combined_data = pd.concat(list(all_data.values()), axis=1).dropna()
    print(f"Combined dataset: {len(combined_data)} observations from {combined_data.index.min().strftime('%Y-%m-%d')} to {combined_data.index.max().strftime('%Y-%m-%d')}")
    print(f"Assets included: {list(combined_data.columns)}")
    
    # Step 3: Calculate returns and risk metrics
    returns, risk_metrics, correlation_matrix = analyzer.calculate_returns_and_risk_metrics(combined_data)
    
    # Step 4: Calculate risk parity weights
    risk_parity_weights = analyzer.calculate_risk_parity_weights(returns)
    
    # Step 5: Analyze correlation regimes
    rolling_corr, avg_correlation = analyzer.analyze_correlation_regimes(returns)
    
    # Save initial results
    data_summary.to_csv('data_availability_summary.csv', index=False)
    risk_metrics.to_csv('risk_metrics_summary.csv')
    correlation_matrix.to_csv('correlation_matrix.csv')
    
    print(f"\n=== INITIAL ANALYSIS COMPLETE ===")
    print("Files saved:")
    print("- data_availability_summary.csv")
    print("- risk_metrics_summary.csv") 
    print("- correlation_matrix.csv")
    
    return {
        'data': combined_data,
        'returns': returns,
        'risk_metrics': risk_metrics,
        'correlation_matrix': correlation_matrix,
        'risk_parity_weights': risk_parity_weights,
        'rolling_correlations': rolling_corr,
        'avg_correlation': avg_correlation,
        'data_summary': data_summary
    }

if __name__ == "__main__":
    results = main()