#!/usr/bin/env python3
"""
Komplett portföljstrategi med risk parity viktning och korrelationsbaserad diversifiering
"""

import numpy as np
import pandas as pd
import yfinance as yf
import requests
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# API-nycklar
EODHD_API_KEY = '647f18a6ead3f0.56528805'
FRED_API_KEY = 'e98443f825cc47acc2bdbd439c15eea7'

class PortfolioStrategy:
    """
    Risk Parity portföljstrategi med månadsvis ombalansering och korrelationsbaserad diversifiering
    """
    
    def __init__(self, initial_capital=25000, transaction_cost=0.001):
        self.initial_capital = initial_capital
        self.transaction_cost = transaction_cost  # 0.1% per handel
        
        # Portföljkomponenter - ETF:er som täcker specifikationerna
        self.assets = {
            'QUAL': 'Quality Factor ETF',  # iShares MSCI USA Quality Factor ETF
            'VWO': 'Emerging Markets ETF',  # Vanguard FTSE Emerging Markets ETF
            'TLT': 'Government Bonds',      # iShares 20+ Year Treasury Bond ETF  
            'GLD': 'Gold ETF',              # SPDR Gold Shares
            'DBA': 'Commodities ETF'        # Invesco DB Agriculture Fund
        }
        
        self.tickers = list(self.assets.keys())
        self.data = None
        self.returns = None
        
    def fetch_data_eodhd(self, start_date='2000-01-01', end_date=None):
        """
        Hämta historisk data från EODHD API
        """
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')
            
        print("Hämtar historisk data från EODHD...")
        all_data = {}
        
        for ticker in self.tickers:
            try:
                url = f"https://eodhistoricaldata.com/api/eod/{ticker}.US"
                params = {
                    'api_token': EODHD_API_KEY,
                    'from': start_date,
                    'to': end_date,
                    'period': 'd',
                    'fmt': 'json'
                }
                
                response = requests.get(url, params=params)
                
                if response.status_code == 200:
                    data = response.json()
                    if data:  # Om data inte är tom
                        df = pd.DataFrame(data)
                        df['date'] = pd.to_datetime(df['date'])
                        df = df.set_index('date')
                        df = df[['adjusted_close']].rename(columns={'adjusted_close': ticker})
                        all_data[ticker] = df
                        print(f"✓ Hämtade data för {ticker}: {len(df)} observationer")
                    else:
                        print(f"✗ Inga data för {ticker}")
                else:
                    print(f"✗ Fel vid hämtning för {ticker}: {response.status_code}")
                    
            except Exception as e:
                print(f"✗ Fel för {ticker}: {e}")
        
        if all_data:
            # Kombinera all data
            self.data = pd.concat(all_data.values(), axis=1)
            self.data = self.data.dropna()  # Ta bort rader med NaN värden
            
            print(f"\nSammanlagt: {len(self.data)} observationer från {self.data.index[0]} till {self.data.index[-1]}")
            print(f"Tillgänliga tillgångar: {list(self.data.columns)}")
            
            # Beräkna avkastning
            self.returns = self.data.pct_change().dropna()
            
        else:
            print("Ingen data kunde hämtas från EODHD. Försöker med yfinance...")
            self.fetch_data_yfinance(start_date, end_date)
            
    def fetch_data_yfinance(self, start_date='2000-01-01', end_date=None):
        """
        Fallback: Hämta data med yfinance om EODHD misslyckas
        """
        print("Hämtar data med yfinance...")
        
        try:
            self.data = yf.download(self.tickers, start=start_date, end=end_date)['Adj Close']
            self.data = self.data.dropna()
            
            if len(self.data) > 0:
                print(f"✓ Hämtade data: {len(self.data)} observationer från {self.data.index[0]} till {self.data.index[-1]}")
                self.returns = self.data.pct_change().dropna()
            else:
                print("✗ Ingen data kunde hämtas")
                
        except Exception as e:
            print(f"✗ Fel med yfinance: {e}")
    
    def calculate_risk_parity_weights(self, returns_window, min_weight=0.05, max_weight=0.5):
        """
        Beräkna risk parity viktningar baserat på invers volatilitet
        """
        # Beräkna volatiliteter (annualiserat)
        volatilities = returns_window.std() * np.sqrt(252)
        
        # Invers volatilitet viktningar
        inv_vol_weights = 1 / volatilities
        normalized_weights = inv_vol_weights / inv_vol_weights.sum()
        
        # Begränsa viktningar
        normalized_weights = np.clip(normalized_weights, min_weight, max_weight)
        normalized_weights = normalized_weights / normalized_weights.sum()
        
        return normalized_weights
    
    def calculate_portfolio_correlation(self, returns_window):
        """
        Beräkna snittkorrelation i portföljen
        """
        corr_matrix = returns_window.corr()
        
        # Ta bort diagonal (korrelation med sig själv = 1)
        mask = np.triu(np.ones_like(corr_matrix), k=1).astype(bool)
        correlations = corr_matrix.where(mask).stack().dropna()
        
        return correlations.mean()
    
    def backtest_strategy(self, lookback_volatility=60, lookback_correlation=20, 
                         correlation_threshold=0.65, rebalance_freq='M'):
        """
        Backtesta strategin
        """
        if self.data is None or len(self.data) == 0:
            print("Ingen data tillgänglig för backtesting")
            return None
            
        print(f"\nBacktestar strategi med:")
        print(f"- Volatilitet lookback: {lookback_volatility} dagar")
        print(f"- Korrelation lookback: {lookback_correlation} dagar") 
        print(f"- Korrelationsgräns: {correlation_threshold}")
        print(f"- Ombalanseringsfrekvens: {rebalance_freq}")
        
        # Skapa månatliga rebalansering-datum
        rebalance_dates = pd.date_range(start=self.returns.index[max(lookback_volatility, lookback_correlation)], 
                                       end=self.returns.index[-1], freq='MS')
        
        # Initialize tracking variables
        portfolio_value = [self.initial_capital]
        portfolio_dates = [self.returns.index[max(lookback_volatility, lookback_correlation) - 1]]
        weights_history = []
        correlation_history = []
        cash_periods = []
        
        current_weights = None
        in_cash = False
        
        # Loop genom alla dagar
        for date in self.returns.index[max(lookback_volatility, lookback_correlation):]:
            
            # Check om vi ska ombalansera (månadsvis)
            should_rebalance = date in rebalance_dates or current_weights is None
            
            if should_rebalance:
                # Hämta historical data för beräkningar
                vol_window = self.returns.loc[:date].tail(lookback_volatility)
                corr_window = self.returns.loc[:date].tail(lookback_correlation)
                
                # Beräkna snittkorrelation
                avg_correlation = self.calculate_portfolio_correlation(corr_window)
                correlation_history.append((date, avg_correlation))
                
                # Beslut: risk parity eller kontanter?
                if avg_correlation > correlation_threshold:
                    # Gå till kontanter
                    current_weights = pd.Series(0.0, index=self.tickers)
                    in_cash = True
                    cash_periods.append(date)
                    print(f"{date.date()}: Korrelation {avg_correlation:.3f} > {correlation_threshold} → Kontanter")
                else:
                    # Risk parity viktning
                    current_weights = self.calculate_risk_parity_weights(vol_window)
                    in_cash = False
                    print(f"{date.date()}: Korrelation {avg_correlation:.3f} ≤ {correlation_threshold} → Risk Parity")
                
                weights_history.append((date, current_weights.copy(), in_cash))
            
            # Beräkna dagens portföljavkastning
            if in_cash:
                daily_return = 0.0  # Antag 0% avkastning på kontanter
            else:
                daily_returns = self.returns.loc[date]
                daily_return = (current_weights * daily_returns).sum()
            
            # Uppdatera portföljvärde
            new_value = portfolio_value[-1] * (1 + daily_return)
            
            # Lägg till transaktionskostnader vid ombalansering
            if should_rebalance and not in_cash:
                transaction_cost = new_value * self.transaction_cost * len(self.tickers)
                new_value -= transaction_cost
            
            portfolio_value.append(new_value)
            portfolio_dates.append(date)
        
        # Skapa resultat DataFrame
        results = pd.DataFrame({
            'Date': portfolio_dates,
            'Portfolio_Value': portfolio_value
        }).set_index('Date')
        
        self.backtest_results = results
        self.weights_history = weights_history
        self.correlation_history = correlation_history
        self.cash_periods = cash_periods
        
        return results
    
    def calculate_performance_metrics(self, results=None):
        """
        Beräkna prestandamått
        """
        if results is None:
            results = self.backtest_results
            
        # Portfolio returns
        portfolio_returns = results['Portfolio_Value'].pct_change().dropna()
        
        # Annualiserad avkastning
        total_return = (results['Portfolio_Value'].iloc[-1] / results['Portfolio_Value'].iloc[0]) - 1
        years = (results.index[-1] - results.index[0]).days / 365.25
        annual_return = (1 + total_return) ** (1/years) - 1
        
        # Volatilitet (annualiserat)
        annual_volatility = portfolio_returns.std() * np.sqrt(252)
        
        # Sharpe ratio (antag 2% riskfri ränta)
        risk_free_rate = 0.02
        sharpe_ratio = (annual_return - risk_free_rate) / annual_volatility
        
        # Maximum drawdown
        rolling_max = results['Portfolio_Value'].expanding().max()
        drawdown = (results['Portfolio_Value'] - rolling_max) / rolling_max
        max_drawdown = drawdown.min()
        
        # Calmar ratio
        calmar_ratio = annual_return / abs(max_drawdown) if max_drawdown != 0 else np.inf
        
        metrics = {
            'Total Return': f"{total_return:.2%}",
            'Annual Return': f"{annual_return:.2%}",
            'Annual Volatility': f"{annual_volatility:.2%}",
            'Sharpe Ratio': f"{sharpe_ratio:.2f}",
            'Maximum Drawdown': f"{max_drawdown:.2%}",
            'Calmar Ratio': f"{calmar_ratio:.2f}",
            'Start Date': results.index[0].strftime('%Y-%m-%d'),
            'End Date': results.index[-1].strftime('%Y-%m-%d'),
            'Duration (Years)': f"{years:.1f}"
        }
        
        return metrics, drawdown

if __name__ == "__main__":
    # Skapa och kör strategin
    strategy = PortfolioStrategy()
    
    # Hämta data
    strategy.fetch_data_eodhd('2000-01-01')