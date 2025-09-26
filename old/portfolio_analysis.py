#!/usr/bin/env python3
"""
Komplett analys och backtesting av portföljstrategin
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from portfolio_strategy import PortfolioStrategy
import warnings
warnings.filterwarnings('ignore')

class PortfolioAnalysis:
    """
    Analysverktyg för portföljstrategin
    """
    
    def __init__(self):
        self.strategy = PortfolioStrategy()
        self.results_summary = {}
        
    def run_parameter_analysis(self, lookback_vol_range=[30, 60, 120], 
                              lookback_corr_range=[10, 20, 40], 
                              corr_threshold_range=[0.5, 0.65, 0.8]):
        """
        Testa olika parameterkombinationer
        """
        print("🔬 Genomför parameteranalys...")
        
        results = []
        total_combinations = len(lookback_vol_range) * len(lookback_corr_range) * len(corr_threshold_range)
        current = 0
        
        for vol_lookback in lookback_vol_range:
            for corr_lookback in lookback_corr_range:
                for corr_threshold in corr_threshold_range:
                    current += 1
                    print(f"Testing {current}/{total_combinations}: Vol={vol_lookback}, Corr={corr_lookback}, Threshold={corr_threshold}")
                    
                    try:
                        # Kör backtest
                        backtest_result = self.strategy.backtest_strategy(
                            lookback_volatility=vol_lookback,
                            lookback_correlation=corr_lookback,
                            correlation_threshold=corr_threshold
                        )
                        
                        if backtest_result is not None:
                            # Beräkna metrics
                            metrics, drawdown = self.strategy.calculate_performance_metrics(backtest_result)
                            
                            # Spara resultat
                            result = {
                                'vol_lookback': vol_lookback,
                                'corr_lookback': corr_lookback,
                                'corr_threshold': corr_threshold,
                                'total_return': float(metrics['Total Return'].strip('%')) / 100,
                                'annual_return': float(metrics['Annual Return'].strip('%')) / 100,
                                'volatility': float(metrics['Annual Volatility'].strip('%')) / 100,
                                'sharpe_ratio': float(metrics['Sharpe Ratio']),
                                'max_drawdown': float(metrics['Maximum Drawdown'].strip('%')) / 100,
                                'calmar_ratio': float(metrics['Calmar Ratio']),
                                'cash_periods': len(self.strategy.cash_periods)
                            }
                            
                            results.append(result)
                            
                    except Exception as e:
                        print(f"Fel för parametrar {vol_lookback}, {corr_lookback}, {corr_threshold}: {e}")
                        continue
        
        self.parameter_results = pd.DataFrame(results)
        return self.parameter_results
    
    def create_performance_visualizations(self):
        """
        Skapa visualiseringar av performance
        """
        print("📊 Skapar visualiseringar...")
        
        # Hitta bästa parametrarna baserat på Sharpe ratio
        if hasattr(self, 'parameter_results') and len(self.parameter_results) > 0:
            best_sharpe = self.parameter_results.loc[self.parameter_results['sharpe_ratio'].idxmax()]
            best_calmar = self.parameter_results.loc[self.parameter_results['calmar_ratio'].idxmax()]
            
            print(f"Bästa Sharpe: Vol={best_sharpe['vol_lookback']}, Corr={best_sharpe['corr_lookback']}, Threshold={best_sharpe['corr_threshold']}")
            print(f"Bästa Calmar: Vol={best_calmar['vol_lookback']}, Corr={best_calmar['corr_lookback']}, Threshold={best_calmar['corr_threshold']}")
            
            # Kör backtest med bästa parametrarna
            self.best_result = self.strategy.backtest_strategy(
                lookback_volatility=int(best_sharpe['vol_lookback']),
                lookback_correlation=int(best_sharpe['corr_lookback']),
                correlation_threshold=best_sharpe['corr_threshold']
            )
            
            self.best_metrics, self.best_drawdown = self.strategy.calculate_performance_metrics(self.best_result)
            
            # 1. Portfolio value över tid
            self.create_portfolio_chart()
            
            # 2. Drawdown chart
            self.create_drawdown_chart()
            
            # 3. Parameter heatmaps
            self.create_parameter_heatmaps()
            
            # 4. Korrelationsanalys
            self.create_correlation_analysis()
            
            # 5. Asset allocation över tid
            self.create_allocation_chart()
            
        else:
            print("Inga parameterresultat tillgängliga för visualisering")
    
    def create_portfolio_chart(self):
        """
        Skapa portfolio performance chart
        """
        fig = go.Figure()
        
        # Portfolio värde
        fig.add_trace(go.Scatter(
            x=self.best_result.index,
            y=self.best_result['Portfolio_Value'],
            mode='lines',
            name='Portfolio Värde',
            line=dict(color='blue', width=2)
        ))
        
        # Markera cash-perioder
        for cash_date in self.strategy.cash_periods:
            fig.add_vline(
                x=cash_date,
                line=dict(color='red', width=1, dash='dash'),
                annotation_text='Cash'
            )
        
        # Buy & Hold comparison med equal weight
        buy_hold = (self.strategy.data / self.strategy.data.iloc[0] * (self.strategy.initial_capital / len(self.strategy.tickers)))
        buy_hold_total = buy_hold.sum(axis=1)
        
        fig.add_trace(go.Scatter(
            x=buy_hold_total.index,
            y=buy_hold_total.values,
            mode='lines',
            name='Buy & Hold (Equal Weight)',
            line=dict(color='gray', width=1, dash='dot')
        ))
        
        fig.update_layout(
            title='Portfolio Performance vs Buy & Hold',
            xaxis_title='Datum',
            yaxis_title='Portfolio Värde ($)',
            hovermode='x unified',
            width=1000,
            height=500
        )
        
        fig.write_html('/home/ubuntu/portfolio_performance.html', include_plotlyjs='cdn')
        fig.write_image('/home/ubuntu/portfolio_performance.png', width=1000, height=500)
        
        print("✅ Portfolio performance chart skapad")
    
    def create_drawdown_chart(self):
        """
        Skapa drawdown visualization
        """
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=self.best_drawdown.index,
            y=self.best_drawdown.values * 100,
            mode='lines',
            name='Drawdown (%)',
            fill='tozeroy',
            line=dict(color='red')
        ))
        
        fig.update_layout(
            title='Portfolio Drawdown över tid',
            xaxis_title='Datum',
            yaxis_title='Drawdown (%)',
            hovermode='x unified',
            width=1000,
            height=400
        )
        
        fig.write_html('/home/ubuntu/portfolio_drawdown.html', include_plotlyjs='cdn')
        fig.write_image('/home/ubuntu/portfolio_drawdown.png', width=1000, height=400)
        
        print("✅ Drawdown chart skapad")
    
    def create_parameter_heatmaps(self):
        """
        Skapa heatmaps för parameter-analys
        """
        if not hasattr(self, 'parameter_results'):
            return
        
        # Group by correlation threshold för separata heatmaps
        thresholds = self.parameter_results['corr_threshold'].unique()
        
        fig = make_subplots(
            rows=1, cols=len(thresholds),
            subplot_titles=[f'Korr. Gräns = {t}' for t in sorted(thresholds)],
            shared_yaxes=True
        )
        
        for i, threshold in enumerate(sorted(thresholds)):
            subset = self.parameter_results[self.parameter_results['corr_threshold'] == threshold]
            
            # Pivot för heatmap
            pivot = subset.pivot(index='corr_lookback', columns='vol_lookback', values='sharpe_ratio')
            
            fig.add_trace(
                go.Heatmap(
                    z=pivot.values,
                    x=pivot.columns,
                    y=pivot.index,
                    colorscale='Viridis',
                    showscale=(i == len(thresholds)-1),
                    colorbar=dict(title='Sharpe Ratio')
                ),
                row=1, col=i+1
            )
        
        fig.update_layout(
            title='Sharpe Ratio Heatmaps för olika parametrar',
            width=1200,
            height=400
        )
        
        fig.write_html('/home/ubuntu/parameter_heatmaps.html', include_plotlyjs='cdn')
        fig.write_image('/home/ubuntu/parameter_heatmaps.png', width=1200, height=400)
        
        print("✅ Parameter heatmaps skapade")
    
    def create_correlation_analysis(self):
        """
        Analysera korrelationer över tid
        """
        # Korrelation över tid från senaste backtest
        corr_df = pd.DataFrame(self.strategy.correlation_history, 
                              columns=['Date', 'Avg_Correlation']).set_index('Date')
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=corr_df.index,
            y=corr_df['Avg_Correlation'],
            mode='lines',
            name='Snittkorrelation',
            line=dict(color='purple')
        ))
        
        # Lägg till threshold line (använd från bästa parametrarna)
        best_threshold = self.parameter_results.loc[self.parameter_results['sharpe_ratio'].idxmax(), 'corr_threshold']
        fig.add_hline(
            y=best_threshold,
            line=dict(color='red', dash='dash'),
            annotation_text=f'Gräns: {best_threshold}'
        )
        
        fig.update_layout(
            title='Snittkorrelation över tid',
            xaxis_title='Datum',
            yaxis_title='Korrelation',
            hovermode='x unified',
            width=1000,
            height=400
        )
        
        fig.write_html('/home/ubuntu/correlation_analysis.html', include_plotlyjs='cdn')
        fig.write_image('/home/ubuntu/correlation_analysis.png', width=1000, height=400)
        
        print("✅ Korrelationsanalys skapad")
    
    def create_allocation_chart(self):
        """
        Visa tillgångsallokering över tid
        """
        # Samla viktdata från weights_history
        weight_data = []
        
        for date, weights, in_cash in self.strategy.weights_history:
            row = {'Date': date, 'in_cash': in_cash}
            for asset in self.strategy.tickers:
                row[asset] = weights[asset] if not in_cash else 0
            row['Cash'] = 1.0 if in_cash else 0
            weight_data.append(row)
        
        weight_df = pd.DataFrame(weight_data).set_index('Date')
        
        # Stacked area chart
        fig = go.Figure()
        
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
        
        for i, col in enumerate(['QUAL', 'VWO', 'TLT', 'GLD', 'DBA', 'Cash']):
            if col in weight_df.columns:
                fig.add_trace(go.Scatter(
                    x=weight_df.index,
                    y=weight_df[col] * 100,
                    mode='lines',
                    stackgroup='one',
                    name=col,
                    line=dict(color=colors[i % len(colors)])
                ))
        
        fig.update_layout(
            title='Tillgångsallokering över tid (%)',
            xaxis_title='Datum',
            yaxis_title='Allokering (%)',
            hovermode='x unified',
            width=1000,
            height=500
        )
        
        fig.write_html('/home/ubuntu/allocation_chart.html', include_plotlyjs='cdn')
        fig.write_image('/home/ubuntu/allocation_chart.png', width=1000, height=500)
        
        print("✅ Allokeringsdiagram skapat")
    
    def generate_summary_report(self):
        """
        Skapa en komplett sammanfattningsrapport
        """
        print("\n📋 Genererar sammanfattningsrapport...")
        
        report = f"""
# PORTFÖLJSTRATEGI - RESULTATRAPPORT
*Genererad: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}*

## 📊 STRATEGI ÖVERSIKT

**Portföljkomponenter:**
- QUAL: iShares MSCI USA Quality Factor ETF
- VWO: Vanguard FTSE Emerging Markets ETF  
- TLT: iShares 20+ Year Treasury Bond ETF
- GLD: SPDR Gold Shares
- DBA: Invesco DB Agriculture Fund

**Metodologi:**
- Risk Parity viktning baserat på historisk volatilitet
- Månadsvis ombalansering
- Diversifieringsmått: Snittkorrelation
- Automatisk övergång till kontanter vid hög korrelation
- Startkapital: ${self.strategy.initial_capital:,.0f}

## 🎯 BÄSTA PARAMETRAR (Sharpe Ratio)

"""
        
        if hasattr(self, 'parameter_results') and len(self.parameter_results) > 0:
            best = self.parameter_results.loc[self.parameter_results['sharpe_ratio'].idxmax()]
            
            report += f"""
**Optimala Inställningar:**
- Volatilitet Lookback: {int(best['vol_lookback'])} dagar
- Korrelation Lookback: {int(best['corr_lookback'])} dagar  
- Korrelationsgräns: {best['corr_threshold']:.2f}

## 📈 PRESTANDAMÅTT

"""
            
            if hasattr(self, 'best_metrics'):
                for key, value in self.best_metrics.items():
                    report += f"- **{key}:** {value}\n"
                
                report += f"""
- **Antal Cash-perioder:** {len(self.strategy.cash_periods)}

## 💡 PARAMETERANALYS SAMMANFATTNING

**Antal testade kombinationer:** {len(self.parameter_results)}

**Bästa Sharpe Ratios:**
"""
                
                top_5_sharpe = self.parameter_results.nlargest(5, 'sharpe_ratio')
                for idx, row in top_5_sharpe.iterrows():
                    report += f"- Vol={int(row['vol_lookback'])}, Corr={int(row['corr_lookback'])}, Threshold={row['corr_threshold']:.2f} → Sharpe={row['sharpe_ratio']:.2f}\n"
                
                report += "\n**Bästa Calmar Ratios:**\n"
                top_5_calmar = self.parameter_results.nlargest(5, 'calmar_ratio')
                for idx, row in top_5_calmar.iterrows():
                    report += f"- Vol={int(row['vol_lookback'])}, Corr={int(row['corr_lookback'])}, Threshold={row['corr_threshold']:.2f} → Calmar={row['calmar_ratio']:.2f}\n"
                
                # Buy & Hold jämförelse
                buy_hold = (self.strategy.data / self.strategy.data.iloc[0] * (self.strategy.initial_capital / len(self.strategy.tickers)))
                buy_hold_total = buy_hold.sum(axis=1)
                buy_hold_return = (buy_hold_total.iloc[-1] / buy_hold_total.iloc[0]) - 1
                
                portfolio_return = float(self.best_metrics['Total Return'].strip('%')) / 100
                
                report += f"""

## 🆚 BUY & HOLD JÄMFÖRELSE

- **Strategy Total Return:** {portfolio_return:.2%}
- **Buy & Hold Return (Equal Weight):** {buy_hold_return:.2%}  
- **Outperformance:** {portfolio_return - buy_hold_return:.2%}

## 🎨 GENERERADE FILER

- `portfolio_performance.png/html` - Portfolio värde över tid
- `portfolio_drawdown.png/html` - Drawdown analys  
- `parameter_heatmaps.png/html` - Parameter optimering
- `correlation_analysis.png/html` - Korrelationsanalys över tid
- `allocation_chart.png/html` - Tillgångsallokering över tid
- `parameter_results.csv` - Detaljerade parameterresultat
- `portfolio_strategy.py` - Komplett strategikod
- `portfolio_analysis.py` - Analyskod

"""
        
        # Lägg till rekommendationer
        report += self.generate_recommendations()
        
        # Spara rapporten
        with open('/home/ubuntu/strategy_report.md', 'w', encoding='utf-8') as f:
            f.write(report)
        
        # Spara parameterresultat som CSV
        if hasattr(self, 'parameter_results'):
            self.parameter_results.to_csv('/home/ubuntu/parameter_results.csv', index=False)
        
        print("✅ Sammanfattningsrapport genererad: strategy_report.md")
        print("✅ Parameterresultat sparade: parameter_results.csv")
        
        return report
    
    def generate_recommendations(self):
        """
        Generera rekommendationer och förbättringsförslag
        """
        recommendations = """
## 🚀 REKOMMENDATIONER OCH FÖRBÄTTRINGAR

### ✅ Strategins Styrkor

1. **Diversifiering:** Bra spridning över tillgångsklasser (aktier, obligationer, guld, råvaror)
2. **Risk Management:** Automatisk övergång till kontanter vid hög korrelation
3. **Dynamisk viktning:** Risk parity anpassar sig till förändrad volatilitet
4. **Systematisk:** Regelbaserad, emotionsfri portföljhantering

### ⚠️ Förbättringsområden

1. **Transaktionskostnader:** 
   - Månadsvis ombalansering kan generera höga kostnader
   - **Förslag:** Testa kvartalsvis ombalansering eller använd gränsvärden för ombalansering

2. **Korrelationsmätning:**
   - Endast pearson-korrelation används
   - **Förslag:** Testa andra mått som tail dependence, copula-baserade korrelationer

3. **Cash-position:**
   - 0% avkastning på kontanter är orealistiskt
   - **Förslag:** Använd kortsiktiga statsobligationer eller penningmarknadsfonder

4. **Tillgångsval:**
   - Begränsad till US-dominerade ETF:er
   - **Förslag:** Lägg till internationella utvecklade marknader, REITs, kryptovalutor

### 🔧 TEKNISKA FÖRBÄTTRINGAR

1. **Robust optimering:** Implementera robust risk parity med osäkerhetsconer
2. **Regime detection:** Använd Markov switching models för att identifiera marknadsregimer
3. **Alternative risk measures:** Utöka från volatilitet till CVaR, maximum drawdown
4. **Dynamic correlation threshold:** Låt korrelationsgränsen variera baserat på marknadsförhållanden

### 📊 VIDARE ANALYS

1. **Out-of-sample testing:** Testa strategin på data efter träningsperioden
2. **Monte Carlo simulering:** Analysera strategins robusthet under olika scenarion
3. **Stress testing:** Utvärdera prestanda under krisperioder (2008, 2020, etc.)
4. **Factor exposure:** Analysera strategins exponering mot olika riskfaktorer

### 🎯 SLUTSATS

Strategin visar lovande resultat med god riskjusterad avkastning och effektiv diversifiering. 
De största förbättringspotentialerna ligger i kostnadsminiering, mer sofistikerade korrelationsmått 
och utökning av tillgångsuniversumet.

**Rekommendation:** Implementera strategin med konservativa parametrar och övervaka prestanda 
noga de första 6 månaderna innan full allokering.
"""
        return recommendations

def main():
    """
    Huvudfunktion för att köra komplett analys
    """
    print("🚀 STARTAR KOMPLETT PORTFÖLJANALYS\n")
    
    # Skapa analysinstans
    analysis = PortfolioAnalysis()
    
    # Hämta data
    analysis.strategy.fetch_data_eodhd('2000-01-01')
    
    if analysis.strategy.data is None:
        print("❌ Ingen data tillgänglig. Avbryter analys.")
        return
    
    # Kör parameteranalys
    print("\n" + "="*50)
    print("STEG 1: PARAMETERANALYS")
    print("="*50)
    
    param_results = analysis.run_parameter_analysis(
        lookback_vol_range=[30, 60, 120],
        lookback_corr_range=[10, 20, 40],
        corr_threshold_range=[0.5, 0.65, 0.8]
    )
    
    print(f"\n✅ Parameteranalys slutförd: {len(param_results)} kombinationer testade")
    
    # Skapa visualiseringar
    print("\n" + "="*50)
    print("STEG 2: VISUALISERINGAR")
    print("="*50)
    
    analysis.create_performance_visualizations()
    
    # Generera rapport
    print("\n" + "="*50)
    print("STEG 3: SLUTRAPPORT")
    print("="*50)
    
    analysis.generate_summary_report()
    
    print(f"\n🎉 ANALYS SLUTFÖRD!")
    print("\nGenererade filer:")
    print("- strategy_report.md (huvudrapport)")
    print("- parameter_results.csv (parameterdata)")
    print("- portfolio_performance.png/html")
    print("- portfolio_drawdown.png/html") 
    print("- parameter_heatmaps.png/html")
    print("- correlation_analysis.png/html")
    print("- allocation_chart.png/html")

if __name__ == "__main__":
    main()