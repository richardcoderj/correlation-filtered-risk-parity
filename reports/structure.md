# Correlation-Filtered Risk Parity Project Structure Analysis

## Project Overview
Detta projekt implementerar en korrelationsfiltrerad risk-paritet portföljstrategi med flera iterationer och försök. Den nuvarande och korrekta strategin är **corrected_portfolio_analysis** som innehåller assets FGOVX, NEM, XOM, VEIEX, SPHQ.

## Current/Final Strategy (Korrekt Implementation)

### Huvudfiler för den korrekta strategin:
- `corrected_portfolio_strategy.py` - Den slutgiltiga och korrekta strategiimplementationen
- `corrected_strategy_report.md` - Rapport för den korrekta strategin
- `corrected_portfolio_analysis.png` - Visualiseringar för den korrekta strategin
- `corrected_parameter_results.csv` - Parameterresultat för den korrekta strategin
- `corrected_portfolio_weights.csv` - Vikter för den korrekta strategin (24.8KB)
- `corrected_portfolio_returns.csv` - Avkastning för den korrekta strategin

### Korrekt Asset Universe (5 Assets - En per kategori):
- **FGOVX** (Fixed Income): Fidelity Government Income Fund - Slutvikt: 9.4%
- **NEM** (Gold): Newmont Corporation - Slutvikt: 29.7%
- **XOM** (Energy/Commodities): ExxonMobil Corporation - Slutvikt: 2.9%
- **VEIEX** (Emerging Markets): Vanguard Emerging Markets - Slutvikt: 35.2%
- **SPHQ** (Quality Factor): Invesco S&P 500 Quality - Slutvikt: 22.8%

### Strategi Specificationer:
- **Metod**: Return-Weighted Risk Parity with Correlation Management
- **Ombalansering**: Monthly (månatlig)
- **Analysperiod**: 2005-12-06 to 2025-09-25 (237 månader)
- **Optimal Parameters**:
  - Lookback Period: 12 månader
  - Correlation Threshold: 0.6
- **Performance**:
  - Total Return: 448.3%
  - Sharpe Ratio: 0.799
  - Max Drawdown: -15.3%

## Previous Attempts/Iterations (Tidigare Försök)

### 1. Original Portfolio Strategy (`portfolio_strategy.py`)
- **Assets**: QUAL, VWO, TLT, GLD, DBA (5 ETF:er)
- **Språk**: Svenska kommentarer
- **Strategi**: Risk Parity med korrelationsbaserad diversifiering
- **Kapital**: 25,000 SEK
- **Fokus**: Månadsvis ombalansering med cash allocation vid hög korrelation

### 2. Comprehensive Portfolio Strategy (`comprehensive_portfolio_strategy.py`)
- **Assets**: NEM, XOM, FGOVX, VUSTX, VWO, EEM (6 assets)
- **Fokus**: Regime detection och cash management
- **Tidsperiod**: Längre historisk analys
- **Metod**: Risk parity med korrelationsfilter och cash allocation

### 3. Extended Portfolio Strategy (`extended_portfolio_strategy.py`)
- **Assets**: 9 assets (NEM, XOM, FCX, FGOVX, VUSTX, VEIEX, TLT, EEM, GLD)
- **Prioritering**: Längsta historiska tillgänglighet
- **Kategorier**: Gold/Mining, Energy, Government Bonds, Treasury Bonds, Emerging Markets
- **Period**: Olika tillgängliga perioder beroende på asset (1940s-2004)

### 4. Focused Portfolio Analysis (`focused_portfolio_analysis.py`)
- **Assets**: 8 assets (NEM, XOM, FGOVX, VUSTX, VEIEX, TLT, EEM, GLD)
- **Fokus**: Snabbhet och nyckelfynd
- **Parameters**: Fokuserad parametertestning (0.5, 0.7 korrelation; 120, 252 lookback)
- **Regimer**: pre_2008, crisis_2008, post_crisis, covid_era

### 5. Complete Portfolio Analysis (`complete_portfolio_analysis.py`)
- **Assets**: 6 assets (NEM, XOM, FGOVX, VUSTX, VWO, EEM)
- **Fokus**: Integrerad data + strategi
- **Marknadsevents**: Dot-com crash, 2008 crisis, COVID-19

### 6. Improved Portfolio Analysis (`improved_portfolio_analysis.py`)
- **Assets**: 6 assets (NEM, XOM, FGOVX, VUSTX, VWO, EEM)
- **Fokus**: Längsta historiska data och korrekta riskberäkningar
- **Start**: 1990-01-01

## Support Files (Stödfiler)

### Data och Resultat:
- `data_availability_summary.csv` - Sammanfattning av datatillgänglighet
- `final_data_summary.csv` - Slutlig datasammanfattning
- `correlation_matrix.csv` - Korrelationsmatris
- `risk_metrics_summary.csv` - Riskmetriker sammanfattning
- `combined_portfolio_metrics.csv` - Kombinerade portföljmetriker
- `long_historical_metrics.csv` - Långa historiska metriker

### Visualiseringar:
- `allocation_chart.html` och `.png` - Allokeringsdiagram
- `complete_portfolio_analysis.html` och `.png` - Komplett portfolioanalys
- `focused_portfolio_analysis.html` och `.png` - Fokuserad analys
- `portfolio_performance.html` och `.png` - Portföljprestation
- `portfolio_drawdown.html` och `.png` - Drawdown analys
- `correlation_analysis.html` och `.png` - Korrelationsanalys
- `parameter_heatmaps.html` och `.png` - Parametrar heatmaps

### Rapporter:
- `complete_portfolio_report.md` - Komplett rapport
- `focused_portfolio_report.md` - Fokuserad rapport
- `strategy_report.md` - Strategirapport

## Key Differences Between Attempts

### Asset Selection Evolution:
1. **portfolio_strategy.py**: ETF-fokuserad (QUAL, VWO, TLT, GLD, DBA)
2. **Extended versions**: Bredare urval med historisk prioritering
3. **Corrected version**: Optimerat till 5 assets, en per kategori för maximal diversifiering

### Strategy Complexity:
1. **Basic**: Enkel risk parity med korrelationsfilter
2. **Comprehensive**: Regime detection och cash management
3. **Corrected**: Return-weighted risk parity med correlation management

### Time Periods:
1. **Earlier attempts**: Varierade start-datum beroende på asset availability
2. **Corrected**: Gemensam period 2005-2025 för alla assets

### Implementation Language:
1. **portfolio_strategy.py**: Svenska kommentarer
2. **Övriga**: Engelska implementation

## Configuration and Settings
- `.claude/settings.local.json` - Claude Code inställningar
- `Uploads/` directory - Uppladdade filer

## Recommendations for Future Restructuring

### Keep (Behåll):
- `corrected_portfolio_strategy.py` och relaterade filer
- Slutliga resultat och rapporter för den korrekta strategin
- Viktiga visualiseringar

### Archive (Arkivera):
- Alla tidigare strategiförsök i en `archive/` eller `previous_attempts/` mapp
- Gamla parameterresultat och visualiseringar
- Experimentella implementationer

### Organize by Category:
```
/src/ - Huvudimplementationer
/data/ - CSV-filer och dataresultat
/reports/ - Markdown rapporter
/visualizations/ - HTML och PNG filer
/archive/ - Tidigare försök
```

## Technical Notes
- API Key används: EODHD (647f18a6ead3f0.56528805)
- Backup data source: yfinance
- Risk-free rate: 2% (används i Sharpe ratio beräkningar)
- Transaction costs: 0.1-0.2% (rekommendation för live implementation)