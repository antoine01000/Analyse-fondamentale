# -*- coding: utf-8 -*-
"""Application Analyse fondamentale

Script refactoré pour :
- Regrouper logiques dans `main()`
- Supprimer les exécutions au top-level
- Ajouter historisation du DataFrame complet `df`
- Gestion des imports en début de fichier
"""

import os
import sys
import datetime
import time
import warnings

import yfinance as yf
import pandas as pd
import numpy as np
import requests
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score


def calculate_performance_metrics(symbol: str, years: int) -> tuple[float, float, float]:
    """
    Calcule la performance totale, annualisée et le R² pour un ticker donné.
    """
    today = datetime.date.today()
    start = today.replace(year=today.year - years)

    data = yf.download(
        symbol,
        start=start.isoformat(),
        end=(today + datetime.timedelta(days=1)).isoformat(),
        progress=False,
        actions=True,
        auto_adjust=False
    )
    if data.empty or len(data) < 2:
        warnings.warn(f"Données insuffisantes pour {symbol} sur {years} ans.")
        return np.nan, np.nan, np.nan

    p0, p1 = data['Adj Close'].iloc[[0, -1]]
    total_div = data['Dividends'].sum()
    total_factor = (p1 - p0 + total_div) / p0 + 1

    # Annualisation géométrique
    years_real = (data.index[-1] - data.index[0]).days / 365.25
    ann_perf = (total_factor ** (1 / years_real) - 1) * 100 if total_factor > 0 and years_real > 0 else np.nan

    # R²
    X = np.arange(len(data)).reshape(-1, 1)
    y = data['Adj Close'].values.reshape(-1, 1)
    if np.all(y == y[0]):
        r2 = np.nan
    else:
        model = LinearRegression().fit(X, y)
        r2 = r2_score(y, model.predict(X))

    return round((total_factor - 1) * 100, 2), round(ann_perf, 2), round(r2, 4)


def fetch_finnhub_metrics(tickers: list[str], api_key: str) -> pd.DataFrame:
    """
    Récupère les metrics Finnhub pour chaque ticker.
    """
    headers = {'X-Finnhub-Token': api_key}
    rows = []
    for symbol in tickers:
        resp = requests.get(
            'https://finnhub.io/api/v1/stock/metric',
            headers=headers,
            params={'symbol': symbol, 'metric': 'all'}
        )
        data = resp.json().get('metric', {}) if resp.ok else {}
        rows.append({'ticker': symbol,
                     'Revenue_Growth_5Y': data.get('revenueGrowth5Y'),
                     'Revenue_Growth_LastYear_%': data.get('revenueGrowthTTMYoy'),
                     'FreeCashFlow5Y': data.get('focfCagr5Y'),
                     'EPS_Growth_5Y': data.get('epsGrowth5Y'),
                     'EPS_Growth_3Y': data.get('epsGrowth3Y'),
                     'ROIC_5Y': data.get('roi5Y'),
                     'ROI_ANNUAL': data.get('roiAnnual'),
                     'Gross_Margin_5Y': data.get('grossMargin5Y'),
                     'Gross_Margin_Annual': data.get('grossMarginAnnual')})
        time.sleep(0.5)
    df = pd.DataFrame(rows)
    for col in df.columns.drop('ticker'):
        df[col] = df[col].round(2)
    return df


def compute_sbc_and_debt(tickers: list[str]) -> pd.DataFrame:
    """
    Calcule SBC/FCF et netDebt/EBITDA pour chaque ticker.
    """
    sbc_rows = []
    debt_ratios = []

    for symbol in tickers:
        t = yf.Ticker(symbol)
        cf = t.cashflow
        if 'Stock Based Compensation' in cf.index and 'Free Cash Flow' in cf.index:
            sbc = cf.loc['Stock Based Compensation'].iloc[0]
            fcf = cf.loc['Free Cash Flow'].iloc[0]
            sbc_rows.append({'ticker': symbol, 'SBC_as_%_of_FCF': round((sbc/fcf)*100, 2)})
        else:
            sbc_rows.append({'ticker': symbol, 'SBC_as_%_of_FCF': np.nan})
        info = t.info
        if info.get('totalDebt') is not None and info.get('totalCash') is not None and info.get('ebitda'):
            nd = info['totalDebt'] - info['totalCash']
            debt_ratios.append({'ticker': symbol, 'net_debt_to_ebitda': nd / info['ebitda']})
        else:
            debt_ratios.append({'ticker': symbol, 'net_debt_to_ebitda': np.nan})

    return pd.merge(pd.DataFrame(sbc_rows), pd.DataFrame(debt_ratios), on='ticker')


def grade_criteria(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applique les seuils et retourne df_score.
    """
    df_score = df[['ticker']].copy()
    # mapping: col_source -> (col_score, thresholds)
    criteria = {
        '10y_R2': ('Score_Linearite_Perf10y', {1.0: 0.8, 0.5: 0.6}),
        '5y_avg_annual_return_%': ('Score_Performance_5y', {1.0: 12, 0.5: 8}),
        '10y_avg_annual_return_%': ('Score_Performance_10y', {1.0: 12, 0.5: 8}),
        'Revenue_Growth_5Y': ('Score_RevenueGrowth_5y', {1.0: 8, 0.5: 5}),
        'Revenue_Growth_LastYear_%': ('Score_RevenueGrowth_LastYear', {1.0: 8, 0.5: 5}),
        'FreeCashFlow5Y': ('Score_FreeCashFlow5ans', {1.0: 14, 0.5: 10}),
        'EPS_Growth_5Y': ('Score_EPS5ans', {1.0: 12, 0.5: 8}),
        'EPS_Growth_3Y': ('Score_EPS3ans', {1.0: 12, 0.5: 8}),
        'ROIC_5Y': ('Score_ROI5ans', {1.0: 15, 0.5: 10}),
        'ROI_ANNUAL': ('Score_ROIannual', {1.0: 15, 0.5: 10}),
        'Gross_Margin_5Y': ('Score_GrossMargin5y', {1.0: 20, 0.5: 10}),
        'Gross_Margin_Annual': ('Score_GrossMarginAnnual', {1.0: 20, 0.5: 10}),
        'SBC_as_%_of_FCF': ('Score_SBCofFCF', {1.0: 0, 0.5: 10}),
        'net_debt_to_ebitda': ('Score_NetDebtToEBITDA', {1.0: 0, 0.5: 3}),
    }

    def apply_threshold(value, thresh):
        try:
            v = float(value)
        except:
            return 0.0
        # for descending thresholds like SBC or debt, flip logic
        if thresh is criteria['SBC_as_%_of_FCF'][1] or thresh is criteria['net_debt_to_ebitda'][1]:
            if v < list(thresh.values())[1]:
                return 1.0
            elif v < list(thresh.values())[0]:
                return 0.5
            else:
                return 0.0
        if v >= thresh[1.0]:
            return 1.0
        if v >= thresh[0.5]:
            return 0.5
        return 0.0

    for src, (score_col, thresh) in criteria.items():
        df_score[score_col] = df[src].apply(lambda x: apply_threshold(x, thresh))

    # total and normalization
    score_cols = [c for c in df_score.columns if c != 'ticker']
    df_score['Total_Score'] = df_score[score_cols].sum(axis=1)
    df_score['Valid_Criteria_Count'] = df_score[score_cols].notna().sum(axis=1)
    df_score['Score_sur_20'] = (df_score['Total_Score']/df_score['Valid_Criteria_Count'])*20
    df_score['Score_sur_20'] = df_score['Score_sur_20'].round(2)

    return df_score.sort_values('Score_sur_20', ascending=False)


def export_history(df_scores: pd.DataFrame, df_full: pd.DataFrame):
    """
    Export historique des scores et metrics complets.
    """
    today = datetime.datetime.today().strftime('%Y-%m-%d')
    now = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    # Scores
    hist_scores = df_scores[['ticker', 'Total_Score', 'Score_sur_20']].copy()
    hist_scores['date'] = today
    hist_scores['horodatage'] = now
    path_s = 'historique_scores.csv'
    if os.path.exists(path_s):
        pd.concat([pd.read_csv(path_s), hist_scores], ignore_index=True).to_csv(path_s, index=False)
    else:
        hist_scores.to_csv(path_s, index=False)

    # Metrics complets
    hist_all = df_full.copy()
    hist_all['date'] = today
    hist_all['horodatage'] = now
    path_m = 'historique_metrics.csv'
    if os.path.exists(path_m):
        pd.concat([pd.read_csv(path_m), hist_all], ignore_index=True).to_csv(path_m, index=False)
    else:
        hist_all.to_csv(path_m, index=False)


def main():
    # variables
    tickers = ["AMZN","ASML","NVDA","GOOG","BKNG","NEM.HA",
               "CRM","INTU","MA","MSFT","SPGI","V","SNY","IONQ","AAPL","TSLA","JNJ"]
    finnhub_key = os.getenv('FINNHUB_TOKEN') or 'csiada1r01qpalorrno0csiada1r01qpalorrnog'

    # Performance
    perf_rows = [
        [sym, *calculate_performance_metrics(sym, 10)] + [*calculate_performance_metrics(sym, 5)[1:2]]
        for sym in tickers
    ]
    df_perf = pd.DataFrame(perf_rows, columns=[
        'ticker','10y_avg_annual_return_%','10y_R2','5y_avg_annual_return_%'
    ])

    # SBC & dette
    df_sbc_debt = compute_sbc_and_debt(tickers)

    # Finnhub
    df_fh = fetch_finnhub_metrics(tickers, finnhub_key)

    # Merge
    df_all = df_perf.merge(df_sbc_debt, on='ticker').merge(df_fh, on='ticker')

    # Scores
    df_scores = grade_criteria(df_all)

    # Export historique
    export_history(df_scores, df_all)

    print("✅ Export historique terminé.")


if __name__ == '__main__':
    main()
