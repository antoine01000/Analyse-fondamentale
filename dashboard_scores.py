import streamlit as st
import pandas as pd
import plotly.express as px
import os

# Titre de l'app
st.title("📈 Évolution hebdomadaire des scores fondamentaux")

# Bouton de rafraîchissement
if st.button("🔄 Rafraîchir les données"):
    st.rerun()

# Chargement des données (pas de cache ici)
def load_data():
    if not os.path.exists("historique_scores.csv"):
        st.warning("Le fichier historique_scores.csv est introuvable.")
        return pd.DataFrame(columns=["ticker", "Total_Score", "Score_sur_20", "date"])
    
    df = pd.read_csv("historique_scores.csv")
    df['date'] = pd.to_datetime(df['date'])
    return df

# Appel réel
df = load_data()

# Diagnostic visuel
st.write("✅ Fichier chargé :")
st.dataframe(df)

st.write("📊 Tickers trouvés :")
st.write(df['ticker'].unique())

# Liste des tickers
tickers = df['ticker'].dropna().unique()
default = [t for t in ["AAPL", "MSFT", "GOOG", "TSLA"] if t in tickers]

# Sélection utilisateur
tickers_selection = st.multiselect(
    "Choisissez une ou plusieurs entreprises :",
    options=tickers,
    default=default
)

# Filtrage
df_filtered = df[df['ticker'].isin(tickers_selection)]
