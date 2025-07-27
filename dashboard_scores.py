import streamlit as st
import pandas as pd
import plotly.express as px
import os

# Titre de l'app
st.title("📈 Évolution hebdomadaire des scores fondamentaux")


# Titre de l'app
st.title("📈 Évolution hebdomadaire des scores fondamentaux")

# Bouton de rafraîchissement
if st.button("🔄 Rafraîchir les données"):
    st.rerun()

# Chargement des données
def load_data():
    if not os.path.exists("historique_scores.csv"):
        st.warning("Le fichier historique_scores.csv est introuvable.")
        return pd.DataFrame(columns=["ticker", "Total_Score", "Score_sur_20", "date"])
    
    df = pd.read_csv("historique_scores.csv")
    df['date'] = pd.to_datetime(df['date'])
    return df

df = load_data()

# Chargement des données
@st.cache_data
def load_data():
    df = pd.read_csv("historique_scores.csv")
    df['date'] = pd.to_datetime(df['date'])
    return df

df = load_data()

# Liste des tickers
tickers = df['ticker'].unique()

# Définir une valeur par défaut uniquement si elle est dans les données
default = [t for t in ["AAPL", "MSFT"] if t in tickers]

# Sélection utilisateur
tickers_selection = st.multiselect(
    "Choisissez une ou plusieurs entreprises :",
    options=tickers,
    default=default
)

# Filtrage
df_filtered = df[df['ticker'].isin(tickers_selection)]

# Vérification
if df_filtered.empty:
    st.warning("Aucune donnée disponible pour la sélection.")
else:
    # Graphique interactif
    fig = px.line(
        df_filtered,
        x='date',
        y='Score_sur_20',
        color='ticker',
        markers=True,
        title='Évolution du Score sur 20 par entreprise',
        labels={'Score_sur_20': 'Note / 20'}
    )
    fig.update_layout(xaxis_title="Date", yaxis_title="Score sur 20")
    st.plotly_chart(fig)

    # Option : afficher la table
    if st.checkbox("Afficher les données brutes"):
        st.dataframe(df_filtered.sort_values(by="date", ascending=False))
