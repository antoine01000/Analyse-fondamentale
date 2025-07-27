import streamlit as st
import pandas as pd
import plotly.express as px
import os

# Titre de l'app
st.title("📈 Évolution hebdomadaire des scores fondamentaux")

# Bouton de rafraîchissement
if st.button("🔄 Rafraîchir les données"):
    st.rerun()

# Fonction de chargement des données
def load_data():
    csv_path = "historique_scores.csv"
    if not os.path.exists(csv_path):
        st.warning(f"Le fichier '{csv_path}' est introuvable.")
        return pd.DataFrame(columns=["ticker", "Total_Score", "Score_sur_20", "date"])
    df = pd.read_csv(csv_path)
    df['date'] = pd.to_datetime(df['date'])
    return df

# Chargement réel
df = load_data()

# Diagnostic visuel
st.write("✅ Fichier chargé :")
st.dataframe(df)

st.write("📊 Tickers trouvés :")
st.write(df['ticker'].dropna().unique())

# --- Sélection des tickers ---
tickers = df['ticker'].dropna().unique()
default = [t for t in ["AAPL", "MSFT", "GOOG", "TSLA"] if t in tickers]

tickers_selection = st.multiselect(
    "Choisissez une ou plusieurs entreprises :",
    options=tickers,
    default=default
)

# Filtrage sur la sélection
df_filtered = df[df['ticker'].isin(tickers_selection)]

# ---------- Affichage du graphique ----------
if not df_filtered.empty:
    fig = px.line(
        df_filtered,
        x='date',
        y='Score_sur_20',
        color='ticker',
        markers=True,
        title='Évolution du Score sur 20 par entreprise',
        labels={'Score_sur_20': 'Note / 20', 'date': 'Date'}
    )
    fig.update_layout(xaxis_title="Date", yaxis_title="Score sur 20")
    st.plotly_chart(fig, use_container_width=True)
else:
    st.warning("Aucune donnée disponible pour la sélection.")

# ---------- Affichage optionnel de la table ----------
if st.checkbox("Afficher les données brutes"):
    st.dataframe(df_filtered.sort_values(by="date", ascending=False))
