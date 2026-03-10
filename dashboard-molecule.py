import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D
import plotly.express as px
from scipy.stats import mannwhitneyu
import itertools

molecule_name = "aspirin"

st.set_page_config(page_title="Dashboard MD17", layout="wide", page_icon="🎯")
st.title("🎯 Dashboard Interactif : Physics-Informed Neural Networks sur MD17")
st.markdown("""
Ce dashboard présente l'analyse complète de différentes stratégies d'apprentissage profond pour régresser l'énergie et les forces, en comparant les approches classiques, basées sur les gradients, et celles utilisant l'auto-différenciation.
""")

@st.cache_data
def load_data():
    df_main = pd.read_csv(f"metrics_md17_7reps_{molecule_name}.csv")
    
    strategy_mapping = {
        'classic': 'Classic (Energy only)',
        'gradient': 'Gradient (Forces only)',
        'auto-diff': 'Auto-diff (Energy + Forces)'
    }
    df_main['Strategy_Name'] = df_main['Strategy'].map(strategy_mapping)
    return df_main

df_main = load_data()

palette = {
    'Classic (Energy only)': '#2ca02c',          
    'Gradient (Forces only)': '#d62728',  
    'Auto-diff (Energy + Forces)': '#1f77b4'       
}

tab1, tab2, tab3, tab4 = st.tabs([
    "1. Performances (Erreurs)", 
    "2. Efficacité (Temps & Convergence)", 
    "3. Distribution des Erreurs", 
    "4. Analyse Statistique"
])

# Onglet 1
with tab1:
    st.header("Analyse de la Précision (MAE)")
    st.markdown("Comparaison des erreurs MAE en fonction de la taille du jeu d'entraînement ($N$).")
    
    plot_type = st.radio("Style de graphique :", ["Interactif (Plotly)", "Format Publication (Seaborn)"], horizontal=True)
    
    if plot_type == "Format Publication (Seaborn)":
        sns.set_theme(style="whitegrid", context="paper", font_scale=1.2, rc={"axes.edgecolor": "0.2", "axes.linewidth": 1.2})
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        n_ticks = sorted(df_main['N_Train'].unique())
        custom_lines = [Line2D([0], [0], color=color, lw=0, marker='o', markersize=8) for color in palette.values()]

        # Figure C (Energy)
        ax = axes[0]
        sns.scatterplot(data=df_main, x="N_Train", y="MAE_Energy", hue="Strategy_Name", palette=palette, alpha=0.3, s=20, ax=ax, legend=False)
        sns.lineplot(data=df_main, x="N_Train", y="MAE_Energy", hue="Strategy_Name", palette=palette, errorbar=("ci", 95), err_style="band", linewidth=1.5, linestyle="--", marker="D", markersize=7, ax=ax, legend=False)
        
        for strat, color in palette.items():
            df_sub = df_main[(df_main['Strategy_Name'] == strat) & (df_main['MAE_Energy'].notnull())]
            if len(df_sub) > 1:
                x_log, y_log = np.log10(df_sub['N_Train']), np.log10(df_sub['MAE_Energy'])
                slope, intercept = np.polyfit(x_log, y_log, 1) 
                x_vals = np.unique(df_sub['N_Train'])
                ax.plot(x_vals, 10**(intercept + slope * np.log10(x_vals)), color=color, linewidth=2.5)
        
        ax.set_xscale('log'); ax.set_yscale('log'); ax.set_xticks(n_ticks); ax.set_xticklabels(n_ticks); ax.set_xticks([], minor=True)
        ax.set_title("(A) Error (Énergie)", fontweight='bold'); ax.set_xlabel("N_Train"); ax.set_ylabel("MAE Energy")
        ax.legend(custom_lines, palette.keys(), loc="upper right", frameon=True, shadow=True) 

        # Figure D (Forces)
        ax = axes[1]
        sns.scatterplot(data=df_main, x="N_Train", y="MAE_Force", hue="Strategy_Name", palette=palette, alpha=0.3, s=20, ax=ax, legend=False)
        sns.lineplot(data=df_main, x="N_Train", y="MAE_Force", hue="Strategy_Name", palette=palette, errorbar=("ci", 95), err_style="band", linewidth=1.5, linestyle="--", marker="D", markersize=7, ax=ax, legend=False)
        
        for strat, color in palette.items():
            df_sub = df_main[(df_main['Strategy_Name'] == strat) & (df_main['MAE_Force'].notnull())]
            if len(df_sub) > 1:
                x_log, y_log = np.log10(df_sub['N_Train']), np.log10(df_sub['MAE_Force'])
                slope, intercept = np.polyfit(x_log, y_log, 1) 
                x_vals = np.unique(df_sub['N_Train'])
                ax.plot(x_vals, 10**(intercept + slope * np.log10(x_vals)), color=color, linewidth=2.5)
        
        ax.set_xscale('log'); ax.set_yscale('log'); ax.set_xticks(n_ticks); ax.set_xticklabels(n_ticks); ax.set_xticks([], minor=True)
        ax.set_title("(B) Gradient Error (Forces)", fontweight='bold'); ax.set_xlabel("N_Train"); ax.set_ylabel("MAE Forces")
        ax.legend(custom_lines, palette.keys(), loc="upper right", frameon=True, shadow=True) 

        st.pyplot(fig)

    else:
        # Version Plotly
        st.info("💡 Survole les points pour voir les valeurs exactes des moyennes.")
        df_mean = df_main.groupby(['N_Train', 'Strategy_Name'])[['MAE_Energy', 'MAE_Force']].mean().reset_index()
        
        col1, col2 = st.columns(2)
        with col1:
            fig_e = px.line(df_mean, x="N_Train", y="MAE_Energy", color="Strategy_Name", markers=True, log_x=True, log_y=True, title="MAE Énergie (Moyenne)", color_discrete_map=palette)
            fig_e.update_layout(xaxis_title="Nombre d'exemples d'entraînement (N)", yaxis_title="Erreur (Log Scale)")
            st.plotly_chart(fig_e, width='stretch')
        with col2:
            fig_f = px.line(df_mean, x="N_Train", y="MAE_Force", color="Strategy_Name", markers=True, log_x=True, log_y=True, title="MAE Forces (Moyenne)", color_discrete_map=palette)
            fig_f.update_layout(xaxis_title="Nombre d'exemples d'entraînement (N)", yaxis_title="Erreur (Log Scale)")
            st.plotly_chart(fig_f, width='stretch')

# Onglet 2
with tab2:
    st.header("Coût Computationnel et Vitesse de Convergence")
    st.markdown("Comparaison des temps d'entraînement et du nombre d'époques nécessaires.")
    
    # Sélecteur interactif pour filtrer N
    selected_n = st.select_slider("Sélectionnez la taille du jeu de données (N_Train) :", options=sorted(df_main['N_Train'].unique()), value=sorted(df_main['N_Train'].unique())[-1])
    
    df_n = df_main[df_main['N_Train'] == selected_n]
    
    col1, col2 = st.columns(2)
    with col1:
        # Graphique des temps d'entraînement
        fig_time = px.box(df_n, x="Strategy_Name", y="Train_Time_s", color="Strategy_Name", title=f"Temps d'entraînement par stratégie (N={selected_n})", color_discrete_map=palette)
        fig_time.update_layout(xaxis_title="Stratégie", yaxis_title="Temps (secondes)", showlegend=False)
        st.plotly_chart(fig_time, width='stretch')
        
    with col2:
        # Graphique des epochs
        fig_epochs = px.box(df_n, x="Strategy_Name", y="Epochs_Run", color="Strategy_Name", title=f"Nombre d'Epochs avant convergence (N={selected_n})", color_discrete_map=palette)
        fig_epochs.update_layout(xaxis_title="Stratégie", yaxis_title="Epochs", showlegend=False)
        st.plotly_chart(fig_epochs, width='stretch')
        
    st.divider()
    st.subheader("Frontière de Pareto : Erreur vs Temps d'entraînement")
    st.markdown("Ce graphique permet de voir quelle méthode offre le meilleur compromis **Précision / Temps**")
    
    # Pareto calculation
    df_pareto = df_main.groupby('Strategy_Name')[['MAE_Energy', 'Train_Time_s']].mean().reset_index()
    fig_pareto = px.scatter(df_pareto, x="Train_Time_s", y="MAE_Energy", color="Strategy_Name", size=[3]*len(df_pareto), size_max=15, title="Compromis Erreur Énergie vs Temps d'entraînement (Moyenne)", color_discrete_map=palette)
    fig_pareto.update_layout(xaxis_title="Temps d'entraînement moyen (secondes)", yaxis_title="MAE Énergie (Moyenne)")
    st.plotly_chart(fig_pareto, width='stretch')

# Onglet 3 (Remplacé l'onglet fine-tuning qui manque, par l'analyse des distributions)
with tab3:
    st.header("Distribution et Variabilité des Erreurs")
    st.markdown("Grâce aux 7 répétitions pour chaque expérience, nous pouvons observer la robustesse (ou variance) de chaque modèle.")
    
    dist_metric = st.radio("Métrique à analyser :", ["MAE_Energy", "MAE_Force"])
    
    fig_dist = px.violin(df_main, x="Strategy_Name", y=dist_metric, color="Strategy_Name", box=True, points="all", hover_data=["N_Train"], color_discrete_map=palette, title=f"Distribution de {dist_metric} sur toutes les configurations")
    fig_dist.update_layout(xaxis_title="Stratégie", yaxis_title="Erreur", showlegend=False)
    st.plotly_chart(fig_dist, width='stretch')

# Onglet 4
with tab4:
    st.header("Significativité des Résultats")
    st.markdown("Génération dynamique de tests de Mann-Whitney (P-value < 0.05) pour comparer statistiquement les stratégies.")
    
    col_a, col_b = st.columns(2)
    metric_filter = col_a.selectbox("📌 Choisir la métrique évaluée :", ["MAE_Energy", "MAE_Force", "Train_Time_s", "Epochs_Run"])
    n_filter = col_b.selectbox("📌 Choisir la taille d'entraînement (N) :", sorted(df_main['N_Train'].unique()))
    
    st.subheader(f"Comparaison deux-à-deux pour N = {n_filter}")
    
    # Calcul dynamique des statistiques
    df_stat_calc = df_main[df_main['N_Train'] == n_filter]
    strategies = df_stat_calc['Strategy_Name'].unique()
    
    stat_results = []
    for s1, s2 in itertools.combinations(strategies, 2):
        d1 = df_stat_calc[df_stat_calc['Strategy_Name'] == s1][metric_filter].dropna()
        d2 = df_stat_calc[df_stat_calc['Strategy_Name'] == s2][metric_filter].dropna()
        
        if len(d1) > 0 and len(d2) > 0:
            stat, pval = mannwhitneyu(d1, d2, alternative='two-sided')
            signif = "Oui ✅" if pval < 0.05 else "Non ❌"
            
            mean1 = d1.mean()
            mean2 = d2.mean()
            gagnant = s1 if mean1 < mean2 else s2
            
            stat_results.append({
                "Stratégie A": s1,
                "Stratégie B": s2,
                "P-Value": f"{pval:.4f}",
                "Différence Significative": signif,
                "Vainqueur (Moyenne plus faible)": gagnant
            })
            
    if stat_results:
        df_stats_show = pd.DataFrame(stat_results)
        
        # Mise en forme
        def highlight_sig(val):
            if 'Oui' in str(val):
                return 'background-color: #d4edda'
            elif 'Non' in str(val):
                return 'background-color: #f8d7da'
            return ''
            
        st.dataframe(df_stats_show.style.map(highlight_sig, subset=['Différence Significative']), width='stretch')
    else:
        st.warning("Pas assez de données pour effectuer le test statistique.")