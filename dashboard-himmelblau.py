import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D
import plotly.express as px
import plotly.graph_objects as go

#config de la page
st.set_page_config(page_title="Dashboard Himmelblau", layout="wide", page_icon="🎯")
st.title("🎯 Dashboard Interactif : Physics-Informed Neural Networks")
st.markdown("""
**Reproduction et amélioration de l'expérience de Himmelblau.** Ce dashboard présente l'analyse complète de 4 stratégies d'apprentissage profond pour régresser la fonction de Himmelblau, incluant l'intégration de l'auto-différenciation et du multitask learning.
""")

#dataloading
@st.cache_data
def load_data():
    df_main = pd.read_csv("Reproduction_Himmelblau_Final-V2.csv")
    df_stats = pd.read_csv("Analyse_Statistique_Himmelblau_V2.csv")
    df_tune = pd.read_csv("Resultats_Fine_Tuning_Lambdas.csv")
  
    strategy_mapping = {
        'Classic': 'u(x,y) only',
        'Gradient': '∇u(x,y) only',
        'Both auto-diff': 'Both (Auto-Diff)',
        'Both multitask': 'Both (Multitask)'
    }
    df_main['Strategy_Name'] = df_main['Strategy'].map(strategy_mapping)
    return df_main, df_stats, df_tune

df_main, df_stats, df_tune = load_data()

palette = {
    'u(x,y) only': '#2ca02c',          
    '∇u(x,y) only': '#d62728',  
    'Both (Auto-Diff)': '#1f77b4',       
    'Both (Multitask)': '#ff7f0e'       
}

#création onglet 
tab1, tab2, tab3, tab4 = st.tabs([
    "1.Performances (Erreurs)", 
    "2.Efficacité (Temps & Convergence)", 
    "3.Fine-Tuning (Lambdas)", 
    "4.Analyse Statistique"
])

#onglet 1
with tab1:
    st.header("Analyse de la Précision (MAE)")
    st.markdown("Comparaison des erreurs MAE en fonction de la taille du jeu d'entraînement ($N$).")
    
    plot_type = st.radio("Style de graphique :", ["Interactif (Plotly)", "Format Publication (Seaborn)"], horizontal=True)
    
    if plot_type == "Format Publication (Seaborn)":
        sns.set_theme(style="whitegrid", context="paper", font_scale=1.2, rc={"axes.edgecolor": "0.2", "axes.linewidth": 1.2})
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        n_ticks = sorted(df_main['Training_Labels'].unique())
        custom_lines = [Line2D([0], [0], color=color, lw=0, marker='o', markersize=8) for color in palette.values()]

        #Figure C
        ax = axes[0]
        sns.scatterplot(data=df_main, x="Training_Labels", y="MAE_Energy", hue="Strategy_Name", palette=palette, alpha=0.3, s=20, ax=ax, legend=False)
        sns.lineplot(data=df_main, x="Training_Labels", y="MAE_Energy", hue="Strategy_Name", palette=palette, errorbar=("ci", 95), err_style="band", linewidth=1.5, linestyle="--", marker="D", markersize=7, ax=ax, legend=False)
        for strat, color in palette.items():
            df_sub = df_main[(df_main['Strategy_Name'] == strat) & (df_main['MAE_Energy'].notnull())]
            if len(df_sub) > 1:
                x_log, y_log = np.log10(df_sub['Training_Labels']), np.log10(df_sub['MAE_Energy'])
                slope, intercept = np.polyfit(x_log, y_log, 1) 
                x_vals = np.unique(df_sub['Training_Labels'])
                ax.plot(x_vals, 10**(intercept + slope * np.log10(x_vals)), color=color, linewidth=2.5)
        ax.set_xscale('log'); ax.set_yscale('log'); ax.set_xticks(n_ticks); ax.set_xticklabels(n_ticks); ax.set_xticks([], minor=True)
        ax.set_title("(C) Error (Énergie)", fontweight='bold'); ax.set_xlabel("# N"); ax.set_ylabel("MAE function value")
        ax.legend(custom_lines, palette.keys(), loc="upper right", frameon=True, shadow=True) 

        #Figure D 
        ax = axes[1]
        sns.scatterplot(data=df_main, x="Training_Labels", y="MAE_Force", hue="Strategy_Name", palette=palette, alpha=0.3, s=20, ax=ax, legend=False)
        sns.lineplot(data=df_main, x="Training_Labels", y="MAE_Force", hue="Strategy_Name", palette=palette, errorbar=("ci", 95), err_style="band", linewidth=1.5, linestyle="--", marker="D", markersize=7, ax=ax, legend=False)
        for strat, color in palette.items():
            df_sub = df_main[(df_main['Strategy_Name'] == strat) & (df_main['MAE_Force'].notnull())]
            if len(df_sub) > 1:
                x_log, y_log = np.log10(df_sub['Training_Labels']), np.log10(df_sub['MAE_Force'])
                slope, intercept = np.polyfit(x_log, y_log, 1) 
                x_vals = np.unique(df_sub['Training_Labels'])
                ax.plot(x_vals, 10**(intercept + slope * np.log10(x_vals)), color=color, linewidth=2.5)
        ax.set_xscale('log'); ax.set_yscale('log'); ax.set_xticks(n_ticks); ax.set_xticklabels(n_ticks); ax.set_xticks([], minor=True)
        ax.set_title("(D) Gradient Error (Forces)", fontweight='bold'); ax.set_xlabel("# N"); ax.set_ylabel("MAE gradient component")
        ax.legend(custom_lines, palette.keys(), loc="upper right", frameon=True, shadow=True) 

        st.pyplot(fig)

    else:
        # Version Plotly
        st.info("💡 Survole les points pour voir les valeurs exactes des moyennes.")
        df_mean = df_main.groupby(['Training_Labels', 'Strategy_Name'])[['MAE_Energy', 'MAE_Force']].mean().reset_index()
        
        col1, col2 = st.columns(2)
        with col1:
            fig_e = px.line(df_mean, x="Training_Labels", y="MAE_Energy", color="Strategy_Name", markers=True, log_x=True, log_y=True, title="MAE Énergie (Moyenne)", color_discrete_map=palette)
            fig_e.update_layout(xaxis_title="Nombre d'exemples d'entraînement (N)", yaxis_title="Erreur (Log Scale)")
            st.plotly_chart(fig_e, width='stretch')
        with col2:
            fig_f = px.line(df_mean, x="Training_Labels", y="MAE_Force", color="Strategy_Name", markers=True, log_x=True, log_y=True, title="MAE Forces (Moyenne)", color_discrete_map=palette)
            fig_f.update_layout(xaxis_title="Nombre d'exemples d'entraînement (N)", yaxis_title="Erreur (Log Scale)")
            st.plotly_chart(fig_f, width='stretch')

#onglet 2
with tab2:
    st.header("Coût Computationnel et Vitesse de Convergence")
    st.markdown("L'approche physique (Auto-diff) est-elle plus lente à s'entraîner ? Converge-t-elle en moins d'epochs ?")
    
    # Sélecteur interactif pour filtrer N
    selected_n = st.select_slider("Sélectionnez la taille du jeu de données (N) :", options=sorted(df_main['Training_Labels'].unique()), value=1000)
    
    df_n = df_main[df_main['Training_Labels'] == selected_n]
    
    col1, col2 = st.columns(2)
    with col1:
        # Graphique des temps d'entraînement (Boxplot)
        fig_time = px.box(df_n, x="Strategy_Name", y="Time_Seconds", color="Strategy_Name", title=f"Temps d'entraînement par stratégie (N={selected_n})", color_discrete_map=palette)
        fig_time.update_layout(xaxis_title="Stratégie", yaxis_title="Temps (secondes)", showlegend=False)
        st.plotly_chart(fig_time, width='stretch')
        
    with col2:
        # Graphique des epochs (Early Stopping)
        fig_epochs = px.box(df_n, x="Strategy_Name", y="Epochs", color="Strategy_Name", title=f"Nombre d'Epochs avant convergence (N={selected_n})", color_discrete_map=palette)
        fig_epochs.update_layout(xaxis_title="Stratégie", yaxis_title="Epochs", showlegend=False)
        st.plotly_chart(fig_epochs, width='stretch')
        
    st.divider()
    st.subheader("Frontière de Pareto : Erreur vs Temps d'entraînement")
    st.markdown("Ce graphique permet de voir quelle méthode offre le meilleur compromis **Précision / Temps**")
    
    df_pareto = df_main.groupby('Strategy_Name')[['MAE_Energy', 'Time_Seconds']].mean().reset_index()
    fig_pareto = px.scatter(df_pareto, x="Time_Seconds", y="MAE_Energy", color="Strategy_Name", size=[3]*len(df_pareto), size_max=15, title="Compromis Erreur Énergie vs Temps d'entraînement (Moyenne globale)", color_discrete_map=palette)
    fig_pareto.update_layout(xaxis_title="Temps d'entraînement moyen (secondes)", yaxis_title="MAE Énergie (Moyenne)")
    st.plotly_chart(fig_pareto, width='stretch')


#onglet 3
with tab3:
    st.header("Recherche des Hyperparamètres (Paramètre $\\lambda$)")
    st.markdown("Impact du poids accordé aux gradients physiques dans la fonction de perte finale.")
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Both auto-diff")
        df_autodiff = df_tune[df_tune['Stratégie'] == 'Both auto-diff'].sort_values('Lambda (λ)')
        fig1 = px.line(df_autodiff, x='Lambda (λ)', y='Erreur Moyenne', error_y='Écart-Type', markers=True, title="Impact de λ sur Auto-Diff")
        fig1.update_traces(line_color=palette['Both (Auto-Diff)'])
        st.plotly_chart(fig1, width='stretch')
        
        best_auto = df_autodiff.sort_values("Erreur Moyenne").iloc[0]
        st.success(f"**Vainqueur Auto-Diff :** λ = {best_auto['Lambda (λ)']} (Erreur : {best_auto['Erreur Moyenne']:.5f})")

    with col2:
        st.subheader("Both multitask")
        df_multi = df_tune[df_tune['Stratégie'] == 'Both multitask'].sort_values('Lambda (λ)')
        fig2 = px.line(df_multi, x='Lambda (λ)', y='Erreur Moyenne', error_y='Écart-Type', markers=True, title="Impact de λ sur Multitask")
        fig2.update_traces(line_color=palette['Both (Multitask)'])
        st.plotly_chart(fig2, width='stretch')
        
        best_multi = df_multi.sort_values("Erreur Moyenne").iloc[0]
        st.success(f"**Vainqueur Multitask :** λ = {best_multi['Lambda (λ)']} (Erreur : {best_multi['Erreur Moyenne']:.5f})")


#onglet 4
with tab4:
    st.header("Significativité des Résultats")
    st.markdown("Les différences observées sur les graphiques sont-elles statistiquement viables ? (Tests de Mann-Whitney)")
    
    col_a, col_b = st.columns(2)
    metric_filter = col_a.selectbox("📌 Choisir la métrique évaluée :", ["MAE_Energy", "MAE_Force"])
    n_filter = col_b.selectbox("📌 Choisir la taille d'entraînement (N) :", sorted(df_stats['N (Exemples)'].unique()))
    
    st.subheader(f"Comparaison deux-à-deux pour N = {n_filter}")
    filtered_stats = df_stats[(df_stats['Métrique'] == metric_filter) & (df_stats['N (Exemples)'] == n_filter)].copy()
    
    # Mise en forme du tableau Pandas
    def highlight_sig(val):
        color = '#d4edda' if 'Oui' in str(val) else '#f8d7da'
        return f'background-color: {color}'
        
    st.dataframe(filtered_stats.style.map(highlight_sig, subset=['Différence Significative']), width='stretch')