# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Por que clientes de e-commerce cancelam? Auditoria, correlações entre fatores e modelos de churn
#
# **Autor:** João Gabriel Tavares Silva · Universidade de Brasília (UnB) · Matrícula 221021071
#
# [![Abrir no Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/joaogabrielts0611-ai/analise-churn-ecommerce/blob/main/analise_churn_ecommerce.ipynb)
#
# **Base:** arquivo `eCommerce-dataset.csv`, base pública do Kaggle sobre comportamento de clientes de e-commerce: 50.000 clientes de uma loja on-line em 8 países, 25 variáveis de perfil, engajamento, compras e atrito, e a indicação de cancelamento (`Churned`).
#
# > **Atenção: a base é simulada.** A Seção 1 mostra as evidências: faltantes em proporções redondas exatas, cidades e trimestres distribuídos de forma perfeita, valores impossíveis inseridos de propósito. As conclusões valem para esta base e para o método, não como retrato do comportamento real de consumidores.
#
# **O que este notebook faz**
#
# 1. **Auditoria.** Faltantes, valores impossíveis, consistência das categorias e sinais de geração sintética.
# 2. **Limpeza documentada.** Cada regra explicada, com o efeito antes e depois.
# 3. **Correlações entre pares.** As 190 combinações de variáveis numéricas: Pearson × Spearman (o efeito dos outliers), tamanho do efeito, clusters, fator comum (PCA) e correlação parcial.
# 4. **Fatores de churn.** Taxa de cancelamento por faixa de cada variável, regressão logística, modelo de árvores (gradient boosting), importância por permutação e efeitos de limiar.
# 5. **Painel interativo.** Explorador de variável, comparador de pares, segmentação e simulador de risco de churn.
#
# > **Como usar:** no Colab, `Ambiente de execução → Executar tudo` (cerca de 1 a 2 minutos). Os gráficos são interativos. Os controles da Seção 6 precisam do notebook em execução.

# %% [markdown]
# ## 0. Configuração

# %%
import os, re, sys, io, contextlib, warnings
warnings.filterwarnings("ignore")

IN_COLAB = "google.colab" in sys.modules
# NB_HEADLESS=1: execução automática (sem navegador); os painéis mostram só o resultado inicial
HEADLESS = os.environ.get("NB_HEADLESS") == "1"
if IN_COLAB:
    # permite que os gráficos Plotly apareçam dentro dos controles interativos (ipywidgets)
    from google.colab import output as _colab_output
    _colab_output.enable_custom_widget_manager()

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots
from scipy import stats
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import squareform
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.api as sm
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance, partial_dependence
from sklearn.metrics import roc_auc_score
import ipywidgets as widgets
from IPython.display import display, Markdown

pd.set_option("display.max_columns", 30)
pd.set_option("display.width", 200)

def md(texto):
    """Exibe Markdown com vírgula decimal (padrão brasileiro)."""
    display(Markdown(re.sub(r"(?<=\d)\.(?=\d)", ",", texto)))

pio.renderers.default = os.environ.get("PLOTLY_RENDERER", "colab" if IN_COLAB else "notebook_connected")

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
INK, INK2, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
DIVERGING = [[0.0, "#e34948"], [0.5, "#f0efec"], [1.0, "#2a78d6"]]
SEQ = [[0.0, "#cde2fb"], [0.5, "#3987e5"], [1.0, "#0d366b"]]
pio.templates["estudo"] = go.layout.Template(layout=dict(
    font=dict(family="system-ui, -apple-system, Segoe UI, sans-serif", size=13, color=INK),
    paper_bgcolor=SURFACE, plot_bgcolor=SURFACE, colorway=SERIES,
    title=dict(font=dict(size=16, color=INK), x=0, xanchor="left"),
    xaxis=dict(gridcolor=GRID, linecolor=AXIS, zerolinecolor=AXIS, tickfont=dict(color=MUTED)),
    yaxis=dict(gridcolor=GRID, linecolor=AXIS, zerolinecolor=AXIS, tickfont=dict(color=MUTED)),
    legend=dict(font=dict(color=INK2)), margin=dict(l=60, r=30, t=80, b=50)))
pio.templates.default = "plotly_white+estudo"
print("Ambiente:", "Google Colab" if IN_COLAB else "Jupyter local", "| pandas", pd.__version__)

# %% [markdown]
# ### 0.1 Carga dos dados

# %%
REPO_RAW = "https://raw.githubusercontent.com/joaogabrielts0611-ai/analise-churn-ecommerce/main/data/"
ARQ = "eCommerce-dataset.csv"
bruto = pd.read_csv(os.path.join("data", ARQ) if os.path.exists(os.path.join("data", ARQ)) else REPO_RAW + ARQ)
print(f"{bruto.shape[0]:,} clientes × {bruto.shape[1]} variáveis".replace(",", "."))
bruto.head()

# %% [markdown]
# ### 0.2 Dicionário de variáveis
# Cada variável recebe um nome em português e um **bloco** temático, que dá a cor nos gráficos.

# %%
VARS = {
    "Age": ("Idade", "Perfil"), "Membership_Years": ("Anos de cadastro", "Perfil"),
    "Login_Frequency": ("Logins por mês", "Engajamento"), "Session_Duration_Avg": ("Duração média da sessão (min)", "Engajamento"),
    "Pages_Per_Session": ("Páginas por sessão", "Engajamento"), "Wishlist_Items": ("Itens na lista de desejos", "Engajamento"),
    "Email_Open_Rate": ("Abertura de e-mails (%)", "Engajamento"), "Social_Media_Engagement_Score": ("Engajamento em redes sociais", "Engajamento"),
    "Mobile_App_Usage": ("Uso do app", "Engajamento"), "Product_Reviews_Written": ("Avaliações escritas", "Engajamento"),
    "Total_Purchases": ("Total de compras", "Compras"), "Average_Order_Value": ("Ticket médio (US$)", "Compras"),
    "Days_Since_Last_Purchase": ("Dias desde a última compra", "Compras"), "Discount_Usage_Rate": ("Uso de descontos (%)", "Compras"),
    "Returns_Rate": ("Taxa de devolução (%)", "Compras"), "Payment_Method_Diversity": ("Meios de pagamento usados", "Compras"),
    "Lifetime_Value": ("Valor do cliente (LTV, US$)", "Compras"), "Credit_Balance": ("Saldo de crédito (US$)", "Compras"),
    "Cart_Abandonment_Rate": ("Abandono de carrinho (%)", "Atrito"), "Customer_Service_Calls": ("Ligações ao atendimento", "Atrito"),
}
BLOCOS = ["Engajamento", "Compras", "Atrito", "Perfil"]
COR_BLOCO = dict(zip(BLOCOS, SERIES[:4]))
ROT = {k: v[0] for k, v in VARS.items()}
BLOCO = {k: v[1] for k, v in VARS.items()}
NUM = list(VARS)
CAT = {"Gender": "Gênero", "Country": "País", "City": "Cidade", "Signup_Quarter": "Trimestre de cadastro"}
assert set(NUM) | set(CAT) | {"Churned"} == set(bruto.columns)
pd.DataFrame({"variável": NUM, "nome": [ROT[v] for v in NUM], "bloco": [BLOCO[v] for v in NUM]})

# %% [markdown]
# ---
# ## 1. Auditoria da base
#
# ### 1.1 Integridade e categorias

# %%
churn_rate = bruto["Churned"].mean()
md(f"* Linhas duplicadas: **{bruto.duplicated().sum()}**. Taxa de churn geral: **{churn_rate:.1%}**.\n"
   "* Gênero: " + ", ".join(f"{k} {v:,}".replace(",", ".") for k, v in bruto["Gender"].value_counts().items()) + ".\n"
   f"* {bruto['Country'].nunique()} países e {bruto['City'].nunique()} cidades: **exatamente {int(bruto.groupby('Country')['City'].nunique().iloc[0])} cidades por país**.\n"
   f"* Trimestre de cadastro: {', '.join(f'{k} {v / len(bruto):.1%}' for k, v in bruto['Signup_Quarter'].value_counts().sort_index().items())}: praticamente uniforme.")

# %% [markdown]
# ### 1.2 Dados faltantes: quanto falta e se a falta "diz algo"
# Se faltar dado fosse aleatório, a taxa de churn seria igual entre quem tem e quem não tem o dado. Testamos com qui-quadrado e corrigimos para múltiplas comparações.

# %%
falt = []
for c in bruto.columns:
    m = bruto[c].isna()
    if m.sum() == 0:
        continue
    tab = pd.crosstab(m, bruto["Churned"])
    p = stats.chi2_contingency(tab)[1]
    falt.append(dict(variavel=c, nome=ROT.get(c, c), faltando=m.mean(), churn_se_falta=bruto.loc[m, "Churned"].mean(),
                     churn_se_tem=bruto.loc[~m, "Churned"].mean(), p=p))
falt = pd.DataFrame(falt).sort_values("faltando", ascending=False)
falt["p_fdr"] = multipletests(falt["p"], method="fdr_bh")[1]
falt["falta informativa?"] = np.where(falt["p_fdr"] < 0.05, "sim", "não")

fig = go.Figure(go.Bar(x=falt["faltando"] * 100, y=falt["nome"], orientation="h", marker_color=SERIES[0],
                       text=[f"{v:.1%}" for v in falt["faltando"]], textposition="outside",
                       hovertemplate="%{y}: %{x:.2f}% faltando<extra></extra>"))
fig.update_layout(title="Percentual de valores faltantes por variável", xaxis_title="% faltando", height=480,
                  yaxis=dict(autorange="reversed"), xaxis=dict(range=[0, 14]))
fig.show()
falt[["nome", "faltando", "churn_se_falta", "churn_se_tem", "p_fdr", "falta informativa?"]].style.format(
    {"faltando": "{:.2%}", "churn_se_falta": "{:.1%}", "churn_se_tem": "{:.1%}", "p_fdr": "{:.1e}"}, decimal=",").hide(axis="index")

# %% [markdown]
# Duas leituras. Primeiro, as proporções são **redondas demais** (6,0%, 7,0%, 8,0%, 10,0%, 12,0%…), algo improvável em dados coletados. Segundo, a falta é informativa sobretudo em **duração da sessão** e **abertura de e-mails**: quem não tem o dado cancela cerca de 13 pontos percentuais a mais (em idade e uso de descontos o efeito é significativo, mas pequeno). Por isso a modelagem usa indicadores de "dado faltante" em vez de simplesmente descartar linhas.

# %% [markdown]
# ### 1.3 Valores impossíveis

# %%
regras = {
    "Idade < 16 ou > 100 anos": (bruto["Age"] < 16) | (bruto["Age"] > 100),
    "Total de compras negativo": bruto["Total_Purchases"] < 0,
    "Abandono de carrinho > 100%": bruto["Cart_Abandonment_Rate"] > 100,
    "Uso de descontos > 100%": bruto["Discount_Usage_Rate"] > 100,
    "Ticket médio > US$ 1.000 (≈ 8× o p99)": bruto["Average_Order_Value"] > 1000,
    "Total de compras > 60 (≈ 2× o p99)": bruto["Total_Purchases"] > 60,
    "Taxa de devolução > 50%": bruto["Returns_Rate"] > 50,
}
imp = pd.DataFrame({"regra": list(regras), "linhas": [int(v.sum()) for v in regras.values()]})
imp["exemplos"] = [", ".join(f"{x:g}" for x in sorted(set(bruto.loc[m, col].round(1)))[:5]) for m, col in
                   zip(regras.values(), ["Age", "Total_Purchases", "Cart_Abandonment_Rate", "Discount_Usage_Rate",
                                          "Average_Order_Value", "Total_Purchases", "Returns_Rate"])]
md(f"**{int(sum(v.sum() for v in regras.values()))} ocorrências** em {int(np.logical_or.reduce(list(regras.values())).sum())} clientes "
   f"({np.logical_or.reduce(list(regras.values())).mean():.2%} da base). As idades impossíveis são sempre os mesmos valores: "
   f"{', '.join(f'{v:g}' for v in sorted(bruto.loc[regras['Idade < 16 ou > 100 anos'], 'Age'].unique()))} anos, típicos de erro inserido.")
imp.style.hide(axis="index")

# %% [markdown]
# ### 1.4 Evidências de base simulada
#
# | Evidência | O que se vê |
# |---|---|
# | Faltantes em proporções redondas | 5,0%, 6,0%, 7,0%, 8,0%, 10,0%, 11,0%, 12,0% em colunas diferentes |
# | Geografia artificial | exatamente 5 cidades por país, sempre as maiores |
# | Cadastro uniforme | 25% dos clientes em cada trimestre |
# | Erros "plantados" | idades de 5, 10, 150 e 200 anos; compras negativas inteiras; taxas acima de 100% |
# | Variável de contagem com decimais | total de compras com valores como 9,1 e 19,5 |
#
# Nada disso impede a análise, mas muda o que se pode concluir: os resultados descrevem as relações **embutidas pelo gerador da base**, e o maior valor do exercício é o método.

# %%
n_dec = int(((bruto["Total_Purchases"] % 1) != 0).sum())
md(f"Total de compras com casas decimais: **{n_dec:,} clientes** ({n_dec / len(bruto):.1%}).".replace(",", "."))

# %% [markdown]
# ### 1.5 Regras de limpeza
# * Valores impossíveis (idade fora de 16–100, compras negativas, taxas acima de 100%) → **faltante**.
# * Extremos plausíveis mas raros (ticket > US\$ 1.000, compras > 60, devolução > 50%) → **mantidos**, e as análises usam medidas robustas (Spearman, quantis, árvores).
# * Faltantes → mantidos como faltantes nas correlações (pares completos); nos modelos, imputação pela mediana **com indicador de falta**.

# %%
df = bruto.copy()
df.loc[(df["Age"] < 16) | (df["Age"] > 100), "Age"] = np.nan
df.loc[df["Total_Purchases"] < 0, "Total_Purchases"] = np.nan
for c in ["Cart_Abandonment_Rate", "Discount_Usage_Rate", "Returns_Rate"]:
    df.loc[df[c] > 100, c] = np.nan
antes_depois = pd.DataFrame({"média antes": bruto[NUM].mean(), "média depois": df[NUM].mean(),
                             "máx. antes": bruto[NUM].max(), "máx. depois": df[NUM].max()}).rename(index=ROT)
antes_depois.loc[antes_depois["máx. antes"] != antes_depois["máx. depois"]].style.format("{:,.1f}", decimal=",", thousands=".")

# %% [markdown]
# ---
# ## 2. Visão geral do churn
# Antes das correlações: o cancelamento varia por país, gênero ou trimestre de cadastro?

# %%
lin = []
for c, nome in CAT.items():
    if c == "City":
        continue
    tab = pd.crosstab(df[c], df["Churned"])
    chi2, p, _, _ = stats.chi2_contingency(tab)
    v = np.sqrt(chi2 / (tab.values.sum() * (min(tab.shape) - 1)))
    lin.append(dict(variavel=nome, menor=f"{df.groupby(c)['Churned'].mean().min():.1%}".replace(".", ","),
                    maior=f"{df.groupby(c)['Churned'].mean().max():.1%}".replace(".", ","), p=p, cramer_v=v))
cat_tab = pd.DataFrame(lin)
md(f"Taxa de churn: **{churn_rate:.1%}**. Entre categorias de perfil, a taxa varia pouco; o tamanho de efeito (V de Cramér) fica abaixo de 0,02 em todas, "
   "o que é desprezível mesmo quando o p-valor é pequeno. **O perfil demográfico quase não separa quem cancela.**")
cat_tab.style.format({"p": "{:.3f}", "cramer_v": "{:.3f}"}, decimal=",").hide(axis="index")

# %%
fig = make_subplots(rows=1, cols=3, subplot_titles=("País", "Gênero", "Trimestre de cadastro"), horizontal_spacing=0.08)
for k, c in enumerate(["Country", "Gender", "Signup_Quarter"], start=1):
    g = df.groupby(c)["Churned"].agg(["mean", "size"]).sort_values("mean")
    ic = 1.96 * np.sqrt(g["mean"] * (1 - g["mean"]) / g["size"])
    fig.add_bar(x=g.index, y=g["mean"] * 100, error_y=dict(type="data", array=ic * 100, color=INK2, thickness=1),
                marker_color=SERIES[0], showlegend=False, row=1, col=k,
                customdata=g["size"], hovertemplate="%{x}: %{y:.1f}% (n = %{customdata})<extra></extra>")
fig.add_hline(y=churn_rate * 100, line_dash="dot", line_color=INK2)
fig.update_yaxes(title_text="Churn (%)", range=[0, 40], row=1, col=1)
fig.update_layout(title="Taxa de churn por categoria (barra = IC 95%; linha = média geral)", height=420)
fig.show()

# %% [markdown]
# ---
# ## 3. Correlações entre pares
#
# ### 3.1 Pearson × Spearman: o peso dos extremos
# Pearson mede relação **linear** e é sensível a valores extremos; Spearman usa a **ordem** dos valores e resiste a eles. Quando os dois divergem muito, alguns pontos extremos estão criando (ou escondendo) a relação. Comparamos as duas medidas nos 190 pares, na base bruta.

# %%
def pares(M, metodo):
    C = M.corr(method=metodo)
    t = C.where(np.triu(np.ones(C.shape), 1).astype(bool)).stack().dropna()
    return t

pb = pares(bruto[NUM], "pearson"); sb = pares(bruto[NUM], "spearman")
comp = pd.DataFrame({"pearson": pb, "spearman": sb}).dropna()
comp["dif"] = comp["pearson"] - comp["spearman"]
comp["par"] = [f"{ROT[a]} × {ROT[b]}" for a, b in comp.index]

fig = go.Figure()
fig.add_scatter(x=comp["spearman"], y=comp["pearson"], mode="markers", text=comp["par"],
                marker=dict(size=8, color=SERIES[0], opacity=.7, line=dict(width=1, color=SURFACE)),
                hovertemplate="%{text}<br>Spearman %{x:.2f} | Pearson %{y:.2f}<extra></extra>", showlegend=False)
fig.add_scatter(x=[-1, 1], y=[-1, 1], mode="lines", line=dict(color=AXIS, dash="dot"), hoverinfo="skip", showlegend=False)
for (a, b), r in comp.reindex(comp["dif"].abs().sort_values(ascending=False).index).head(3).iterrows():
    fig.add_annotation(x=r["spearman"], y=r["pearson"], text=r["par"], showarrow=True, arrowhead=0, ax=40, ay=-30,
                       font=dict(size=11, color=INK2))
fig.update_layout(title="Cada ponto é um par de variáveis: fora da diagonal, os extremos distorcem Pearson",
                  xaxis_title="Correlação de Spearman (ordem)", yaxis_title="Correlação de Pearson (linear)", height=560,
                  xaxis=dict(range=[-.8, .9]), yaxis=dict(range=[-.8, .9]))
fig.show()

# %% [markdown]
# **Dois exemplos.** Ticket médio × LTV parece quase sem relação em Pearson, mas tem relação moderada em Spearman: 25 tickets acima de US\$ 1.000 achatam o coeficiente linear. Já total de compras × taxa de devolução aparece correlacionado em Pearson só por causa de poucos extremos; pela ordem, a relação some.

# %%
ex = [("Average_Order_Value", "Lifetime_Value"), ("Total_Purchases", "Returns_Rate")]
fig = make_subplots(rows=1, cols=2, subplot_titles=[
    f"{ROT[a]} × {ROT[b]}<br><sup>Pearson {comp.loc[(a, b), 'pearson'] if (a, b) in comp.index else comp.loc[(b, a), 'pearson']:.2f} · "
    f"Spearman {comp.loc[(a, b), 'spearman'] if (a, b) in comp.index else comp.loc[(b, a), 'spearman']:.2f}</sup>" for a, b in ex])
amostra_pts = bruto.sample(12000, random_state=1)
extremos = bruto[(bruto["Average_Order_Value"] > 1000) | (bruto["Total_Purchases"] > 60) | (bruto["Total_Purchases"] < 0) | (bruto["Returns_Rate"] > 50)]
pts = pd.concat([amostra_pts, extremos]).drop_duplicates()
for k, (a, b) in enumerate(ex, start=1):
    fig.add_scattergl(x=pts[a], y=pts[b], mode="markers", marker=dict(size=4, color=SERIES[0], opacity=.25),
                      showlegend=False, row=1, col=k, hovertemplate="%{x:.1f} / %{y:.1f}<extra></extra>")
    fig.update_xaxes(title_text=ROT[a], type="log" if a == "Average_Order_Value" else "linear", row=1, col=k)
    fig.update_yaxes(title_text=ROT[b], row=1, col=k)
fig.update_layout(title="Os extremos em ação (12 mil clientes sorteados + todos os extremos; ticket em escala log)", height=460)
fig.show()

# %% [markdown]
# ### 3.2 Tamanho do efeito e um padrão "tudo ou nada"
# Com 50 mil clientes, uma correlação de apenas 0,01 já seria estatisticamente significativa. O critério que importa, então, é o **tamanho** da correlação. A distribuição por faixa revela outro traço da base: os pares ou são praticamente independentes, ou fortemente ligados, e quase não há meio-termo. Em dados reais, correlações fracas (0,1 a 0,3) costumam ser as mais comuns. Daqui em diante usamos Spearman na base limpa.

# %%
def corr_pares(M, metodo="spearman"):
    cols = M.columns
    out = []
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            x = M[[cols[i], cols[j]]].dropna()
            r, p = (stats.spearmanr if metodo == "spearman" else stats.pearsonr)(x.iloc[:, 0], x.iloc[:, 1])
            out.append((cols[i], cols[j], r, len(x), p))
    d = pd.DataFrame(out, columns=["a", "b", "r", "n", "p"])
    d["p_fdr"] = multipletests(d["p"], method="fdr_bh")[1]
    d["rotulo_a"], d["rotulo_b"] = d["a"].map(ROT), d["b"].map(ROT)
    d["bloco_a"], d["bloco_b"] = d["a"].map(BLOCO), d["b"].map(BLOCO)
    d["mesmo_bloco"] = d["bloco_a"] == d["bloco_b"]
    return d

PARES = corr_pares(df[NUM])
faixas = pd.cut(PARES["r"].abs(), [0, .1, .3, .5, 1], labels=["desprezível (< 0,1)", "fraca (0,1–0,3)", "moderada (0,3–0,5)", "forte (≥ 0,5)"], include_lowest=True)
resumo_sig = pd.DataFrame({"pares": faixas.value_counts().sort_index(),
                           "significativos (FDR 5%)": PARES.groupby(faixas)["p_fdr"].apply(lambda s: (s < .05).sum())})
md(f"**{(PARES['r'].abs() < .1).sum()} de {len(PARES)} pares** têm |ρ| < 0,1 (e quase nenhum é significativo); **{(PARES['r'].abs() >= .3).sum()}** têm |ρ| ≥ 0,3; "
   f"só **{((PARES['r'].abs() >= .1) & (PARES['r'].abs() < .3)).sum()}** ficam na faixa fraca. Significativos após Benjamini-Hochberg: {(PARES['p_fdr'] < .05).sum()}.")
resumo_sig

# %% [markdown]
# ### 3.3 Matriz clusterizada
# Variáveis reordenadas por agrupamento hierárquico (distância = 1 − |ρ|). O bloco grande em azul é o **engajamento**: quem usa mais o site também usa mais o app, abre mais e-mails e compra mais. O atrito (abandono de carrinho, ligações ao atendimento) aparece em vermelho contra esse bloco.

# %%
C = df[NUM].corr(method="spearman")
D = (1 - C.abs()).to_numpy(copy=True); np.fill_diagonal(D, 0)
ordem = C.index[leaves_list(linkage(squareform(D, checks=False), "average"))]
Cc = C.loc[ordem, ordem]
fig = go.Figure(go.Heatmap(z=Cc.values, x=[ROT[v] for v in ordem], y=[ROT[v] for v in ordem], colorscale=DIVERGING,
                           zmin=-1, zmax=1, colorbar=dict(title="ρ"), text=np.round(Cc.values, 2),
                           texttemplate="%{text:.2f}", textfont=dict(size=9),
                           hovertemplate="%{y}<br>%{x}<br>ρ = %{z:.2f}<extra></extra>"))
fig.update_layout(title="Correlação de Spearman entre as 20 variáveis numéricas (ordenadas por cluster)", height=820, width=1000,
                  xaxis=dict(tickangle=-50, tickfont=dict(size=10)), yaxis=dict(autorange="reversed", tickfont=dict(size=10)))
fig.show()

# %% [markdown]
# ### 3.4 Ranking dos pares

# %%
def mostrar_pares(x):
    return (x[["rotulo_a", "rotulo_b", "bloco_a", "bloco_b", "r", "n", "p_fdr"]]
            .rename(columns={"rotulo_a": "variável A", "rotulo_b": "variável B", "bloco_a": "bloco A", "bloco_b": "bloco B", "r": "ρ", "p_fdr": "p (FDR)"})
            .style.format({"ρ": "{:+.2f}", "p (FDR)": "{:.1e}"}, decimal=",")
            .background_gradient(subset=["ρ"], cmap="RdBu", vmin=-1, vmax=1).hide(axis="index"))

md("#### a) Os 12 pares mais fortes")
display(mostrar_pares(PARES.reindex(PARES["r"].abs().sort_values(ascending=False).index).head(12)))
md("#### b) Os 10 pares mais fortes **entre blocos diferentes**")
x = PARES[~PARES["mesmo_bloco"]]
display(mostrar_pares(x.reindex(x["r"].abs().sort_values(ascending=False).index).head(10)))

# %% [markdown]
# ### 3.5 Um fator comum? Análise de componentes principais
# Se o bloco de engajamento for de fato uma coisa só, o primeiro componente principal deve concentrar boa parte da variância.

# %%
Zn = df[NUM].fillna(df[NUM].median())
Zn = (Zn - Zn.mean()) / Zn.std()
U, Sv, Vt = np.linalg.svd(Zn.values, full_matrices=False)
expl = Sv ** 2 / (Sv ** 2).sum()
ld = pd.Series(Vt[0], index=NUM)
if ld.sum() < 0:
    ld, U = -ld, -U
PC1 = pd.Series(U[:, 0] * Sv[0], index=df.index, name="fator_engajamento")
md(f"* O **1º componente explica {expl[0]:.1%}** da variância das 20 variáveis; o 2º, {expl[1]:.1%}. Com variáveis independentes, cada um explicaria cerca de 5%.\n"
   f"* O 1º componente é um **fator de engajamento**: pesos positivos em sessão, páginas, app, logins e compras; negativos em abandono de carrinho e ligações ao atendimento.\n"
   f"* Correlação do fator com o churn: **ρ = {stats.spearmanr(PC1, df['Churned'])[0]:+.2f}**.")
ldo = ld.sort_values()
fig = go.Figure(go.Bar(x=ldo.values, y=[ROT[v] for v in ldo.index], orientation="h", showlegend=False,
                       marker_color=[COR_BLOCO[BLOCO[v]] for v in ldo.index],
                       hovertemplate="%{y}: %{x:.2f}<extra></extra>"))
for b in BLOCOS:
    fig.add_scatter(x=[None], y=[None], mode="markers", marker=dict(size=10, color=COR_BLOCO[b], symbol="square"), name=b)
fig.update_layout(title=f"Pesos das variáveis no 1º componente principal ({expl[0]:.0%} da variância)", height=600,
                  xaxis_title="Peso (loading)", legend=dict(orientation="h", y=1.04, yanchor="bottom", x=0), margin=dict(t=110))
fig.show()

# %% [markdown]
# ### 3.6 Correlação com o churn: bruta e parcial
# A correlação **bruta** mostra quanto cada variável anda junto com o cancelamento. A **parcial** mostra o que sobra depois de descontar o fator de engajamento: se uma variável perde quase toda a correlação, ela era só mais uma medida do engajamento geral.

# %%
def parcial(x, y, z):
    """Correlação de Spearman parcial de x e y controlando z (via resíduos dos postos)."""
    d = pd.concat([x, y, z], axis=1).dropna().rank()
    rx = sm.OLS(d.iloc[:, 0], sm.add_constant(d.iloc[:, 2])).fit().resid
    ry = sm.OLS(d.iloc[:, 1], sm.add_constant(d.iloc[:, 2])).fit().resid
    return np.corrcoef(rx, ry)[0, 1]

CH = pd.DataFrame({"bruta": [stats.spearmanr(df[v], df["Churned"], nan_policy="omit")[0] for v in NUM],
                   "parcial": [parcial(df[v], df["Churned"], PC1) for v in NUM]}, index=NUM)
CH["nome"] = CH.index.map(ROT); CH["bloco"] = CH.index.map(BLOCO)
CH = CH.reindex(CH["bruta"].abs().sort_values().index)
fig = go.Figure()
for i, (v, r) in enumerate(CH.iterrows()):
    fig.add_scatter(x=[r["bruta"], r["parcial"]], y=[r["nome"]] * 2, mode="lines", line=dict(color=AXIS, width=2),
                    showlegend=False, hoverinfo="skip")
fig.add_scatter(x=CH["bruta"], y=CH["nome"], mode="markers", name="Bruta", marker=dict(size=11, color=SERIES[0]),
                hovertemplate="%{y}<br>bruta: %{x:+.2f}<extra></extra>")
fig.add_scatter(x=CH["parcial"], y=CH["nome"], mode="markers", name="Parcial (sem o fator de engajamento)",
                marker=dict(size=11, color=SERIES[1], symbol="diamond"), hovertemplate="%{y}<br>parcial: %{x:+.2f}<extra></extra>")
fig.add_vline(x=0, line_color=INK2, line_width=1)
fig.update_layout(title="Correlação de Spearman com o churn: bruta × descontado o engajamento geral", xaxis_title="ρ com churn",
                  height=640, legend=dict(orientation="h", y=1.03, yanchor="bottom", x=0), margin=dict(t=110))
fig.show()
eng = [v for v in NUM if BLOCO[v] == "Engajamento"]
md(f"* As 8 variáveis de engajamento têm correlação bruta com o churn entre {CH.loc[eng, 'bruta'].min():+.2f} e {CH.loc[eng, 'bruta'].max():+.2f}, "
   f"mas a parcial cai para perto de zero ({CH.loc[eng, 'parcial'].min():+.2f} a {CH.loc[eng, 'parcial'].max():+.2f}): **são oito medidas da mesma coisa**.\n"
   f"* Sobrevivem ao controle: ligações ao atendimento ({CH.loc['Customer_Service_Calls', 'parcial']:+.2f}), ticket médio ({CH.loc['Average_Order_Value', 'parcial']:+.2f}), "
   f"dias desde a última compra ({CH.loc['Days_Since_Last_Purchase', 'parcial']:+.2f}) e idade ({CH.loc['Age', 'parcial']:+.2f}).\n"
   f"* O LTV muda de sinal: {CH.loc['Lifetime_Value', 'bruta']:+.2f} bruto, {CH.loc['Lifetime_Value', 'parcial']:+.2f} parcial. A Seção 4 mostra por quê.")

# %% [markdown]
# ---
# ## 4. Fatores de churn
#
# ### 4.1 Churn por faixa de cada variável
# Cada linha divide os clientes em cinco faixas (quintis) da variável e mostra a taxa de churn em cada faixa. Isso revela **limiares** que uma correlação, por ser um número só, esconde.

# %%
def churn_por_faixa(v, q=5):
    s = df[[v, "Churned"]].dropna()
    f = pd.qcut(s[v].rank(method="first"), q, labels=False)
    g = s.groupby(f).agg(meio=(v, "median"), churn=("Churned", "mean"), n=("Churned", "size"))
    return g

top_vars = CH["bruta"].abs().sort_values(ascending=False).index[:12].tolist()
Q = pd.DataFrame({ROT[v]: churn_por_faixa(v)["churn"].values * 100 for v in top_vars}, index=[f"Q{i}" for i in range(1, 6)]).T
fig = px.imshow(Q, text_auto=".0f", color_continuous_scale=SEQ, aspect="auto", labels=dict(color="Churn (%)"))
fig.update_traces(hovertemplate="%{y}<br>%{x}: %{z:.1f}% de churn<extra></extra>")
fig.update_layout(title="Taxa de churn por quintil (Q1 = 20% com menor valor; Q5 = 20% com maior valor)", height=560,
                  xaxis_title="Quintil da variável")
fig.show()
q_ltv = churn_por_faixa("Lifetime_Value")["churn"]
md(f"O **LTV tem relação em U** com o churn: {q_ltv.iloc[0]:.0%} no quintil mais baixo, {q_ltv.min():.0%} no meio e {q_ltv.iloc[-1]:.0%} no mais alto. "
   "Uma correlação, que resume tudo numa reta, enxerga quase nada (ρ ≈ −0,06); é por isso que o LTV parece irrelevante na Seção 3 e vira a variável mais importante no modelo de árvores.")

# %% [markdown]
# ### 4.2 Regressão logística
# Modelo com todas as variáveis padronizadas (e indicadores de dado faltante). O coeficiente indica quanto o log-odds de churn muda para um desvio-padrão a mais da variável, **com as outras constantes**. A comparação com a correlação bruta revela um **efeito de supressão**: uma variável que muda de sinal quando outra entra no modelo.

# %%
X = df[NUM].copy()
for v in NUM:
    if X[v].isna().any():
        X[f"falta_{v}"] = X[v].isna().astype(int)
        X[v] = X[v].fillna(X[v].median())
Xs = X.copy()
Xs[NUM] = (X[NUM] - X[NUM].mean()) / X[NUM].std()
y = df["Churned"]
logit = sm.Logit(y, sm.add_constant(Xs)).fit(disp=0)
coefs = pd.DataFrame({"coef": logit.params[NUM], "lo": logit.conf_int().loc[NUM, 0], "hi": logit.conf_int().loc[NUM, 1],
                      "p": logit.pvalues[NUM]})
coefs["odds_ratio"] = np.exp(coefs["coef"])
coefs["nome"] = coefs.index.map(ROT); coefs["bruta"] = CH.loc[coefs.index, "bruta"]
coefs = coefs.sort_values("coef")
vif = pd.Series([variance_inflation_factor(Xs[NUM].values, i) for i in range(len(NUM))], index=NUM)

fig = go.Figure(go.Scatter(x=coefs["coef"], y=coefs["nome"], mode="markers",
                           error_x=dict(type="data", symmetric=False, array=coefs["hi"] - coefs["coef"], arrayminus=coefs["coef"] - coefs["lo"], color=INK2),
                           marker=dict(size=10, color=[COR_BLOCO[BLOCO[v]] for v in coefs.index]),
                           customdata=np.c_[coefs["odds_ratio"], coefs["bruta"]],
                           hovertemplate="%{y}<br>coef %{x:+.2f} (OR %{customdata[0]:.2f})<br>ρ bruta com churn %{customdata[1]:+.2f}<extra></extra>"))
fig.add_vline(x=0, line_color=INK2, line_width=1)
fig.update_layout(title="Regressão logística: efeito de +1 desvio-padrão sobre o log-odds de churn (IC 95%)", height=620,
                  xaxis_title="Coeficiente padronizado")
fig.show()

auc_lr = roc_auc_score(y, logit.predict(sm.add_constant(Xs)))
inv = coefs[np.sign(coefs["coef"]) != np.sign(coefs["bruta"])]
inv = inv[(inv["p"] < .05) & (inv["bruta"].abs() > .03)]
md(f"* AUC no próprio ajuste: **{auc_lr:.3f}**; pseudo-R² de McFadden: {logit.prsquared:.3f}.\n"
   "* Maiores efeitos: " + ", ".join(f"{ROT[v]} ({coefs.loc[v, 'coef']:+.2f})" for v in coefs["coef"].abs().sort_values(ascending=False).index[:4]) + ".\n"
   + (f"* **Sinal invertido:** {', '.join(f'{r.nome} (ρ bruta {r.bruta:+.2f}, coef. {r.coef:+.2f})' for r in inv.itertuples())}. "
      f"Sozinho, um LTV maior acompanha levemente menos churn; com o total de compras e o engajamento no modelo, o LTV passa a medir o *valor gasto por compra*, e aí o efeito se inverte. "
      f"Não é multicolinearidade grave (VIF do LTV = {vif['Lifetime_Value']:.1f}), e sim supressão: o coeficiente responde a outra pergunta que a correlação." if len(inv) else ""))

# %% [markdown]
# ### 4.3 Modelo de árvores e importância das variáveis
# A logística supõe efeitos lineares no log-odds. Um modelo de *gradient boosting* capta limiares e interações. Comparamos os dois com validação cruzada estratificada de 5 partes.

# %%
cv = StratifiedKFold(5, shuffle=True, random_state=42)
gb = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.08, random_state=42)
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
lr_sk = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000))
auc_cv_lr = cross_val_score(lr_sk, X, y, cv=cv, scoring="roc_auc")
auc_cv_gb = cross_val_score(gb, df[NUM], y, cv=cv, scoring="roc_auc")
md(f"AUC em validação cruzada: **regressão logística {auc_cv_lr.mean():.3f}** (± {auc_cv_lr.std():.3f}) × "
   f"**gradient boosting {auc_cv_gb.mean():.3f}** (± {auc_cv_gb.std():.3f}). "
   f"A diferença de {auc_cv_gb.mean() - auc_cv_lr.mean():.2f} indica que boa parte da relação com o churn **não é linear**.")

Xtr, Xte, ytr, yte = train_test_split(df[NUM], y, test_size=0.25, stratify=y, random_state=42)
gb.fit(Xtr, ytr)
pi = permutation_importance(gb, Xte, yte, scoring="roc_auc", n_repeats=5, random_state=42)
IMP = pd.DataFrame({"queda_auc": pi.importances_mean, "dp": pi.importances_std}, index=NUM).sort_values("queda_auc")
fig = go.Figure(go.Bar(x=IMP["queda_auc"], y=[ROT[v] for v in IMP.index], orientation="h",
                       error_x=dict(type="data", array=IMP["dp"], color=INK2),
                       marker_color=[COR_BLOCO[BLOCO[v]] for v in IMP.index],
                       hovertemplate="%{y}: queda de %{x:.3f} no AUC<extra></extra>"))
fig.update_layout(title=f"Importância por permutação no gradient boosting (AUC no teste = {roc_auc_score(yte, gb.predict_proba(Xte)[:, 1]):.3f})",
                  xaxis_title="Queda no AUC ao embaralhar a variável", height=620)
fig.show()

# %% [markdown]
# ### 4.4 Onde estão os limiares
# Dependência parcial: a probabilidade média prevista de churn ao variar uma variável, mantendo as demais como estão. Mostramos as quatro mais importantes.

# %%
top4 = IMP["queda_auc"].sort_values(ascending=False).index[:4].tolist()
fig = make_subplots(rows=2, cols=2, subplot_titles=[ROT[v] for v in top4], vertical_spacing=0.16, horizontal_spacing=0.1)
amostra = Xte.sample(3000, random_state=0)

def dependencia_parcial(modelo, X0, v, n=30):
    """Probabilidade média prevista ao fixar a variável v em cada ponto da grade (demais variáveis como estão)."""
    grade = np.unique(np.nanquantile(df[v], np.linspace(0.01, 0.99, n)))
    out = []
    for g_ in grade:
        Xg = X0.copy(); Xg[v] = g_
        out.append(modelo.predict_proba(Xg)[:, 1].mean())
    return grade, np.array(out)

PDP = {}
for k, v in enumerate(top4):
    grid, media = dependencia_parcial(gb, amostra, v)
    PDP[v] = (grid, media)
    fig.add_scatter(x=grid, y=media * 100, mode="lines+markers", line=dict(color=COR_BLOCO[BLOCO[v]], width=2.5), marker=dict(size=5),
                    showlegend=False, row=k // 2 + 1, col=k % 2 + 1,
                    hovertemplate=f"{ROT[v]} = %{{x:.1f}}<br>churn previsto %{{y:.1f}}%<extra></extra>")
    fig.add_hline(y=churn_rate * 100, line_dash="dot", line_color=AXIS, row=k // 2 + 1, col=k % 2 + 1)
    fig.update_yaxes(title_text="Churn previsto (%)" if k % 2 == 0 else None, row=k // 2 + 1, col=k % 2 + 1)
fig.update_layout(title="Dependência parcial: churn previsto ao longo de cada variável (linha pontilhada = média geral)", height=620)
fig.show()
DEGRAUS = {}
for v, (grid, media) in PDP.items():
    j = int(np.argmax(np.abs(np.diff(media))))
    DEGRAUS[v] = (grid[j], grid[j + 1], media[j], media[j + 1])
md("Os efeitos são **degraus**, não curvas suaves:\n" + "\n".join(
    f"* {ROT[v]}: churn previsto passa de {d[2]:.0%} para {d[3]:.0%} entre {d[0]:.0f} e {d[1]:.0f}." for v, d in DEGRAUS.items()) +
   "\n\nDegraus tão nítidos sugerem que o gerador da base usou **regras de corte** para definir o churn. É exatamente o que a regressão logística, "
   "linear no log-odds, não consegue representar, e explica a vantagem do modelo de árvores. Note também que o LTV, condicionado às demais variáveis, "
   "tem um único degrau, enquanto no mapa de churn por quintil (sem condicionar) aparece em U: os clientes de LTV baixo cancelam mais porque também engajam menos.")

# %% [markdown]
# ---
# ## 5. Gráfico interativo sem servidor: churn por faixa de qualquer variável
# Funciona também na versão estática (nbviewer/HTML). Use o menu para trocar a variável.

# %%
ordem_menu = sorted(NUM, key=lambda v: (BLOCOS.index(BLOCO[v]), ROT[v]))
k0 = ordem_menu.index("Cart_Abandonment_Rate")
fig = go.Figure()
for k, v in enumerate(ordem_menu):
    g = churn_por_faixa(v, 10)
    ic = 1.96 * np.sqrt(g["churn"] * (1 - g["churn"]) / g["n"]) * 100
    fig.add_scatter(x=g["meio"], y=g["churn"] * 100, mode="lines+markers", visible=(k == k0), name=ROT[v],
                    error_y=dict(type="data", array=ic, color=AXIS), line=dict(color=COR_BLOCO[BLOCO[v]], width=2.5),
                    marker=dict(size=8), customdata=g["n"],
                    hovertemplate="mediana da faixa %{x:.1f}<br>churn %{y:.1f}% (n = %{customdata})<extra></extra>")
botoes = []
for k, v in enumerate(ordem_menu):
    vis = [False] * len(ordem_menu); vis[k] = True
    botoes.append(dict(label=f"{ROT[v]} · {BLOCO[v]}", method="update",
                       args=[{"visible": vis}, {"title.text": f"Churn por decil de {ROT[v]}", "xaxis.title.text": ROT[v]}]))
fig.add_hline(y=churn_rate * 100, line_dash="dot", line_color=INK2, annotation_text="média geral",
              annotation_font=dict(color=INK2, size=11))
fig.update_layout(updatemenus=[dict(buttons=botoes, active=k0, direction="down", x=0, xanchor="left", y=1.2, yanchor="top", bgcolor=SURFACE)],
                  title=dict(text=f"Churn por decil de {ROT[ordem_menu[k0]]}", y=0.9), xaxis_title=ROT[ordem_menu[k0]],
                  yaxis_title="Churn (%)", yaxis=dict(range=[0, 80]), height=520, margin=dict(t=140), showlegend=False)
fig.show()

# %% [markdown]
# ---
# ## 6. Painel interativo (precisa do notebook em execução)
#
# ### 6.1 Comparador de pares
# Escolha duas variáveis. O painel mostra a densidade dos clientes (mapa de calor) e a taxa de churn em cada combinação de faixas, com Pearson e Spearman antes e depois da limpeza.

# %%
opcoes = [(f"{ROT[v]} · {BLOCO[v]}", v) for v in ordem_menu]

def comparar(a, b, base):
    if a == b:
        print("Escolha duas variáveis diferentes."); return
    D_ = bruto if base == "Bruta" else df
    s = D_[[a, b, "Churned"]].dropna()
    rp, rs = s[a].corr(s[b]), s[a].corr(s[b], method="spearman")
    fa = pd.qcut(s[a].rank(method="first"), 8, labels=False); fb = pd.qcut(s[b].rank(method="first"), 8, labels=False)
    grade = s.groupby([fb, fa])["Churned"].mean().unstack() * 100
    ma = s.groupby(fa)[a].median().round(1); mb = s.groupby(fb)[b].median().round(1)
    fig = make_subplots(rows=1, cols=2, column_widths=[.5, .5], horizontal_spacing=.14,
                        subplot_titles=(f"Densidade · Pearson {rp:+.2f} | Spearman {rs:+.2f}", "Churn (%) por faixa (octis) de A e B"))
    lo, hi = s[a].quantile([.005, .995]); lo2, hi2 = s[b].quantile([.005, .995])
    fig.add_trace(go.Histogram2d(x=s[a].clip(lo, hi), y=s[b].clip(lo2, hi2), nbinsx=40, nbinsy=40, colorscale=SEQ, showscale=False,
                                 hovertemplate="A %{x}<br>B %{y}<br>%{z} clientes<extra></extra>"), row=1, col=1)
    fig.add_trace(go.Heatmap(z=grade.values, x=[f"{v:g}" for v in ma], y=[f"{v:g}" for v in mb], colorscale=SEQ, zmin=0, zmax=80,
                             colorbar=dict(title="churn %", x=1.02), hovertemplate="A ≈ %{x}<br>B ≈ %{y}<br>churn %{z:.0f}%<extra></extra>"), row=1, col=2)
    fig.update_xaxes(title_text=ROT[a]); fig.update_yaxes(title_text=ROT[b])
    fig.update_layout(title=f"{ROT[a]} × {ROT[b]} (base {base.lower()})", height=470)
    fig.show()
    print(f"n = {len(s):,} | Pearson {rp:+.3f} | Spearman {rs:+.3f} | diferença {rp - rs:+.3f}".replace(",", "."))

w_a = widgets.Dropdown(options=opcoes, value="Customer_Service_Calls", description="Variável A", layout=widgets.Layout(width="440px"))
w_b = widgets.Dropdown(options=opcoes, value="Cart_Abandonment_Rate", description="Variável B", layout=widgets.Layout(width="440px"))
w_base = widgets.ToggleButtons(options=["Limpa", "Bruta"], value="Limpa", description="Base")
if HEADLESS:
    comparar(w_a.value, w_b.value, w_base.value)
else:
    display(widgets.VBox([widgets.HBox([w_a, w_b]), w_base]),
            widgets.interactive_output(comparar, dict(a=w_a, b=w_b, base=w_base)))

# %% [markdown]
# ### 6.2 Segmentação: churn por faixa dentro de um grupo
# Filtre por país e gênero e veja como a taxa de churn muda ao longo de uma variável.

# %%
def segmento(v, pais, genero):
    s = df.copy()
    if pais != "Todos": s = s[s["Country"] == pais]
    if genero != "Todos": s = s[s["Gender"] == genero]
    s = s[[v, "Churned"]].dropna()
    f = pd.qcut(s[v].rank(method="first"), 10, labels=False)
    g = s.groupby(f).agg(meio=(v, "median"), churn=("Churned", "mean"), n=("Churned", "size"))
    geral = churn_por_faixa(v, 10)
    fig = go.Figure()
    fig.add_scatter(x=geral["meio"], y=geral["churn"] * 100, name="Base inteira", line=dict(color=AXIS, width=2, dash="dot"),
                    hovertemplate="%{x:.1f}: %{y:.1f}%<extra>Base inteira</extra>")
    fig.add_scatter(x=g["meio"], y=g["churn"] * 100, name=f"{pais} · {genero}", mode="lines+markers",
                    line=dict(color=SERIES[0], width=2.5), customdata=g["n"],
                    hovertemplate="%{x:.1f}: %{y:.1f}% (n = %{customdata})<extra></extra>")
    fig.update_layout(title=f"Churn por decil de {ROT[v]} · {pais} · {genero} (n = {len(s):,})".replace(",", "."),
                      xaxis_title=ROT[v], yaxis_title="Churn (%)", yaxis=dict(range=[0, 85]), height=440,
                      legend=dict(orientation="h", y=1.02, yanchor="bottom", x=0))
    fig.show()

if HEADLESS:
    segmento("Customer_Service_Calls", "Todos", "Todos")
else:
  display(widgets.interactive(
    segmento,
    v=widgets.Dropdown(options=opcoes, value="Customer_Service_Calls", description="Variável", layout=widgets.Layout(width="440px")),
    pais=widgets.Dropdown(options=["Todos"] + sorted(df["Country"].unique()), value="Todos", description="País"),
    genero=widgets.Dropdown(options=["Todos"] + sorted(df["Gender"].unique()), value="Todos", description="Gênero")))

# %% [markdown]
# ### 6.3 Simulador de risco de churn
# Ajuste as cinco variáveis mais importantes do modelo de árvores; as demais ficam na mediana da base. O valor é a probabilidade de churn prevista para esse cliente-tipo.

# %%
gb_full = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.08, random_state=42).fit(df[NUM], y)
top5 = IMP["queda_auc"].sort_values(ascending=False).index[:5].tolist()
base_cliente = df[NUM].median()

def sliders():
    ws = {}
    for v in top5:
        lo, hi = df[v].quantile([.01, .99])
        passo = 1.0 if df[v].dropna().round().eq(df[v].dropna()).mean() > .9 else round((hi - lo) / 100, 2) or 0.1
        ws[v] = widgets.FloatSlider(value=float(base_cliente[v]), min=float(lo), max=float(hi), step=float(passo),
                                    description=ROT[v][:28], style={"description_width": "220px"}, layout=widgets.Layout(width="560px"))
    return ws

WS = sliders()

def simular(**vals):
    cli = base_cliente.copy()
    for v, val in vals.items():
        cli[v] = val
    p = gb_full.predict_proba(cli.to_frame().T[NUM])[0, 1]
    cor = "#e34948" if p >= .5 else ("#eda100" if p >= churn_rate else "#1baf7a")
    rot = "alto" if p >= .5 else ("acima da média" if p >= churn_rate else "abaixo da média")
    fig = go.Figure(go.Bar(x=[p * 100], y=["Risco"], orientation="h", marker_color=cor, width=.5,
                           text=[f"{p:.0%} · {rot}"], textposition="outside", hoverinfo="skip"))
    fig.add_vline(x=churn_rate * 100, line_dash="dot", line_color=INK2, annotation_text=f"média {churn_rate:.0%}",
                  annotation_font=dict(color=INK2, size=11))
    fig.update_layout(title="Probabilidade prevista de churn", xaxis=dict(range=[0, 110], title="%"), height=220,
                      yaxis=dict(visible=False), margin=dict(t=60, b=40))
    fig.show()

if HEADLESS:
    simular(**{v: w.value for v, w in WS.items()})
else:
    display(widgets.VBox(list(WS.values())), widgets.interactive_output(simular, WS))

# %% [markdown]
# ---
# ## 7. Síntese dos achados

# %%
forte = PARES.reindex(PARES["r"].abs().sort_values(ascending=False).index).iloc[0]
q_cart = churn_por_faixa("Cart_Abandonment_Rate")["churn"]
q_call = churn_por_faixa("Customer_Service_Calls")["churn"]
md(f"""
**Base**
1. A base é simulada: faltantes em proporções redondas, 5 cidades por país, cadastro uniforme por trimestre e {int(np.logical_or.reduce(list(regras.values())).sum())} clientes com valores impossíveis ou extremos plantados. As conclusões valem para a base e o método.
2. A falta de dado é informativa em duração da sessão e abertura de e-mails: quem não tem o dado cancela mais. Excluir essas linhas enviesaria a análise.

**Correlações**

3. Extremos distorcem Pearson: ticket médio × LTV vai de {comp.loc[('Average_Order_Value', 'Lifetime_Value'), 'pearson']:.2f} (Pearson) a {comp.loc[('Average_Order_Value', 'Lifetime_Value'), 'spearman']:.2f} (Spearman). As correlações são "tudo ou nada": {(PARES['r'].abs() < .1).sum()} pares com |ρ| < 0,1, {(PARES['r'].abs() >= .3).sum()} com |ρ| ≥ 0,3 e só {((PARES['r'].abs() >= .1) & (PARES['r'].abs() < .3)).sum()} no meio, outro traço de base gerada.
4. Um único fator de engajamento explica {expl[0]:.0%} da variância das 20 variáveis. O par mais forte é {forte['rotulo_a']} × {forte['rotulo_b']} (ρ = {forte['r']:+.2f}). Controlado esse fator, as 8 medidas de engajamento perdem quase toda a correlação com o churn: são redundantes entre si.
5. O perfil demográfico (país, gênero, trimestre) praticamente não separa quem cancela (V de Cramér < 0,02).

**Churn**

6. Os sinais mais fortes são de atrito: no quintil mais alto de abandono de carrinho o churn é {q_cart.iloc[-1]:.0%}, contra {q_cart.iloc[0]:.0%} no mais baixo; em ligações ao atendimento, {q_call.iloc[-1]:.0%} contra {q_call.iloc[0]:.0%}.
7. O gradient boosting (AUC {auc_cv_gb.mean():.2f}) supera a regressão logística (AUC {auc_cv_lr.mean():.2f}): os efeitos têm limiares e curvas. O caso mais claro é o LTV, com churn em U ({q_ltv.iloc[0]:.0%} no quintil baixo, {q_ltv.min():.0%} no meio, {q_ltv.iloc[-1]:.0%} no alto), invisível para a correlação e a variável mais importante do modelo de árvores.
8. A dependência parcial mostra degraus nítidos (ligações: {DEGRAUS['Customer_Service_Calls'][2]:.0%} → {DEGRAUS['Customer_Service_Calls'][3]:.0%} entre {DEGRAUS['Customer_Service_Calls'][0]:g} e {DEGRAUS['Customer_Service_Calls'][1]:g} ligações), compatíveis com regras de corte usadas pelo gerador da base.
9. Na logística, o LTV inverte de sinal ao entrar junto com o total de compras (efeito de supressão): um coeficiente de regressão não é uma correlação.

**Limitações:** base simulada; corte transversal sem datas (não há como estudar a ordem temporal entre engajamento e cancelamento); associações não implicam causalidade.
""")

# %% [markdown]
# ## Referências
# * Kaggle. Base `eCommerce-dataset.csv` (comportamento e churn de clientes de e-commerce). Disponível em: https://www.kaggle.com
# * Benjamini, Y.; Hochberg, Y. (1995). Controlling the false discovery rate. *Journal of the Royal Statistical Society B*, 57(1), 289–300.
# * Cramér, H. (1946). *Mathematical Methods of Statistics*. Princeton University Press.
# * Friedman, J. H. (2001). Greedy function approximation: a gradient boosting machine. *Annals of Statistics*, 29(5), 1189–1232.
# * Little, R. J. A.; Rubin, D. B. (2019). *Statistical Analysis with Missing Data*. 3. ed. Wiley.
# * Spearman, C. (1904). The proof and measurement of association between two things. *American Journal of Psychology*, 15(1), 72–101.
