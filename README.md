# Churn em e-commerce: auditoria de dados, correlações entre fatores e modelos de cancelamento

[![Abrir no Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/joaogabrielts0611-ai/analise-churn-ecommerce/blob/main/analise_churn_ecommerce.ipynb)
[![Ver no nbviewer](https://img.shields.io/badge/ver%20com%20gr%C3%A1ficos-nbviewer-orange)](https://nbviewer.org/github/joaogabrielts0611-ai/analise-churn-ecommerce/blob/main/analise_churn_ecommerce.ipynb)

**Autor:** João Gabriel Tavares Silva · Universidade de Brasília (UnB) · Matrícula 221021071

Análise do [*Ecommerce Customer Behavior Dataset*](https://www.kaggle.com/datasets/srisyra02/ecommerce-customer-behavior-dataset) (Kaggle), com 50.000 clientes de e-commerce (25 variáveis de perfil, engajamento, compras e atrito) e a indicação de cancelamento (churn, 28,9% dos clientes).

> **A base é simulada.** O notebook documenta as evidências (faltantes em proporções redondas exatas, geografia e cadastro perfeitamente equilibrados, erros inseridos, correlações "tudo ou nada", efeitos em degrau). Os resultados valem como estudo de método.

## Como abrir

| Onde | Para quê |
|---|---|
| **Colab** (botão acima) | Rodar tudo e usar o painel interativo: `Ambiente de execução → Executar tudo` (1 a 2 minutos). Os dados são baixados deste repositório. |
| **nbviewer** (botão acima) | Ler o notebook já executado, com os gráficos interativos, sem rodar nada. |
| `relatorio_analise.html` | O notebook executado em uma página HTML única: baixe e abra no navegador. |

## O que o notebook faz

1. **Auditoria:** faltantes (e se a falta é informativa), valores impossíveis, sinais de base simulada.
2. **Limpeza documentada**, com efeito antes e depois.
3. **Correlações entre pares:** Pearson × Spearman (efeito dos extremos), tamanho do efeito, matriz clusterizada, fator comum (PCA), correlação parcial com o churn.
4. **Fatores de churn:** churn por quintil, regressão logística (e o efeito de supressão do LTV), gradient boosting, importância por permutação, dependência parcial.
5. **Painel interativo:** churn por faixa de qualquer variável, comparador de pares, segmentação por país e gênero, simulador de risco.

## Principais achados

* Extremos distorcem Pearson: ticket médio × LTV = 0,15 (Pearson) contra 0,56 (Spearman).
* Um fator de engajamento explica 36% da variância; controlado esse fator, as 8 medidas de engajamento perdem quase toda a correlação com o churn.
* Perfil demográfico não separa quem cancela (V de Cramér < 0,02).
* Atrito pesa: churn de 12% a 47% entre o menor e o maior quintil de ligações ao atendimento, com degrau entre 5 e 6 ligações.
* Gradient boosting (AUC 0,93) supera a regressão logística (AUC 0,79): os efeitos são degraus, não retas.

## Estrutura

```
analise_churn_ecommerce.ipynb   notebook principal (executado)
analise_churn_ecommerce.py      mesmo notebook em texto (jupytext)
relatorio_analise.html          versão HTML do notebook executado
data/eCommerce-dataset.csv      base original do Kaggle
requirements.txt
```

## Rodar localmente

```bash
pip install -r requirements.txt
jupyter lab analise_churn_ecommerce.ipynb
```
