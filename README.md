# Predição de Readmissão Hospitalar em Pacientes Diabéticos

Predição de readmissão em até 30 dias usando o dataset Diabetes 130-US hospitals (UCI ML Repository, ~101k internações entre 1999-2008).

## Problema clínico

Readmissão hospitalar precoce (<30 dias) em pacientes diabéticos é um indicador de qualidade assistencial associado a maior morbimortalidade e custo. Identificar pacientes de alto risco no momento da alta permite intervenções direcionadas: ajuste de medicação, follow-up ambulatorial precoce e educação em autocuidado.

## Dataset

- Fonte: [UCI ML Repository – Diabetes 130-US hospitals](https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008)
- 101.766 internações, 50 variáveis
- Inclui: demografia, tipo de admissão, diagnósticos (ICD-9), procedimentos, medicações antidiabéticas, A1C, tempo de internação
- Target: `readmitted` (`<30`, `>30`, `NO`) — neste projeto binarizado como `<30` vs. demais

Para reproduzir, baixe os arquivos do link acima e coloque em `data/raw/`.

## Estrutura

```
diabetes-readmission/
├── data/
│   ├── raw/         # dados originais (não versionados)
│   ├── interim/     # transformações intermediárias
│   └── processed/   # dados prontos para modelagem
├── notebooks/       # EDA, modelagem, interpretabilidade
│   ├── 01_eda.ipynb
│   ├── 02_cleaning_feature_engineering.ipynb
│   ├── 03_modeling.ipynb
│   └── 04_interpretability_shap.ipynb
├── src/             # funções reutilizáveis
│   ├── data.py
│   ├── features.py
│   └── models.py
└── reports/
    └── figures/     # figuras exportadas dos notebooks
```

## Resultados

_(a preencher após execução dos notebooks)_

## Como reproduzir

```bash
conda create -n diabetes-readm python=3.12 -y
conda activate diabetes-readm
pip install -r requirements.txt
python -m ipykernel install --user --name diabetes-readm --display-name "Python (diabetes-readm)"
jupyter lab
```

## Stack

- Python 3.12
- pandas, numpy (manipulação)
- scikit-learn, xgboost (modelagem)
- imbalanced-learn (tratamento de desbalanceamento)
- shap (interpretabilidade)
- matplotlib, seaborn (visualização)
