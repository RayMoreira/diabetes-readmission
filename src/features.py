"""Feature engineering: agrupamento de códigos ICD-9 por sistema,
binarização do target, derivação de variáveis clínicas agregadas.

Preencher conforme o notebook 02_cleaning_feature_engineering avança.
"""
import pandas as pd


def binarize_readmission(series: pd.Series) -> pd.Series:
    """Converte o target original (3 classes) em binário.

    1 = readmissão em <30 dias (desfecho clínico de interesse)
    0 = readmissão tardia (>30 dias) ou nenhuma readmissão
    """
    return (series == "<30").astype(int)


def group_icd9(code: str) -> str:
    """Agrupa um código ICD-9 em categoria clínica (sistema/grupo).

    Stub — implementar com a tabela oficial de capítulos do ICD-9-CM.
    """
    raise NotImplementedError("A implementar no notebook 02")
