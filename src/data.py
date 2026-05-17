"""Funções de carregamento e leitura dos dados brutos.

Centraliza a leitura dos CSVs e o parsing do IDS_mapping.csv (formato
peculiar com 3 mini-tabelas separadas por linhas em branco).
"""
from pathlib import Path
import pandas as pd

# Raiz do projeto = pasta pai de src/
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"


def load_diabetic_data(path: Path | None = None) -> pd.DataFrame:
    """Carrega diabetic_data.csv tratando '?' como NaN.

    O dataset original usa a string '?' como código de missing. Sem
    `na_values='?'`, o pandas trata como categoria válida e mascara
    o problema silenciosamente.
    """
    path = path or (RAW_DIR / "diabetic_data.csv")
    return pd.read_csv(path, na_values="?")


def load_ids_mapping(path: Path | None = None) -> dict[str, dict[int, str]]:
    """Parser do IDS_mapping.csv.

    O arquivo tem 3 mini-tabelas (admission_type_id, discharge_disposition_id,
    admission_source_id) separadas por linhas em branco. Retorna um dict
    aninhado: {nome_da_coluna: {id_numerico: descricao}}.
    """
    path = path or (RAW_DIR / "IDS_mapping.csv")
    with open(path) as f:
        lines = [line.rstrip("\n") for line in f]

    mappings: dict[str, dict[int, str]] = {}
    current_col: str | None = None
    current_map: dict[int, str] = {}

    for line in lines:
        if not line.strip() or line.strip() == ",":
            if current_col and current_map:
                mappings[current_col] = current_map
            current_col = None
            current_map = {}
            continue
        key, val = line.split(",", 1)
        if current_col is None:
            current_col = key.strip()
        else:
            try:
                current_map[int(key)] = val.strip().strip('"')
            except ValueError:
                continue

    if current_col and current_map:
        mappings[current_col] = current_map

    return mappings
