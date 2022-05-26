import pytest
import wandb
import numpy as np
import pandas as pd

# Inicia projeto
run = wandb.init(project='mlreanimacao', job_type='data_checks')

@pytest.fixture(scope='session')
# Captura artefeto
def data():
    local_path = run.use_artifact('mlreanimacao/clean_data.csv:latest').file()
    df = pd.read_csv(local_path)

    return df

# Testa o tamanho do dataset(> 100 linhas
def test_data_length(data):
    assert len(data) > 100

# Testa quantidade de colunas
def test_qt_columns(data):
    assert data.shape[1] == 15

# Testa consistência das colunas do dataset:  presença e tipo
# 13 colunas categóricas e 2 numéricas
def test_consistency_dataset(data):
    colunas = {
        "idade_materna": pd.api.types.is_float_dtype,
        "fumo": pd.api.types.is_object_dtype,
        "alcool": pd.api.types.is_object_dtype,
        "psicoativas": pd.api.types.is_object_dtype,
        "tpp": pd.api.types.is_object_dtype,
        "dpp": pd.api.types.is_object_dtype,
        "oligoamnio": pd.api.types.is_object_dtype,
        "sifilis": pd.api.types.is_object_dtype,
        "hiv": pd.api.types.is_object_dtype,
        "covid_mae": pd.api.types.is_object_dtype,
        "dheg": pd.api.types.is_object_dtype,
        "dm": pd.api.types.is_object_dtype,
        "sexo": pd.api.types.is_object_dtype,
        "apgar_1_minuto": pd.api.types.is_float_dtype,
        "reanimacao": pd.api.types.is_object_dtype,
    }

    # Verifica se os nomes das colunas estão consistentes
    assert set(data.columns.values).issuperset(set(colunas.keys()))

    # Verifica consistência de nome e tipo das colunas
    for col_name, format_verif_funct in colunas.items():
        assert format_verif_funct(data[col_name]), f"{col_name} não é do tipo {format_verif_funct}!"

# Testa consistência dos valores numéricos e categóricos
def test_values_consistency(data):
    for col_name, tipo in data.items():
        if data[col_name].dtype == "float64":
            assert data[col_name].values.all() != np.nan, f"Coluna {col_name} está inconsistente!"
        else:
            if col_name == 'sexo':
                assert data[col_name].values.all() in ["Feminino", "Masculino"], \
                    f"Coluna {col_name} está inconsistente!"
            elif col_name == 'dm':
                assert data[col_name].values.all() in ["s_dm", 'n_dm', 'd_dm'], \
                    f"Coluna {col_name} está inconsistente!"
            elif col_name == 'reanimacao':
                assert data[col_name].values.all() in ["sr", 'nr'], \
                    f"Coluna {col_name} está inconsistente!"
            else:
                assert data[col_name].values.all().startswith("s") | \
                       data[col_name].values.all().startswith("n") | \
                       data[col_name].values.all().startswith("d"), \
                    f"Coluna {col_name} está inconsistente!"


