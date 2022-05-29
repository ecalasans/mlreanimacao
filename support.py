from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import pandas as pd

'''
Class FeatureSelector
'''
class FeatureSelector(BaseEstimator, TransformerMixin):
    # Construtor
    def __init__(self, feature_names):
        self.feature_names = feature_names

    # Override de fit
    def fit(self, X, y=None):
        return self

    # Retorna as colunas passadas no construtor
    def transform(self, X, y=None):
        return X[self.feature_names]



class CategoricalTransformer(BaseEstimator, TransformerMixin):
    # Construtor
    def __init__(self, new_features=True, colnames=None):
        self.new_features = new_features
        self.colnames = colnames

    # Override de fit
    def fit(self, X, y=None):
        return self

    # Override de get_feature_names_out
    def get_feature_names_out(self):
        return self.colnames.tolist()

    # Transformer method we wrote for this transformer
    def transform(self, X, y=None):
        df = pd.DataFrame(X, columns=self.colnames)

        # Remove eventuais espaços em branco dos valores - a princípio isso não existe na base de dados
        df = df.apply(lambda row: row.str.strip())

        # Opção de fazer preprocessamento se new_features for True(padrão da classe)
        if self.new_features:
            df['fumo'].fillna(2, inplace=True)
            df['alcool'].fillna(2, inplace=True)
            df['psicoativas'].fillna(2, inplace=True)
            df['tpp'].fillna(2, inplace=True)
            df['dheg'].fillna(2, inplace=True)
            df['dm'].fillna(2, inplace=True)
            df['sexo'].fillna(3, inplace=True)
            df['oligoamnio'].fillna(2, inplace=True)
            df['dpp'].fillna(2, inplace=True)
            df['sifilis'].fillna(2, inplace=True)
            df['hiv'].fillna(2, inplace=True)
            df['covid_mae'].fillna(2, inplace=True)

            # Elimina valores com sexo indefinido
            df.drop(df[df['sexo'] == 3].index, inplace=True)

            # Categoriza as features pois o artefato retorna sempre valores numéricos
            df['fumo'].replace([0, 1, 2, 3], ['n_fumo', 's_fumo', 'n_fumo', 'd_fumo'], inplace=True)
            df['alcool'].replace([0, 1, 2, 3], ['n_alcool', 's_alcool', 'n_alcool', 'd_alcool'], inplace=True)
            df['psicoativas'].replace([0, 1, 2, 3], ['n_psico', 's_psico', 'n_psico', 'd_psico'], inplace=True)
            df['tpp'].replace([0, 1, 2, 3], ['n_tpp', 's_tpp', 'n_tpp', 'd_tpp'], inplace=True)
            df['dheg'].replace([0, 1, 2, 3], ['n_dheg', 's_dheg', 'n_dheg', 'd_dheg'], inplace=True)

            # Melhor adequação da feature sexo
            df['sexo'].replace([1,2], ['Feminino', 'Masculino'], inplace=True)
            df['dpp'].replace([0, 1, 2, 3], ['n_dpp', 's_dpp', 'n_dpp', 'd_dpp'], inplace=True)
            df['oligoamnio'].replace([0, 1, 2, 3], ['n_oligo', 's_oligo', 'n_oligo', 'd_oligo'], inplace=True)
            df['sifilis'].replace([0, 1, 2, 3], ['n_sifilis', 's_sifilis', 'n_sifilis', 'd_sifilis'], inplace=True)
            df['hiv'].replace([0, 1, 2, 3], ['n_hiv', 's_hiv', 'n_hiv', 'd_hiv'], inplace=True)
            df['covid_mae'].replace([0, 1, 2, 3], ['n_covid', 's_covid', 'n_covid', 'd_covid'], inplace=True)

            # Consolidações
            df['dm'].replace([0,1,2,3,4,5,6], ['n_dm', 'n_dm','s_dm','s_dm', 's_dm','s_dm','d_dm'], inplace=True)

        self.colnames = df.columns

        return df



class NumericalTransformer(BaseEstimator, TransformerMixin):
    # Tipos de scalers
    # model 0: minmax
    # model 1: standard
    # model 2: without scaler
    def __init__(self, model=0, colnames=None):
        self.model = model
        self.colnames = colnames
        self.scaler = None

    def fit(self, X, y=None):
        df = pd.DataFrame(X, columns=self.colnames)
        # minmax
        if self.model == 0:
            self.scaler = MinMaxScaler()
            self.scaler.fit(df)
        # standard scaler
        elif self.model == 1:
            self.scaler = StandardScaler()
            self.scaler.fit(df)
        return self

    def get_feature_names_out(self):
        return self.colnames

    def transform(self, X, y=None):
        df = pd.DataFrame(X, columns=self.colnames)

        # update columns name
        self.colnames = df.columns.tolist()

        # minmax
        if self.model == 0:
            # transform data
            df = self.scaler.transform(df)
        elif self.model == 1:
            # transform data
            df = self.scaler.transform(df)
        else:
            df = df.values

        return df