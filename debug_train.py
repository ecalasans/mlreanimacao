import logging
import wandb
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.tree import DecisionTreeClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import fbeta_score, precision_score, recall_score, accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay

# Tamanho do conjunto de teste(fração do dataset que é utilizada como dados de teste)
val_size = 0.3

# Coeficiente de aleatoriedade
seed = 1618

# Variável alvo(target)
stratify = 'reanimacao'

# Artefato de entrada
input_artifact = 'mlreanimacao/train.csv:latest'

# Tipo do artefato
artifact_type = 'Train'

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(message)s",
                    datefmt='%d-%m-%Y %H:%M:%S')

# Objeto logging
logger = logging.getLogger()


run = wandb.init(project='mlreanimacao', job_type='train')

# Registra um log desta ação
logger.info("Baixando artefato e realizando leitura...")
artifact = run.use_artifact(input_artifact)
artifact_file = artifact.file()
df_to_split = pd.read_csv(artifact_file)


logger.info("Divisão em train/val")

x_train, x_val, y_train, y_val = train_test_split(df_to_split.drop(labels=stratify,axis=1),
                                                  df_to_split[stratify],
                                                  test_size=val_size,
                                                  random_state=seed,
                                                  shuffle=True,
                                                  stratify=df_to_split[stratify])

logger.info("x train: {}".format(x_train.shape))
logger.info("y train: {}".format(y_train.shape))
logger.info("x val: {}".format(x_val.shape))
logger.info("y val: {}".format(y_val.shape))

logger.info("Remoção de outliers")

# Variável temporária
x = x_train['idade_materna'].copy()

# Redimensiona variável para adequar ao procedimento - técnica sugerida pela mensagem de erro
x = x.values.reshape(-1, 1)

# Identifica e prevê outliers em um único passo
lof = LocalOutlierFactor()
outlier = lof.fit_predict(x)
mask = outlier != -1

logger.info("x_train shape [original]: {}".format(x_train.shape))
logger.info("x_train shape [outlier removal]: {}".format(x_train.loc[mask,:].shape))

x_train = x_train.loc[mask,:].copy()
y_train = y_train[mask].copy()

logger.info("Codificando variável target")
# Objeto codificador
le = LabelEncoder()

# Treinamento e transformação do conjunto de treinamento
y_train = le.fit_transform(y_train)

# Transformação do conjunto de validação - não é necessário treinamento pois já
# foi realizado
y_val = le.transform(y_val)

logger.info("Classes [0, 1]: {}".format(le.inverse_transform([0, 1])))


colunas = x_train.select_dtypes('object').columns.to_list()

for col in colunas:
    # Cria o objeto OneHotEncoder
    one_hot = OneHotEncoder(sparse=False, drop='first')

    # Treinamento do codificador
    one_hot.fit(x_train[col].values.reshape(-1,1))

    # Criação de novas colunas
    x_train[one_hot.get_feature_names_out()] = one_hot.transform(x_train[col].values.reshape(-1,1))
    x_val[one_hot.get_feature_names_out()] = one_hot.transform(x_val[col].values.reshape(-1,1))