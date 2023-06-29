import pandas  as pd
import numpy   as np
import seaborn as sn
import matplotlib.pyplot as plt

def grafico_distribuicao(base,titulo,coluna):
    plt.title(titulo)
    base[coluna].value_counts().head(25).plot.bar()
    plt.show()

def correlacao_dados(base,colunas):
    correlacao = base[colunas].corr()
    sn.heatmap(correlacao, annot = True, fmt=".1f", linewidths=.6)

def plot_feature_importance(importância,nomes,model_type):

    #Criar arrays de importância e nomes de recursos
    feature_importance = np.array(importância)
    feature_names = np.array(nomes)

    #Criar um DataFrame usando um Dicionário
    data={ 'feature_names' :feature_names, 'feature_importance' :feature_importance}
    fi_df = pd.DataFrame(data)

    #Classifique o DataFrame em ordem decrescente de importância do recurso
    fi_df.sort_values(by=[ 'feature_importance' ], ascending= False ,inplace= True )

    #Define o tamanho do gráfico de barras
    plt.figure(figsize=(7,5))
    #Plot gráfico de barras de Seaborn
    sns.barplot(x=fi_df[ 'feature_importance' ], y=fi_df[ 'feature_names' ])
    #Adicionar rótulos de gráfico
    plt.title(model_type + 'Feature Importance' )
    plt.xlabel( 'Feature Importance' )
    plt.ylabel( 'Feature Names' )
