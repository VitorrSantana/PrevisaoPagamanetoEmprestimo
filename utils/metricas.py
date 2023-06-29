import pandas  as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve, precision_recall_curve, confusion_matrix
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score
from sklearn.metrics import f1_score, roc_auc_score

def plot_matrix_confusao(y_real,y_pred):
    # sns.set_context('talk')
    cm = confusion_matrix(y_real, y_pred)
    ax = sns.heatmap(cm, annot=True, fmt='d')
    
    labels = ['Falso', 'Verdadeiro']
    ax.set_xticklabels(labels);
    ax.set_yticklabels(labels);
    ax.set_ylabel('Valor Real');
    ax.set_xlabel('Valor Predito')


def metricas_modelos(y_real, y_pred):
    cr = classification_report(y_real, y_pred)

    score_df = pd.DataFrame({'accuracy': accuracy_score(y_real, y_pred),
                            'precision': precision_score(y_real, y_pred),
                            'recall': recall_score(y_real, y_pred),
                            'f1': f1_score(y_real, y_pred),
                            'auc': roc_auc_score(y_real, y_pred)},
                            index=pd.Index([0]))

    return score_df,cr
