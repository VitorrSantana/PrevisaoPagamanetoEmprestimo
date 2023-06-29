import pandas as pd
import numpy  as np
from sklearn.model_selection import cross_val_score,cross_validate,cross_val_predict,KFold
from sklearn.metrics import accuracy_score

def treinar_modelo_tree(classificador,X_train,y_train,X_test,y_test,numeroArvores):
    metricas = pd.DataFrame({'n_trees':[],'oob':[],'acuracia_train':[],'acuracia_test':[]})
    for i,qtdArvores in enumerate(numeroArvores):

        classificador.set_params(n_estimators=qtdArvores)

        # Fit the model
        classificador.fit(X_train, y_train)
        y_pred_test = classificador.predict(X_test)
        y_pred_train = classificador.predict(X_train)
        if 'XGBClassifier' not in str(classificador): 
            oob = (1 - classificador.oob_score_)
        else:
            oob = np.nan
        # Metricas de Acuracia e Erro 
        metricas.loc[i,:] = [qtdArvores,oob,accuracy_score(y_train,y_pred_train),accuracy_score(y_test,y_pred_test)]
    return metricas,y_pred_test

def cross_validation(classificador,X,y,folders):
    kfold = KFold(n_splits=folders,shuffle=True,random_state=42)
    scores = cross_val_score(classificador, X, y, cv=kfold, scoring='accuracy')
    predict = cross_val_predict(classificador, X, y, cv=kfold)
    info_valid = cross_validate(classificador, X, y, cv=kfold)
    return scores,predict,info_valid
