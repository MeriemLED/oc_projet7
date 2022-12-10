
# Importation de librairies
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify
import pickle
from lightgbm import LGBMClassifier as lgb
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
import json



# importé le Data Set
path_df =(r"/home/ubuntu/oc_projet7/test_API.csv")
df = pd.read_csv(path_df, encoding ='utf-8')
print('La taille de df est: ', df.shape)

app = Flask(__name__)
# Chargement du modèle
model = pickle.load(open('lgbm_Classifier.pkl','rb'))
print (model)
@app.route('/predict',methods=['POST'])
def predict():
    data =  request.get_json(force=True)
    print('id client =', data)
    ID = int(data["ID"])
    print(ID)
    #Récupération des données du client en question
    
    DF = df[df['SK_ID_CURR'] == ID]
    colonnes_off = ['Unnamed: 0','SK_ID_CURR', 'INDEX', 'TARGET']
    col_pertinentes = [col for col in df.columns if col not in colonnes_off]
    DF = DF[col_pertinentes]
    
    print('X shape = ', DF.shape)
    
    proba = model.predict_proba(DF)
    prediction = model.predict(DF)
    
    dict_final = {
        'prediction' : int(prediction),
        'proba' : float(proba[0][0])
        }
    
    
    #print('Nouvelle Prédiction : \n', dict_final)

   # return jsonify(dict_final)
    
    return (dict_final)
    
#lancement de l'application
if __name__ == "__main__":
    app.run(host="172.31.95.153",port=5000,debug=True)
    








