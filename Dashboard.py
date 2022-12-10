import streamlit as st
import numpy as np
import math
import pandas as pd
import pickle
import shap  
import time
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
import pickle
from numpy import NaN
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
import requests
from urllib.request import urlopen
import json
import plotly.graph_objects as go
import warnings
import json
warnings.filterwarnings("ignore")

######################
#Page d'acceuil:
######################

st.markdown("<h1 style='text-align: center; color: yellowgreen;'>Application d'évaluation du risque de crédit💸</h1>", unsafe_allow_html=True)
st.markdown("<h1 style='text-align: center; color: mediumvioletred;'>🏦 Bienvenus! 🏦</h1>", unsafe_allow_html=True)
logo = Image.open("/home/ubuntu/oc_projet7/logo.png")

st.sidebar.image(logo)
st.markdown("<h2 style ='text-align: left; color: yellowgreen;'>Les principaux objectifs du projet:</h2>", unsafe_allow_html=True)


st.markdown("#### - Construire un modèle d'apprentissage automatique qui prédit la probabilité qu'un client ne rembourse pas son prêt.")
st.markdown("#### - Rendre ce modèle d'apprentissage automatique disponible via une API.")
st.markdown("#### - Créez un tableau de bord interactif pour les responsables des relations bancaires.")

img1 = Image.open("/home/ubuntu/oc_projet7/picture.png")
img2 = Image.open("/home/ubuntu/oc_projet7/vignette.png")

col1, col2 = st.columns([1, 1])

with col1: 
    st.image(img1)
    st.image(img2)
with col2: 
    st.markdown("<h3 style='text-align: left; color: mediumvioletred;'>Enjeux:</h3>", unsafe_allow_html=True)
    st.markdown("##### - Ecarter les pertes: Détecter si les clients seront capables de rembourser leurs crédits, pour se protéger de la perte en  accordant des prêts aux mauvais emprunteur.")
    st.markdown("##### - Gagner du profit: Eviter de perdre les bons  emprunteurs (qui sont en mesure de rembourser leurs crédits), en refusant de leurs accorder des crédits.")
    st.markdown("##### - Transparence: Avoir les arguments convaincants, pour expliquer aux clients de manière exhaustive, la décision d’accord ou de refus d’un crédit.")


###############
#DATASET
###############
@st.cache
def load_data():
    path_df =(r"/home/ubuntu/oc_projet7/test_DASH.csv")
    df = pd.read_csv(path_df, index_col='SK_ID_CURR', encoding ='utf-8')#,index_col='SK_ID_CURR',nrows=100
    return df

df = load_data()
df=df.iloc[:,1:]

################
# CLIENT
################
def client(df, id):
        client =df[df.index == int(id)]# df[df.SK_ID_CURR == int(id)]#df[df.index == int(id)]
        return client 
    

################
# SIDBAR
################
st.sidebar.header("Informations Client")
st.sidebar.write("Veuillez selectionner un 🙋‍♀️ Client ID 🙋‍♂️")

 #Loading selectbox
ID= st.sidebar.selectbox("Client ID", df.index)#("Client ID", df['SK_ID_CURR'])#("Client ID", df.index)
infos_client = client(df, ID)


st.sidebar.markdown("**Sexe**")
st.sidebar.text( infos_client["CODE_GENDER"].values[0])

st.sidebar.markdown("**Âge :**")
st.sidebar.text("{:.0f} ans".format(int(infos_client["DAYS_BIRTH"]/-365)))

st.sidebar.markdown("**Statut familial**")
st.sidebar.text(infos_client["NAME_FAMILY_STATUS"].values[0])

st.sidebar.markdown("**Nombre d'enfants :**")
st.sidebar.text("{:.0f}".format(infos_client["CNT_CHILDREN"].values[0])) 

st.sidebar.markdown("**Type de profession**")
st.sidebar.text(infos_client["OCCUPATION_TYPE"].values[0])

st.sidebar.markdown("**Type de contrat**")
st.sidebar.text(infos_client["NAME_CONTRACT_TYPE"].values[0])



st.write("### Client selectionné:",ID)

#####################################################
#positionnement du client =>Âge distribution plot
#####################################################
@st.cache
def load_age_population(df):
    data_age = (df["DAYS_BIRTH"]//-365)
    return data_age

data_age = load_age_population(df)

######################################################
#positionnement du client =>income distribution plot
######################################################
@st.cache
def load_income_population(df):
    df_income = pd.DataFrame(df["AMT_INCOME_TOTAL"])
    return df_income

data_income = load_income_population(df)

######################################################
#positionnement du client =>credit distribution plot
######################################################
@st.cache
def load_credit_population(df):
    df_credit = pd.DataFrame(df["AMT_CREDIT"])
    return df_credit

data_credit = load_credit_population(df)

#########################################################################################
# La position du client par rapport aux clients du data de traitement (Anciens clients)
#########################################################################################
path_df_histo =(r"/home/ubuntu/oc_projet7/train_DASH.csv")
df_h = pd.read_csv(path_df_histo, index_col='SK_ID_CURR', encoding ='utf-8')

#L'Âge
st.subheader("Position du client par rapport aux anciens clients ")
if st.checkbox("Voir la situation du client sélectionné par rapport aux anciens clients?"):
    fig, ax = plt.subplots(figsize=(10, 5)) 
    sns.distplot(df_h[df_h['TARGET'] == 0]['DAYS_BIRTH']/-365, label = 'client solvable')
    sns.distplot(df_h[df_h['TARGET'] == 1]['DAYS_BIRTH']/-365, label = 'client non solvable')
    ax.axvline(int(infos_client["DAYS_BIRTH"].values / -365), color="mediumvioletred", linestyle='--')
    ax.set_title ("Distributions des clients selon leurs âges", fontname="Times New Roman", size=20,fontweight="bold")
    ax.set_xlabel('Âge du client (ans)', fontname="Times New Roman", size=15,fontweight="bold")
    ax.set_ylabel("Density", fontname="Times New Roman", size=15,fontweight="bold")
    plt.legend()
    st.pyplot(fig)

#EXT_SOURCE_3
    fig, ax = plt.subplots(figsize=(10, 5))
        
    sns.distplot(df_h[df_h['TARGET'] == 0]['EXT_SOURCE_3'], label = 'client solvable')
    sns.distplot(df_h[df_h['TARGET'] == 1]['EXT_SOURCE_3'], label = 'client non solvable')
    ax.axvline(infos_client["EXT_SOURCE_3"].values, color="mediumvioletred", linestyle='--')
    ax.set_title ("Distributions des clients selon le score normalisé à partir d'une source de données externe", fontname="Times New Roman", size=20,fontweight="bold")
    ax.set_xlabel("Score normalisé à partir d'une source de données externe", fontname="Times New Roman", size=15,fontweight="bold")
    ax.set_ylabel("Density", fontname="Times New Roman", size=15,fontweight="bold")
    plt.legend()
    st.pyplot(fig)
#Revenu
    fig, ax = plt.subplots(figsize=(10, 5))    
    sns.distplot(df_h[df_h['TARGET'] == 0]["AMT_INCOME_TOTAL"], label = 'client solvable')
    sns.distplot(df_h[df_h['TARGET'] == 1]["AMT_INCOME_TOTAL"], label = 'client non solvable')
    ax.axvline(int(infos_client["AMT_INCOME_TOTAL"]), color="mediumvioletred", linestyle='--')
    ax.set_title ("La distribution des clients selon le revenu", fontname="Times New Roman", size=20,fontweight="bold") 
    ax.set_xlabel("Revenus", fontname="Times New Roman", size=15,fontweight="bold")
    ax.set_ylabel("Density", fontname="Times New Roman", size=15,fontweight="bold")
    plt.legend()
    st.pyplot(fig)
else:
        st.markdown("<i>…</i>", unsafe_allow_html=True)
        
########################################################
# La position du client par rapport aux clients du data de traitement (Anciens clients)
########################################################

st.subheader("Position du client par rapport aux autres clients avant la prédiction ")
if st.checkbox("Voir la situation du client sélectionné par rapport aux autres clients avant la prédiction?"):    
    fig, ax = plt.subplots(figsize=(12, 4))
    sns.histplot(data_age, edgecolor = 'k', color="yellowgreen", bins=30)
    ax.axvline(int(infos_client["DAYS_BIRTH"].values // -365), color="mediumvioletred", linestyle='solid')
    ax.set_title ("Distributions des clients selon leurs âges", fontname="Times New Roman", size=20,fontweight="bold")
    ax.set_xlabel('Âge du client (ans)', fontname="Times New Roman", size=15,fontweight="bold")
    ax.set_ylabel("Nombre de clients", fontname="Times New Roman", size=15,fontweight="bold")
    st.pyplot(fig)


    colo1, colo2 = st.columns([1, 1])
    with colo1: 
        figure, ax = plt.subplots(figsize=(12, 8))
        sns.histplot(data_income, edgecolor = 'k', color="yellowgreen", bins=20)
        ax.axvline(int(infos_client["AMT_INCOME_TOTAL"].values[0]), color="mediumvioletred", linestyle='solid')
        ax.set_title ("La distribution des clients selon le revenu", fontname="Times New Roman", size=20,fontweight="bold") 
        ax.set_xlabel("Revenus", fontname="Times New Roman", size=15,fontweight="bold")
        ax.set_ylabel("Nombre de client", fontname="Times New Roman", size=15,fontweight="bold")
        st.pyplot(figure)
    with colo2: 
        figure, ax = plt.subplots(figsize=(12, 8))
        sns.histplot(data_credit, edgecolor = 'k', color="yellowgreen", bins=20)
        ax.axvline(int(infos_client["AMT_CREDIT"].values[0]), color="mediumvioletred", linestyle='--')
        ax.set_title ("La distribution des clients selon le montant de crédit", fontname="Times New Roman", size=20,fontweight="bold") 
        ax.set_xlabel("Montant de crédit", fontname="Times New Roman", size=15,fontweight="bold")
        ax.set_ylabel("Nombre de client", fontname="Times New Roman", size=15,fontweight="bold")
        st.pyplot(figure)
    #####################################
    #Graphique (Crédit/Revenu)
    st.write("#### Graphique (Crédit/Revenu)", infos_client["AMT_INCOME_TOTAL"], infos_client["AMT_CREDIT"])   
    chart_data = pd.DataFrame(df,columns=['AMT_CREDIT', 'AMT_INCOME_TOTAL'])
    st.area_chart(chart_data)
    

###########################################
#Affichage de la décision d'accordement de crédit 
###########################################
#Appel de l'API : 
url_API = 'http://172.31.95.153:5000/predict'
r = requests.post(url_API,json={'ID':ID})
#print(r.json())  
data_API = r.json()
#st.write(r.json())

proba = 1-data_API['proba']
client_score = round(proba*100, 2)
#####################################################
st.header('‍⚖️ Scoring et décision du modèle⚖️')
if st.checkbox("Afficher la décision de crédit"):
    st.write("##### Proba : ", data_API["proba"])
    st.write("##### Prediction : ",data_API["prediction"])
   
    with st.spinner("Chargement du 'Score Client'⌛"):
        time.sleep(3)
        data_API = r.json()
        classe_predite = data_API['prediction']
        if classe_predite == 1:
            decision = '❌ Client non solvable (Crédit Refusé)'
        else:
            decision = '✅ Client solvable (Crédit Accordé)'
        proba = 1-data_API['proba']

        client_score = round(proba*100, 2)

        left_column, right_column = st.columns((1, 2))

        left_column.markdown('Risque de défaut: **{}%**'.format(str(client_score)))
        left_column.markdown('Seuil par défaut du modèle: **50%**')

        if classe_predite == 1:
            left_column.markdown(
                'Décision: <span style="color:red">**{}**</span>'.format(decision),\
                unsafe_allow_html=True)   
        else:    
            right_column.markdown(
                'Décision: <span style="color:yellowgreen">**{}**</span>'\
                .format(decision), \
                unsafe_allow_html=True)
        
    ####################################
    #La jauge
    ####################################

    gauge = go.Figure(go.Indicator(
        mode = "gauge+delta+number",
        title = {'text': 'Pourcentage de risque de défaut'},
        value = client_score,
        domain = {'x': [0, 1], 'y': [0, 1]},
        gauge = {'axis': {'range': [None, 100]},
                    'steps' : [
                        {'range': [0, 25], 'color': "greenyellow"},
                        {'range': [25, 50], 'color': "yellow"},
                        {'range': [50, 75], 'color': "orange"},
                        {'range': [75, 100], 'color': "red"},
                        ],
                    'threshold': {
                'line': {'color': "mediumvioletred", 'width': 10},
                'thickness': 0.8,
                'value': client_score},

                    'bar': {'color': "mediumvioletred", 'thickness' : 0.2},
                },
        ))

    gauge.update_layout(width=450, height=250, 
                        margin=dict(l=50, r=50, b=0, t=0, pad=4))

    st.plotly_chart(gauge)
###########################################
#SHAP values (L'importance locale des variables)
###########################################


path_df =(r"/home/ubuntu/oc_projet7/test_API.csv")
df_API = pd.read_csv(path_df, encoding ='utf-8')
model = pickle.load(open('/home/ubuntu/oc_projet7/lgbm_Classifier.pkl','rb'))
print(model)
st.write("### Feature importance locale")
variables_importance_locale = st.checkbox(
                "Quelles sont les variables qui ont contribué le plus à la décision du modèle ?")
if (variables_importance_locale):
    shap.initjs()

    number = st.slider('Veuillez choisir le nombre de variables à visualiser : ', \
                                   3, 25, 10)

    DF = df_API[df_API['SK_ID_CURR'] == ID]
    colonnes_off = ['Unnamed: 0','SK_ID_CURR', 'INDEX', 'TARGET']
    col_pertinentes = [col for col in df_API.columns if col not in colonnes_off]
    DF = DF[col_pertinentes]

    fig, ax = plt.subplots(figsize=(10, 10))
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(DF)
    shap.summary_plot(shap_values[0], DF, plot_type ="bar", \
                        max_display=number, color_bar=False, plot_size=(8, 8))


    st.pyplot(fig)
    
    
######################################################
#SHAP values (L'importance globale des variables)
######################################################
st.write("### Feature importance globale")
variables_importance_globale = st.checkbox("Afficher la feature importance globale")
if (variables_importance_globale):
    shap.initjs()
    DF1 = df_API
    DF2 = DF1[col_pertinentes]
    figu, ax = plt.subplots(figsize=(10, 10))
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(DF2)
    shap.summary_plot(shap_values[0], DF2, plot_type ="bar", \
                    max_display=number, color_bar=False, plot_size=(8, 8))


    st.pyplot(figu)


###########################################################################
#La position du client par rapport aux autres clients après la prédiction
###########################################################################

# Ajout de la prédiction au dataframe


colonnes_off = ['Unnamed: 0','SK_ID_CURR', 'INDEX', 'TARGET']
col_pertinentes = [col for col in df_API.columns if col not in colonnes_off]
DF1 = df_API
DF2 = DF1[col_pertinentes]

pred = model.predict(DF2)

df_API['accord_crédit']= pred 


st.subheader("Position du client")
if st.checkbox("Voir la situation du client sélectionné par rapport aux autres clients (solvables ou non solvables?)"):
    # Position du client selon un score normalisé à partir d'une source de données externe du client par rapport aux clients
   
    fig, ax = plt.subplots(figsize=(10, 5))
    
    sns.distplot(df_API[df_API['accord_crédit'] == 0]['EXT_SOURCE_3'], label = 'client solvable')
    sns.distplot(df_API[df_API['accord_crédit'] == 1]['EXT_SOURCE_3'], label = 'client non solvable')
    ax.axvline(infos_client["EXT_SOURCE_3"].values, color="mediumvioletred", linestyle='--')
    ax.set_title ("Distributions des clients selon le score normalisé à partir d'une source de données externe", fontname="Times New Roman", size=20,fontweight="bold")
    ax.set_xlabel("Score normalisé à partir d'une source de données externe", fontname="Times New Roman", size=15,fontweight="bold")
    ax.set_ylabel("Density", fontname="Times New Roman", size=15,fontweight="bold")
    plt.legend()
    st.pyplot(fig)
    
    
    # Position du client selon l'âge par rapport aux autres clients 
    fig, ax = plt.subplots(figsize=(10, 5))
    
    sns.distplot(df_API[df_API['accord_crédit'] == 0]['DAYS_BIRTH']/-365, label = 'client solvable')
    sns.distplot(df_API[df_API['accord_crédit'] == 1]['DAYS_BIRTH']/-365, label = 'client non solvable')
    ax.axvline(int(infos_client["DAYS_BIRTH"].values / -365), color="mediumvioletred", linestyle='--')
    ax.set_title ("Distributions des clients selon leurs âges", fontname="Times New Roman", size=20,fontweight="bold")
    ax.set_xlabel('Âge du client (ans)', fontname="Times New Roman", size=15,fontweight="bold")
    ax.set_ylabel("Density", fontname="Times New Roman", size=15,fontweight="bold")
    plt.legend()
    st.pyplot(fig)
    
   
    # Position du client selon le revenu par rapport au autres clients
    fig, ax = plt.subplots(figsize=(10, 5))
    
    sns.distplot(df_API[df_API['accord_crédit'] == 0]["AMT_INCOME_TOTAL"], label = 'client solvable')
    sns.distplot(df_API[df_API['accord_crédit'] == 1]["AMT_INCOME_TOTAL"], label = 'client non solvable')
    ax.axvline(int(infos_client["AMT_INCOME_TOTAL"]), color="mediumvioletred", linestyle='--')
    ax.set_title ("La distribution des clients selon le revenu", fontname="Times New Roman", size=20,fontweight="bold") 
    ax.set_xlabel("Revenus", fontname="Times New Roman", size=15,fontweight="bold")
    ax.set_ylabel("Density", fontname="Times New Roman", size=15,fontweight="bold")
    plt.legend()
    st.pyplot(fig)

else:
        st.markdown("<i>…</i>", unsafe_allow_html=True)
        
        
    
        
       
    
             


       
           
