#!/usr/bin/env python
# coding: utf-8

# ### Projeto de previsão de preços de passagens Aéreas

# #### Contexto:
#     Os preços das passagens aéreas podem ser algo complicado de se prever diante das diversas variáveis que existem que podem influenciar no preço do voo. Diante disso, temos um banco de dados com informações referentes aos meses de março e junho entre várias cidades da Índia que auxiliará na previsão do preço da passagem.

# In[1]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


# ### 1º Lendo nosso DataFrame

# In[2]:


df= pd.read_excel(r'D:\Downloads\Data_Train.xlsx')


# In[3]:


df.head()


# ### 2º Entendendo nosso conjunto de dados

# *Airline*: Esta coluna é referente aos tipos de companhias aéreas
# 
# *Data_of_journey*: Esta coluna é referente sobre a data em que a viagem do passageiro começará
# 
# *Source*: Esta coluna é referente ao local onde começará a viagem do passageiro
# 
# *Destination*: Esta coluna é referente ao local onde o passageiro irá desembarcar
# 
# *Route*: Esta coluna é referente a rota pela qual os passageiros optaram por viajar desde a sua origem até ao seu destino
# 
# *Arrival_time*: Esta coluna é referente a hora de chegada do passageiro ao seu destino
# 
# *Duration*: Esta coluna é referente a duração de todo o período de tempo de voo até chegar ao seus destino final
# 
# *Total_Stops*: Esta coluna é referente a quantas paradas foram realizadas até chegar ao destino final
# 
# *Additional_info*: Esta coluna é referente a informações adicionais tais como tipo de comida,etc
# 
# *Price*: Esta coluna é referente ao preço da passagem em Rúpias

# ### 3º Alterando o Dataset
#     Como nosso preço está em Rúpias, vamos passar para real p/ uma maior afinidade 

# In[4]:


df['Price']=df['Price']*0.061


# ### 4º Analisando os dados
#     Vamos observar as informações, descrições, etc dos nossos dados

# In[5]:


df.info()


# In[6]:


df.head(2)


# #### Alterando a coluna Duration
#     A coluna duration,dep_time, Date_of_Journey e Arrival_time deve ser um valor numérico, portanto, vamos transformá-la p um número inteiro que representará a quantidade de tempo em minutos.

# In[7]:


# Duration
# df['Duration']=df['Duration'].str.replace('h','*60').str.replace(' ','+').str.replace('m','*1').apply(eval)

#Date_of_journey
df['Journey_day']=df['Date_of_Journey'].str.split('/').str[0].astype(int)
df['Journey_mounth']=df['Date_of_Journey'].str.split('/').str[1].astype(int)
df.drop(['Date_of_Journey'],axis=1,inplace=True)

#Dep_time
df['Dep_hour']=pd.to_datetime(df['Dep_Time']).dt.hour
df['Dep_min']=pd.to_datetime(df['Dep_Time']).dt.minute
df.drop(['Dep_Time'],axis=1,inplace=True)

#Arrival_Time
df['Arrival_hour']=pd.to_datetime(df['Arrival_Time']).dt.hour
df['Arrival_min']=pd.to_datetime(df['Arrival_Time']).dt.minute
df.drop(['Arrival_Time'],axis=1,inplace=True)


# In[8]:


df.head(2)


# #### Verificando os valores nulos e duplicados

# In[9]:


df.isnull().sum()


# Temos apenas dois valores nulos, portanto, iremos excluir esses valores.

# In[10]:


df.dropna(inplace=True)


# In[11]:


df[df.duplicated()].count()


# Iremos excluir os 222 valores que estão duplicados

# In[12]:


df.drop_duplicates(keep='first',inplace=True)


# In[13]:


df[df.duplicated()].count()


# ### 5º Análise Gráfica
#     Vamos plotar gráfico do tipo boxplot p/ entender melhor como nossas variáveis se comportam diante do preço.
#     Além disso, faremos algumas outras análises gráficas para tentar entender melhor nosso dado

# #### Quanto a companhia aérea

# In[14]:


sns.catplot(y='Price',x='Airline',data=df.sort_values('Price',ascending=False),kind='box',height=8,aspect=2)


# Nota-se que a companhia Jet Airways Business possui um preço bem elevado ao ser comparado com as demais, possivelmente ela se trata de uma companhia para voos executivos

# #### Quanto ao destino

# In[15]:


sns.catplot(y='Price',x='Destination',data=df.sort_values('Price',ascending=False),kind='box',height=8,aspect=2)


# #### Quanto ao local de partida

# In[16]:


sns.catplot(y='Price',x='Source',data=df.sort_values('Price',ascending=False),kind='box',height=8,aspect=2)


# In[17]:


def grafico_barra(dataset,coluna):
    plt.figure(figsize=(15,5))
    plt.title('Contagem {}'.format(coluna))
    ax=sns.countplot(x='{}'.format(coluna),data=dataset)
    plt.xticks(rotation=45)


# #### Quantidade de voos por companhia aérea

# In[18]:


grafico_barra(df,'Airline')


# #### Quantidade de voos por companhia mês

# In[19]:


grafico_barra(df,'Journey_mounth')


# ### 6º Correlação entre variáveis
#     Agora, iremos observar como as variáveis e relacionam com a coluna preço.

# In[20]:


plt.figure(figsize=(15, 10))
sns.heatmap(df.corr(), annot=True, cmap='Greens')


# ### 7º Separação dos dados quantitativos e qualitativos
# 

# In[21]:


df_cod=df.drop(['Price'],axis=1)


# In[22]:


df_qualitativo=df_cod.select_dtypes(exclude=['int64','float','int32'])
df_quantitativo=df_cod.select_dtypes(include=['int64','float','int32'])


# ### 8º Fazendo o Encode para nossas categorias

# In[23]:


from sklearn.preprocessing import LabelEncoder


# In[24]:


l_encode=LabelEncoder()
df_qualitativo=df_qualitativo.apply(LabelEncoder().fit_transform)


# In[25]:


df_qualitativo


# ### 9º Conctenando os dados qualitativos e quantitavos

# In[26]:


df_cod=pd.concat([df_qualitativo,df_quantitativo],axis=1)


# ### 10º Modelo de Aprendizado

# In[27]:


from sklearn.metrics import r2_score, mean_squared_error
def avaliar_modelo(nome_modelo, y_teste, previsao):
    r2 = r2_score(y_teste, previsao)
    RSME = np.sqrt(mean_squared_error(y_teste, previsao))
    return f'Modelo {nome_modelo}:\nR²:{r2:.2%}\nRSME:{RSME:.2f}'


# In[28]:


from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor

modelo_rf = RandomForestRegressor()
modelo_lr = LinearRegression()
modelo_et = ExtraTreesRegressor()

modelos = {'RandomForest': modelo_rf,
          'LinearRegression': modelo_lr,
          'ExtraTrees': modelo_et,
          }

y = df['Price']
X = df_cod


# In[29]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.3, random_state=7)

for nome_modelo, modelo in modelos.items():
    #treinar
    modelo.fit(X_train, y_train)
    #testar
    previsao = modelo.predict(X_test)
    print(avaliar_modelo(nome_modelo, y_test, previsao))


# Diante disso, nota-se que o melhor modelo eh o ExtraTress, portanto, será ele que vamos adotar

# ### 11º Observando a importância de cada variável no modelo

# In[30]:


importancia_features = pd.DataFrame(modelo_et.feature_importances_, X_train.columns)
importancia_features = importancia_features.sort_values(by=0, ascending=False)
display(importancia_features)
plt.figure(figsize=(15, 5))
ax = sns.barplot(x=importancia_features.index, y=importancia_features[0])
ax.tick_params(axis='x', rotation=90)


# ### 12º Usando nosso modelo

# In[31]:


base_teste = df.copy()

y = base_teste['Price']
X = df_cod

X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.2, random_state=7)

modelo_et.fit(X_train, y_train)
previsao = modelo_et.predict(X_test)
print(avaliar_modelo('ExtraTrees', y_test, previsao))


# ### 13º Deploy

# In[32]:


#import joblib
#joblib.dump(modelo_et, 'modelo.joblib')


# In[ ]:





# In[111]:


plt.figure(figsize=(15, 5))
az=sns.barplot(x=df.groupby('Airline').Price.sum().index,y=list(df.groupby('Airline').Price.sum()))
az.tick_params(axis='x', rotation=90)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




