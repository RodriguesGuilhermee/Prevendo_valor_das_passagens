#!/usr/bin/env python
# coding: utf-8

# In[64]:


import pandas as pd
import streamlit as st
import joblib


# In[2]:


st.header('Previsão de preços de passagens aéreas')
st.subheader('Descrição:')
st.write('''As informações a seguir serão preenchidas com o intuito de prever o preço das passagens aéreas para os territórios da Índia
que foram fornecidos na base dados, Sendo:
''')


# In[3]:


Airline={'Air Asia':0, 'Air India':1, 'GoAir':2, 'IndiGo':3, 'Jet Airways':4,
       'Jet Airways Business':5, 'Multiple carriers':6,
       'Multiple carriers Premium economy':7, 'SpiceJet':8, 'Trujet':9,
       'Vistara':10, 'Vistara Premium economy':11}

Source={'Banglore':0, 'Chennai':1, 'Delhi':2, 'Kolkata':3, 'Mumbai':4}

Destination={'Banglore':0, 'Cochin':1, 'Delhi':2, 'Hyderabad':3, 'Kolkata':4, 'New Delhi':5}

Additional_Info={'1 Long layover':0, '1 Short layover':1, '2 Long layover':2,
       'Business class':3, 'Change airports':4, 'In-flight meal not included':5,
       'No Info':6, 'No check-in baggage included':7, 'No info':8,
       'Red-eye flight':9}


# In[4]:


x_num={'Duration':0,'Total_Stops':0,'Journey_day':0,'Journey_mounth':0,'Dep_hour':0,'Dep_min':0,'Arrival_hour':0,'Arrival_min':0,
      }

x_lista=[Airline, Source, Destination, Additional_Info]





# In[30]:


dicionario={}
final=[]


# In[6]:


for item in x_num:
    if item=='Duration':
        valor=st.sidebar.number_input('Duração de voo em minutos',step=10,value=60,min_value=0,max_value=1000)
    elif item=='Total_Stops':
        valor=st.sidebar.slider('Total de paradas',0,4,0)
    elif item=='Journey_day':
        valor=st.sidebar.slider('Dia de Partida',0,31,0)
    elif item=='Journey_mounth':
        valor=st.sidebar.slider('Mês de partida',0,12,0)
    elif item=='Dep_hour':
        valor=st.sidebar.slider('Hora de partida',0,24,0)
    elif item=='Dep_min':
        valor=st.sidebar.slider('Minuto da partida',0,60,0)
    elif item=='Arrival_hour':
        valor=st.sidebar.slider('Hora de chegada',0,24,0)
    else:
        valor=st.sidebar.slider('Minuto da chegada',0,60,0)
    x_num[item] = valor


# In[31]:


companhia=st.selectbox('Companhia Aérea',Airline)
valor=(Airline[companhia])
final.append(valor)

partida=st.selectbox('Partida',Source)
valor=Source[partida]
final.append(valor)

destino=st.selectbox('Destino',Destination)
valor=Destination[destino]
final.append(valor)

add=st.selectbox('Informação adicional',Additional_Info)
valor=Additional_Info[add]
final.append(valor)

for i,j in enumerate(final):
    dicionario[i]=j


# In[ ]:





# In[21]:


botao = st.button('Prever Valor da Passagem')
    


# In[74]:


if botao:
    dicionario.update(x_num)
    valores_x = pd.DataFrame(dicionario, index=[0])
    valor_x=valores_x[[0,1,2,'Duration','Total_Stops',
       3, 'Journey_day', 'Journey_mounth', 'Dep_hour',
       'Dep_min', 'Arrival_hour', 'Arrival_min']]
    st.write(valor_x)
    modelo = joblib.load(r'C:\Users\rodri\Ciência de dados - Previsão do preço da passagem\modelo.joblib')
    preco = modelo.predict(valor_x)
    st.write('O valor da passagem será de: R$ {:.2f}'.format(preco[0]))

