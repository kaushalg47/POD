import streamlit as st
import pandas as pd
import numpy as np
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn import linear_model
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn import metrics
from sklearn.svm import SVC
from sklearn import preprocessing
import matplotlib.axes as ax
from patsy import dmatrices
from statsmodels.tools.tools import add_constant  
import matplotlib.colors as mcolors
from sklearn.preprocessing import MinMaxScaler
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import statsmodels.api as sm

#PREPROCESSING

df = pd.read_excel(io=r"C:\Users\kaushal.i.g\Downloads\Synthetic data for CU solution (1).xlsx", sheet_name="Data dump")
df = df.fillna(0)

median = df['PD_Final'].median()
df['AT_02'] = df['AT_02'].map({'M':1, 'F':0})
df['AT_06'] = df['AT_06'].map({'Salaried':1, 'Social Security':3, 'Business':2})
for i, j in df.iterrows():
    if j['PD_Final'] >= median:
        df.at[i, 'status'] = 1
    else:
        df.at[i, 'status'] = 0



#LOGISTIC REGRESSION

X = df[['AT_09', 'AT_12', 'AT_13']]
y = df['status']

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)
logr = LogisticRegression()
logr.fit(X_scaled, y)
pickle.dump(logr, open('model.pkl', 'wb'))



#COEFFICIENT
# df = sm.add_constant(df)
# probit_model = sm.Probit(df['status'], df[[
#          'AT_09', 'AT_12', 'AT_13', 
#     ]])


# probit_results = probit_model.fit()

# coef1 = "{:.8f}".format(float(probit_results.params['AT_09']))
# coef2 = "{:.8f}".format(float(probit_results.params['AT_12']))
# coef3 = "{:.8f}".format(float(probit_results.params['AT_13']))

# df['Intercept'] = 1

# probit_model = sm.Probit(df['status'], df[['Intercept','AT_09','AT_12', 'AT_13']])
# probit_results = probit_model.fit()
# coefficients = pd.DataFrame({'Variable': ['Intercept','AT_09', 'AT_12', 'AT_13'],
#                              'Coefficient': probit_results.params})

# coef1 = coefficients[coefficients['Variable'].isin(['AT_09'])]
# coef2 = coefficients[coefficients['Variable'].isin(['AT_12'])]
# coef3 = coefficients[coefficients['Variable'].isin(['AT_13'])]

coef1 = '- 0.000044'
coef2 = '- 0.002171'
coef3 = '0.000146'

#STREAMLIT UI

st.markdown("<h1 style='text-align: center'> Probablity of Default </h1>", unsafe_allow_html=True)
st.markdown("<div style='height: 30px'></div", unsafe_allow_html=True )
col1, col2, col3 = st.columns(3)

with col1:
    st.text(f"coeff: {coef1}")
    SAT_09 = st.number_input(f"Income per month: ")
with col2:
    st.text(f"coeff: {coef2}")
    SAT_12 = st.number_input("Number of existing credit lines: \n")    
with col3:
    st.text(f"coeff: {coef3}")
    SAT_13 = st.number_input("Total outstanding credit lines: ")



#MODEL
pickled_model = pickle.load(open('model.pkl', 'rb'))

new_data = pd.DataFrame([[SAT_09, SAT_12, SAT_13]])
new_data_scaled = scaler.transform(new_data)
predicted = pickled_model.predict(new_data_scaled)
print(predicted)

#PROBABILITY

prob = pickled_model.predict_proba(new_data_scaled)
print(prob[0][1])


#OUTPUT 

if st.button('Calculate'):
  st.subheader('PD: '+ str(prob[0][1]))    
  st.markdown("<div style='height: 30px'></div", unsafe_allow_html=True )
  st.text("NOTE: PD:0 has higher chances of loan acceptance while PD:1 has the least")


      