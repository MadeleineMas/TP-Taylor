######### Import des bibliothèques
import streamlit as st
import autograd.numpy as np
from autograd import grad
from math import factorial
import matplotlib.pyplot as plt

######## Import des fonctions Taylor/custom
def taylor1(X, derivatives):
    P = np.zeros(len(X))
    for j in range(len(X)):
        for k in range(len(derivatives)):
            P[j] += derivatives[k]/factorial(k)*X[j]**k
    return P

def taylor2(X, f, n):
    derivee = []
    s = f
    for i in range (n+1):
        derivee.append(s(0.0))
        s=grad(s)
    return taylor1(X,derivee)

def custom(X):
    return 2* np.sin(X) + np.cos(X/4)

####### texte de Test du début

st.title ("Approximation de Taylor")
st.subheader("ceci est un super sous-titre", icon="🔥")
st.latex( r'''P_n(x) = \sum_{k=0}^{n} \frac{f^{(k)}(0)}{k!} \, x^k''')

# 4.2 Catalogue de fonctions
ln = lambda x : np.log(1+x)
FONCTIONS = {'sin': np.sin, 'cos': np.cos, 'exp' : np.exp, 'ln(1+x)': ln, 'custom' : custom }

# 4.3 Widgets dans la barre latérale
with st.sidebar:
    st.selectbox('selection de la fonction', FONCTIONS)
    st.slider('ordre', min_value=0, max_value=10, value = 3)
    st.slider('largeur de l_intervalle étudieé', min_value=2, max_value=20, value= 13)


