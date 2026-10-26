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
ln = lambda x : np.log(1+x) #on crée la fonction associé a ln(1+x)
FONCTIONS = {'sin': np.sin, 'cos': np.cos, 'exp' : np.exp, 'ln(1+x)': ln, 'custom' : custom }

# 4.3 Widgets dans la barre latérale
with st.sidebar:
    st.selectbox('selection de la fonction', FONCTIONS, key="choix") #on crée chaque slider, on associe a chacun une key qui nous est utile pour les appelé pour la suite
    st.slider('ordre', min_value=0, max_value=10, value = 3, key = "ordre")
    st.slider('largeur de l_intervalle étudieé', min_value=2.0, max_value=20.0, value= 12.6, key="largeur")

f = st.session_state.choix #on met dans la variable f la fonction séléctionnée
X = np.linspace( -st.session_state.largeur / 2,st.session_state.largeur / 2, 400) #on crée les 400 points sur lequel on va effectué le calcul (axe X)
Y = FONCTIONS[f](X) # on calcul la fonction qui a pour clé f (qu'on a attribué plus tot)
Y_taylor = taylor2(X,FONCTIONS[f],st.session_state.ordre) # on calcul aussi l'approximation de Taylor de la fonction a l'ordre choisis


fig,ax=plt.subplots() # on plot le graphique
ax.plot(X, Y, label="Fonction f", color = 'red')
ax.plot(X, Y_taylor, label="Approximation de Taylor", color = 'blue')
ax.legend()
st.pyplot(fig)

# la qualité de l'aproximation diminue, pour ln(X+1) lorsuqe l'on augmente n, la fonction atteint un plateau 




######### 5. Qualité de l'approximation #########################################################################################




#5.1 Taux de coïncidence

def taux_coincidence(Y,Y_taylor,seuil):  # on regarde si la moyenne est bien en dessous du seuil True ou pas False
    return np.mean(np.abs(Y - Y_taylor) < seuil)

st.select_slider( "Seuil de coïncidence",options=[1e-6, 1e-5, 1e-4, 1e-3, 1e-2],value=1e-3,key="seuil") # on crée le slider pour choisir le sueil

taux = taux_coincidence(Y,Y_taylor,st.session_state.seuil) # on regarde si oui ou non l'écart est en dessous du seuil choisis

st.write(f"Taux de coïncidence : {taux:.2%}") #on affiche le resultat

#5.2 Courbe d'erreur

st.checkbox('Afficher l erreur', key='err') # on crée la boite "afficher l'erreur"

if st.session_state.err: # si la boite est cochée
    erreur = np.abs(Y - Y_taylor) #on calcul l'erruer

    fig_erreur, ax_erreur = plt.subplots() # on plot la difference
    ax_erreur.plot(X, erreur, label="Erreur absolue", color="green")

    ax_erreur.set_yscale("log") # on met l'echelle des ordonnée en logarithmique
    ax_erreur.legend()
    st.pyplot(fig_erreur)





####### 6. Comparer plusieurs ordres ##################################################################################################################################




if "ordres" not in st.session_state: # on initialise la liste des ordres pour s'en souvenir
    st.session_state.ordres = []

with st.sidebar:
    if st.button("Ajouter cet ordre à la comparaison"): # on crée le bouton dans la barre latérale
        n = st.session_state.ordre # on donne a la valeur n l'ordre choisis
        if n not in st.session_state.ordres: # on verifie que l'ordre n'est pas deja dans la liste (évite les doublons) avant de l'ajouter dans la liste crée avant
            st.session_state.ordres.append(n)

with st.sidebar:
    if st.button("Réinitialiser la comparaison"): # on crée la manière de recommencer la liste
        st.session_state.ordres = []

Y = FONCTIONS[f](X)# Calcul de la fonction

fig, ax = plt.subplots() # on trace la fonctions et les approximations
ax.plot(X, Y, label="Fonction f", color="black", linewidth=2)

for n in st.session_state.ordres: 
    Y_taylor = taylor2(X, FONCTIONS[f], n) # on calcul les approximation pour tout les ordres qui ont été stocké dans la liste
    ax.plot(X, Y_taylor, label=f"Taylor ordre {n}") #on affiche sur le même graphique chaque courbe pour chaque ordre


ax.set_title("Comparaison des approximations de Taylor")
ax.legend()
st.pyplot(fig)

st.write("Ordres sélectionnés :", st.session_state.ordres) # Affichage des ordres mémorisés, sinon on comprends rien


######### Vérification finale #####################################################################@