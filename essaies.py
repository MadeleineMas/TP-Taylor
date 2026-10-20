import autograd.numpy as np
from autograd import grad

############## Domaine

# Q1
X= np.linspace(0, 2*np.pi, 500)

#Q2 Dérivée

def sin_der(X):
    grad_vec=np.vectorize(grad(np.sin))
    return grad_vec(X)

sin_der(X)

#Q3 vérificatin

print (all(np.isclose(sin_der(X), np.cos(X),rtol=0.001)))

############### Formule de Taylor

# 3.1 Version 1 : dérivées fournies

from math import factorial
import matplotlib.pyplot as plt

def taylor1(X, derivatives):
    P = np.zeros(len(X))
    for j in range(len(X)):
        for k in range(len(derivatives)):
            P[j] += derivatives[k]/factorial(k)*X[j]**k
    return P

sinus10 = [0, 1, 0, -1, 0, 1, 0, -1, 0, 1]
X = np.linspace(-2 * np.pi, 2 * np.pi, 400)

plt.plot(X, taylor1(X, sinus10), 'b')
plt.plot(X, np.sin(X), 'r')
plt.title ('Figure Taylor 1')
plt.show()


# 3.2 Version 2 : dérivées calculées avec autograd

def taylor2(X, f, n):
    derivee = []
    s = f
    for i in range (n+1):
        derivee.append(s(0.0))
        s=grad(s)
    return taylor1(X,derivee)

# exp à l'ordre 2 : 1 + x + x^2/2
assert np.allclose(taylor2(X, np.exp, 2), 1 + X + X**2 / 2)

# sinus au degré 10, à comparer avec np.sin(X)
Y = np.sin(X)
Y2 = taylor2(X, np.sin, 10)

#1.
plt.plot(X,Y,'r')
plt.plot(X,Y2,'b')
plt.title("Test Taylor 2")
plt.show()

#2. L'approximation semble acceptable sur [-4;4]

#3. 

plt.plot(X, np.sin(X), 'r')
plt.plot(X, taylor2(X, np.cos,0), 'b')
plt.title ("test avec cos a l'ordre 0")
plt.show()
    
# on constate que cela ne marche pas du tout, à l'ordre 0 l'approximation est constnate égale a 1.
# Il faut un n suffisant pour que l'approximation soit valable
         

#3.3 Une fonction personnalisé

def custom(X):
    return 2* np.sin(X) + np.cos(X/4)

Y3 = custom(X)
Y4 = taylor2(X, custom, 14)

plt.plot(X, Y3, 'r')
plt.plot(X, Y4, 'b')
plt.title ("test avec custom")
plt.show()

# De même sur les bords l'approxiamtion n'est plus valable, mais sur le domaine [-4:4] cela semble 
# encore cohérent

################# 4. L'application : widgets et graphique


