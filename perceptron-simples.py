
# -*- coding: utf-8 -*-

import numpy as np

#define o numero de épocas e de amostras (q)
numEpocas =70
q = 6 

#atributos
peso = np.array([113,122,107,98,115, 120])
ph = np.array([6.8, 4.7, 5.2, 3.6, 2.9,4.2])

#Bias
bias = 1

#Entrada do perceptron

x = np.vstack((peso,ph))
y = np.array([-1, 1, -1, -1, 1, 1])

# Defina o vetor de pesos
w = np.zeros([1,3])
#taxa de aprendizado
eta = 0.1

#Array para armazenar os erros
e = np.zeros(6)

# Define o vetor de pesos

def funcaoAtivacao(valor):
    # A função de ativação a degrau bipolar 
    if valor < 0.0:
        return (-1)
    else:
        return (1)

for j in range(numEpocas):
    for k in range (q):
        #insere o bias no vetor de entrada 
        xb = np.hstack((bias, x[:,k]))
        
        # Calcule o campo induzido
        v = np.dot(w,xb)        #Equação 5
        
        # Calcular a saída do perceptron
        yr = funcaoAtivacao(v)      #Equação 6  
        
        # Calcula erro e: e=(y - yr)
        e[k] = y[k] - yr
        
        # Treinamento do perceptron
        w = w + eta*e[k]*xb
        #print (e[k])
    print (w)
print ("Vetor de erros e = " + str(e))
            
        


