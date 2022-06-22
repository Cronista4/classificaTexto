## Imports
import time
import pandas as pd

import palavrasParada
import preProcessador
import numpy
import sys
import textblob
from googletrans import Translator
from textblob import TextBlob


def aplicaLimpeza(dadosBrutosCSV):
    ativaPOStag = False # Não vamos chamar a função de Part-Of-Speech

    #Lista de dados textuais limpos
    listaFrasesLimpa = []

    #Lista de contextos em ordem de leitura
    listaContextos = dadosBrutosCSV['context']

    indice = 0
    # Cada sentença ou comentario é pre-processado ou limpo por meio do preProcessador
    for sentenca in dadosBrutosCSV.review:
        print("ANTES:",sentenca)
        #indice = dadosFilmes[dadosFilmes['review']==sentenca].index.item()
        print("indice:",indice)
        contextoLinha = listaContextos[indice]
        print("contexto: ", contextoLinha)
        token = preProcessador.preprocessadorParaSalvar(sentenca,contextoLinha,ativaPOStag)
        print("DEPOIS:",token)
        listaFrasesLimpa.append(sentenca)
        dadosBrutosCSV.at[indice,'review'] = token
        print("####\n ")
        indice = indice + 1 #fecha o for
        #time.sleep(1)
    print("FIM")
    dadosBrutosCSV.to_csv('comentarios/DadosLimpos.csv')

    dadosLimposCSV = dadosBrutosCSV
    return dadosLimposCSV