# Essa função ou script tem como objetivo realizar a classificação de
# um conjunto de textos ou amostras textuais para que seja possível criar
# uma linha analisável sobre todo o processo de classificar um texto,
#  desde a leitura da entrada de dados até guardar o modelo de aprendizado de máquina
# Autor: Luís Marcello . Contato: luis.marcello@unesp.br

#Primeiro, os imports:
import pickle

import sklearn
from sklearn.datasets import load_breast_cancer

import limpaTexto
import time # para
import pandas as pd
import scipy
#from cffi.backend_ctypes import xrange

import palavrasParada
import preProcessador
import numpy
import sys
import textblob

##Classificadores
from sklearn.feature_extraction.text import CountVectorizer  ## Converter string para vetor
from sklearn.feature_extraction import text  # Possui stop-words
from sklearn.feature_extraction.text import TfidfVectorizer  # Extração TFIDF
from sklearn.model_selection import train_test_split, cross_validate  ## Método hold-out
from sklearn.model_selection import cross_val_score ## Método K-Fold
from sklearn.model_selection import KFold ## Método K-Fold
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score
from sklearn.linear_model import LogisticRegression  ## Regressão logística
from sklearn import svm  # SVM
from sklearn.naive_bayes import GaussianNB  # NB
from sklearn.ensemble import RandomForestClassifier  # RF
from sklearn.linear_model import SGDClassifier  # SGD - Stochastic Gradient Descent
from sklearn.neighbors import KNeighborsClassifier  # KNN
from sklearn import tree  # C4,5
from sklearn.neural_network import MLPClassifier  # MLP


##Métricas de qualidade
from sklearn.metrics import accuracy_score, make_scorer
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import roc_auc_score



def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Olá, {name}')  # Press Ctrl+F8 to toggle the breakpoint.

def funcaoTeste(valorA,valorB):
    if valorA>valorB:
        print("valor A é maior que valor B",valorA)
    else:
        print("valor B é maior que valor A", valorB)

    for i in range(0, 5):
        print("algo",i,"++")

def carregaDadosBrutos(caminho):
    #Lê o CSV de entrada. O CSV possui três colunas:
    #       sentiment, review, context
    # a primeira é o sentimento ou classe, o review é o comentário em si e o contexto é
    # de onde o comentário foi retirado
    dadosBrutosCSV = pd.read_csv(caminho)  # .sample(amostras,random_state=1)##random_state=estadoInicial

    print("Rótulos das classes:")
    print(dadosBrutosCSV.sentiment.value_counts())
    print("Rótulos dos contextos:")
    print(dadosBrutosCSV.context.value_counts())
    print("Rótulos das frases:")
    print(dadosBrutosCSV.review.value_counts())
    print("apresentação dos dados completos:\n")
    print(dadosBrutosCSV.to_string())
    #print(dadosBrutosCSV.head())
    return dadosBrutosCSV

# função responsável por realizar a limpeza dos dados brutos, remoção de URLs,
#emojis, acentos, pontuação, palavras de parada e radicalização.
def limpaDadosBrutos(dadosBrutosCSV):
    print("Limpa DADOS###############")
    dadosLimposCSV = limpaTexto.aplicaLimpeza(dadosBrutosCSV)
    return dadosLimposCSV

def construirMatrizAtributos(dadosLimposCSV):
    # Set das palavras de parada usadas na TF-IDF
    #palavrasDeParada = text.ENGLISH_STOP_WORDS.union(palavrasParada.minhas_paralavras_de_parada) -> 684 tokens
    palavrasDeParada = palavrasParada.minhas_paralavras_de_parada  ##Maior acurácia no M.E. ** -> 644 tokens
    # palavrasDeParada = text.ENGLISH_STOP_WORDS #Conjunto padrão do SK-Learn
    print("Palavras de parada usadas: ", palavrasDeParada, "\n Quantidade de palavras de parada:", len(palavrasDeParada))
    # Aplicamos a técnica TF-IDF; poderíamos aplicar três tipos:
    # TfidfVectorizer -> estado da arte
    # CountVectorizer -> Faz a conta fazendo a contagem da frequencia de cada termo
    # HashVectorizer -> pouco usado
    # Converte uma coleção de documentos brutos em uma matriz de características TF-IDF.
    vetorTFIDF = TfidfVectorizer(smooth_idf=True,  ##Evita divisão por zero na contagem
                                 use_idf=True,
                                 min_df=3,  # Ignore termos que aparecem em menos de 3 documentos (1)
                                 max_df=0.7,  # Ignore termos que aparecem em 70% ou mais dos documentos
                                 lowercase=True,  # Converte as letras para minúsculas
                                 stop_words=palavrasDeParada,  # minha lista de stop-words
                                 # preprocessor=preProcessador.meu_preprocessador,
                                 ##Realiza processos de remoção caracteres especiais e radicalização

                                 ngram_range=(1, 1)  ## N-gramas 1,1 unigrama
                                 )

    # Com base nos parâmetros anteriores, aprende um vocabulário a partir
    # das reviews com todos os tokens que passaram no critério
    vetorTFIDF = vetorTFIDF.fit(dadosLimposCSV.review)

    matrizTextoTFIDF = vetorTFIDF.transform(dadosLimposCSV.review)  # Transforma os documentos na matriz de documentos.
    # Extract token counts out of raw text documents using the vocabulary fitted with fit or the one provided to the constructor.

    print("\nMatriz de texto (linhas:colunas): ", matrizTextoTFIDF.shape)
    #print("Sentimentos:\n", dadosLimposCSV.sentiment)

    #print("Vocabulário: \n", vetorTFIDF.vocabulary_)# idem ao get_feature_names_out
    print("VETOR:\n", vetorTFIDF)
    print("ATRIBUTOS: \n", vetorTFIDF.get_feature_names_out())

    print("\n ->", len(vetorTFIDF.vocabulary_))
    print("matriz Texto:\n", matrizTextoTFIDF)
    print("TIPO: ", type(matrizTextoTFIDF))

    return matrizTextoTFIDF

def separaDadoTreinoTesteHoldOut(matrizModificada,dadosLimposCSV,estadoInicial):
    ##Separação dos dados - 70% - 30%
    ##Função train_test_split. Precisa receber: matriz de termos, as classes, o tamanho do set de teste e uma seed
    ## Os x são os atributos e Y é a classe
    x_treino, x_teste, y_treino, y_teste = train_test_split(
        matrizModificada,
        dadosLimposCSV.sentiment,
        test_size=0.3,
        random_state=estadoInicial
    )
    print("\nx_treino\n", x_treino)
    print("\nx_teste\n", x_teste)
    print("\ny_treino\n", y_treino)
    print("\ny_teste\n", y_teste)

    return x_treino, x_teste, y_treino, y_teste

def separaDadoTreinoTesteKFold(matrizModificada,dadosLimposCSV):
    k = 10
    kf = KFold(n_splits=k, random_state=None)
    classificadorME = LogisticRegression(C=2,
                                    random_state=0,
                                    solver='sag')

    vetorAcuracias = cross_val_score(classificadorME, matrizModificada, dadosLimposCSV.sentiment, cv=kf)

    #print("Avg accuracy: {}".format(vetorAcuracias.mean()))
    print("Média de acurácias", numpy.mean(vetorAcuracias)*100)
    print("Acurácias:", vetorAcuracias)

    #kfold = KFold(n_splits=5, random_state=None)
    #model = LogisticRegression(solver='liblinear')

    #print(sklearn.metrics.get_scorer_names())

    #nome_metricas = ['accuracy',
    #                 'precision_macro',
    #                 'recall_macro',
    #                 'f1_macro',
    #                 'average_precision',
    #                 'roc_auc',
    #                 'neg_mean_absolute_error']

    #metricas = cross_validate(model,
    #                          X=matrizModificada,
    #                          y=dadosLimposCSV.sentiment,
    #                          cv=kfold,
    #                          scoring=nome_metricas)
    #print(metricas,type(metricas))

    #for met in metricas:
    #    print(f"- {met}:")
    #    print(f"-- {metricas[met]}")
    #    print(f"-- {numpy.mean(metricas[met])} +- {numpy.std(metricas[met])}\n")

def escolheClassificadorHoldOut(x_treino,x_teste,y_treino,y_teste):
    print("Digite a opção de classificadores:")
    print("[1] - classificador ME")
    print("[2] - classificador NB")
    print("[3] - classificador SVM")
    print("[4] - classificador RF")
    print("[5] - classificador ME")
    print("[6] - classificador K-NN")
    print("[7] - classificador árvore de decisão")
    print("[8] - classificador MLP")
    escolhaClassificador = int(input())
    if escolhaClassificador<1 or escolhaClassificador>8:
        print("Escolha inválida! Encerrando...")
        exit()
    elif escolhaClassificador==1:
        print("#####################################")
        print("\nClassificador regressão logistica - Max Ent:\n")
        tempoExecucao = time.time()
        # solver{‘newton-cg’, ‘lbfgs’, ‘liblinear’, ‘sag’, ‘saga’}, default=’lbfgs’
        # Algorithm to use in the optimization problem.
        # For small datasets, ‘liblinear’ is a good choice, whereas ‘sag’ and ‘saga’ are faster for large ones.
        # For multiclass problems, only ‘newton-cg’, ‘sag’, ‘saga’ and ‘lbfgs’ handle multinomial loss; ‘liblinear’ is limited to one-versus-rest schemes.
        # ‘newton-cg’, ‘lbfgs’, ‘sag’ and ‘saga’ handle L2 or no penalty
        # ‘liblinear’ and ‘saga’ also handle L1 penalty
        # ‘saga’ also supports ‘elasticnet’ penalty
        # ‘liblinear’ does not support setting penalty='none'
        regLog = LogisticRegression(C=2,
                                    random_state=0,
                                    solver='sag')
        regLog = regLog.fit(x_treino, y_treino)
        y_predito = regLog.predict(x_teste)
        tempoExecucao = time.time() - tempoExecucao
        print("Tempo de execução:", tempoExecucao, "segundos")

        acuracia = accuracy_score(y_predito, y_teste)
        print("Acurácia: ", acuracia)
        precisao = precision_score(y_predito, y_teste, average='weighted')
        print("Precisão: ", precisao)
        sensibilidade = recall_score(y_predito, y_teste, average='weighted')
        print("Sensibilidade: ", sensibilidade)
        erroMedioAbsoluto = mean_absolute_error(y_predito, y_teste)
        print("Erro absoluto médio: ", erroMedioAbsoluto)
        # curvaROC = roc_auc_score(y_predito, y_teste)
        # print("Curva ROC: ", curvaROC)
        f1 = f1_score(y_predito, y_teste, average='weighted')
        print("F1: ", f1)
        matrizConfusao = confusion_matrix(y_predito, y_teste)
        print("Matriz de confusão: \n", matrizConfusao)
        return regLog

    elif escolhaClassificador == 2:
        print("#####################################")
        print("\nClassificador NB:\n")
        tempoExecucao = time.time()
        bayesianoIngenuo = GaussianNB()
        bayesianoIngenuo.fit(x_treino.toarray(), y_treino)
        y_predito = bayesianoIngenuo.predict(x_teste.toarray())
        tempoExecucao = time.time() - tempoExecucao
        print("Tempo de execução:", tempoExecucao, "segundos")

        acuracia = accuracy_score(y_predito, y_teste)
        print("Acurácia: ", acuracia)
        precisao = precision_score(y_predito, y_teste, average='weighted')
        print("Precisão: ", precisao)
        sensibilidade = recall_score(y_predito, y_teste, average='weighted')
        print("Sensibilidade: ", sensibilidade)
        erroMedioAbsoluto = mean_absolute_error(y_predito, y_teste)
        print("Erro absoluto médio: ", erroMedioAbsoluto)
        #curvaROC = roc_auc_score(y_predito, y_teste)
        #print("Curva ROC: ", curvaROC)
        f1 = f1_score(y_predito, y_teste, average='weighted')
        print("F1: ", f1)
        matrizConfusao = confusion_matrix(y_predito, y_teste)
        print("Matriz de confusão: \n", matrizConfusao)
        return bayesianoIngenuo

    elif escolhaClassificador == 3:
        print("#####################################")
        print("\nClassificador SVM:\n")
        tempoExecucao = time.time()
        clSVM = svm.SVC(kernel='linear',
                        degree=3,
                        max_iter=-1)  ## os kernels disponíveis são ('linear', 'poly', 'rbf')
        clSVM = clSVM.fit(x_treino, y_treino)
        y_predito = clSVM.predict(x_teste)
        tempoExecucao = time.time() - tempoExecucao
        print("Tempo de execução:", tempoExecucao, "segundos")

        acuracia = accuracy_score(y_predito, y_teste)
        print("Acurácia: ", acuracia)
        precisao = precision_score(y_predito, y_teste, average='weighted')
        print("Precisão: ", precisao)
        sensibilidade = recall_score(y_predito, y_teste, average='weighted')
        print("Sensibilidade: ", sensibilidade)
        erroMedioAbsoluto = mean_absolute_error(y_predito, y_teste)
        print("Erro absoluto médio: ", erroMedioAbsoluto)
        # curvaROC = roc_auc_score(y_predito, y_teste)
        # print("Curva ROC: ", curvaROC)
        f1 = f1_score(y_predito, y_teste, average='weighted')
        print("F1: ", f1)
        matrizConfusao = confusion_matrix(y_predito, y_teste)
        print("Matriz de confusão: \n", matrizConfusao)
        return clSVM

    elif escolhaClassificador == 4:
        print("#####################################")
        print("\nClassificador Random Forest:\n")
        tempoExecucao = time.time()
        clRF = RandomForestClassifier(n_estimators=20, random_state=0)
        clRF = clRF.fit(x_treino, y_treino)
        y_predito = clRF.predict(x_teste)
        tempoExecucao = time.time() - tempoExecucao
        print("Tempo de execução:", tempoExecucao, "segundos")
        acuracia = accuracy_score(y_predito, y_teste)
        print("Acurácia: ", acuracia)
        precisao = precision_score(y_predito, y_teste, average='weighted')
        print("Precisão: ", precisao)
        sensibilidade = recall_score(y_predito, y_teste, average='weighted')
        print("Sensibilidade: ", sensibilidade)
        erroMedioAbsoluto = mean_absolute_error(y_predito, y_teste)
        print("Erro absoluto médio: ", erroMedioAbsoluto)
        # curvaROC = roc_auc_score(y_predito, y_teste)
        # print("Curva ROC: ", curvaROC)
        f1 = f1_score(y_predito, y_teste, average='weighted')
        print("F1: ", f1)
        matrizConfusao = confusion_matrix(y_predito, y_teste)
        print("Matriz de confusão: \n", matrizConfusao)
        return clRF

    elif escolhaClassificador == 5:
        print("#####################################")
        print("\nClassificador K-NN:\n")
        tempoExecucao = time.time()
        clKNN = KNeighborsClassifier(n_neighbors=20)
        clKNN = clKNN.fit(x_treino, y_treino)
        y_predito = clKNN.predict(x_teste)
        tempoExecucao = time.time() - tempoExecucao
        print("Tempo de execução:", tempoExecucao, "segundos")

        acuracia = accuracy_score(y_predito, y_teste)
        print("Acurácia: ", acuracia)
        precisao = precision_score(y_predito, y_teste, average='weighted')
        print("Precisão: ", precisao)
        sensibilidade = recall_score(y_predito, y_teste, average='weighted')
        print("Sensibilidade: ", sensibilidade)
        erroMedioAbsoluto = mean_absolute_error(y_predito, y_teste)
        print("Erro absoluto médio: ", erroMedioAbsoluto)
        # curvaROC = roc_auc_score(y_predito, y_teste)
        # print("Curva ROC: ",curvaROC)
        f1 = f1_score(y_predito, y_teste, average='weighted')
        print("F1: ", f1)
        matrizConfusao = confusion_matrix(y_predito, y_teste)
        print("Matriz de confusão: \n", matrizConfusao)
        return clKNN

    elif escolhaClassificador == 6:
        print("#####################################")
        print("\nClassificador Árvore de decisão:\n")
        tempoExecucao = time.time()
        clAD = tree.DecisionTreeClassifier()
        clAD = clAD.fit(x_treino, y_treino)
        y_predito = clAD.predict(x_teste)
        tempoExecucao = time.time() - tempoExecucao
        print("Tempo de execução:", tempoExecucao, "segundos")

        acuracia = accuracy_score(y_predito, y_teste)
        print("Acurácia: ", acuracia)
        precisao = precision_score(y_predito, y_teste, average='weighted')
        print("Precisão: ", precisao)
        sensibilidade = recall_score(y_predito, y_teste, average='weighted')
        print("Sensibilidade: ", sensibilidade)
        erroMedioAbsoluto = mean_absolute_error(y_predito, y_teste)
        print("Erro absoluto médio: ", erroMedioAbsoluto)
        # curvaROC = roc_auc_score(y_predito, y_teste)
        # print("Curva ROC: ",curvaROC)
        f1 = f1_score(y_predito, y_teste, average='weighted')
        print("F1: ", f1)
        matrizConfusao = confusion_matrix(y_predito, y_teste)
        print("Matriz de confusão: \n", matrizConfusao)
        return clAD

    elif escolhaClassificador == 7:
        print("#####################################")
        print("\nClassificador Multi-layer Perceptron:\n")
        tempoExecucao = time.time()
        # The solver for weight optimization.
        # ‘lbfgs’ is an optimizer in the family of quasi-Newton methods.
        # ‘sgd’ refers to stochastic gradient descent.
        # ‘adam’ refers to a stochastic gradient-based optimizer
        # Note: The default solver ‘adam’ works pretty well on relatively large datasets
        # (with thousands of training samples or more) in terms of both training time and validation score.
        # For small datasets, however, ‘lbfgs’ can converge faster and perform better.
        clMLP = MLPClassifier(solver='adam',
                            alpha=1e-5,
                            hidden_layer_sizes=(150, 2),
                            random_state=1)
        ##hidden_layer_sizes -> cada valor é a quantidade de neuronios na camada. Neste caso, o 150 representa o número de features
        clMLP = clMLP.fit(x_treino, y_treino)
        y_predito = clMLP.predict(x_teste)
        tempoExecucao = time.time() - tempoExecucao
        print("Tempo de execução:",tempoExecucao,"segundos")
        acuracia = accuracy_score(y_predito, y_teste)
        print("Acurácia: ",acuracia)
        precisao = precision_score(y_predito, y_teste, average='weighted')
        print("Precisão: ",precisao)
        sensibilidade = recall_score(y_predito, y_teste, average='weighted')
        print("Sensibilidade: ",sensibilidade)
        erroMedioAbsoluto = mean_absolute_error(y_predito, y_teste)
        print("Erro absoluto médio: ",erroMedioAbsoluto)
        # curvaROC = roc_auc_score(y_predito, y_teste)
        # print("Curva ROC: ",curvaROC)
        f1 = f1_score(y_predito, y_teste, average='weighted')
        print("F1: ",f1)
        matrizConfusao = confusion_matrix(y_predito, y_teste)
        print("Matriz de confusão: \n",matrizConfusao)
        return clMLP

# Vamos modularizar o código e fazer todas as chamadas na main
def main():
    # Variaveis de controle
    # Usada para randomizar a ordem dos dados da base. É bom testar a acurácia com uns 5 estados (evita viés).
    estadoInicial = 1

    #print_hi('Posso passar qualquer parâmetro textual aqui. Python não é tipado!')
    #funcaoTeste(25,50)
    #funcaoTeste(10,1)

    # 1º precisamos carregar os dados, neste caso, os comentários:
    # Base nova A: base_limpa_padrao_ingles_3_classes_A_600
    # Base nova B: base_limpa_padrao_ingles_3_classes_B_600
    # Base nova C: base_limpa_padrao_ingles_3_classes_C_600
    # Base nova D: base_limpa_padrao_ingles_3_classes_D_900
    # Base nova E: base_limpa_ingles_3_classes_NOVA_3000 -> 3000 amostras
    # Base bruta F: base_bruta_3_classes_cheia_ingles -> 12.000 amostras
    #base_limpa_padrao_ingles_3_classes_NOVA_600

    #Carrega os dados na variavel dadosBrutosCSV
    dadosBrutosCSV = carregaDadosBrutos('comentarios/base_limpa_padrao_ingles_3_classes_A_600.csv')
    print("Início dos dados Brutos:")
    print(dadosBrutosCSV.head())
    # 2º precisamos limpar esses dados e tratar cada um de acordo com o seu contexto
    # em caso de necessidade
    ativaLimpaDadosBrutos = True
    print("Deseja aplicar a limpeza nos dados? [S/n]")
    resposta = input()
    if resposta.lower()=='s':
        ativaLimpaDadosBrutos = True
    elif resposta.lower()=='n':
        ativaLimpaDadosBrutos = False
    else:
        print("Resposta inválida! Encerrando execução...")
        exit()

    if ativaLimpaDadosBrutos:
        dadosLimposCSV = limpaDadosBrutos(dadosBrutosCSV)
    else:# Se não precisa de limpeza, os dados brutos já são os limpos
        dadosLimposCSV = dadosBrutosCSV

    # 3º precisamos trocar as classes para que sejam rótulos numéricos
    # Troca a classe para valores numéricos
    dadosLimposCSV.sentiment = dadosLimposCSV['sentiment'].map({'positiva': 1, 'negativa': -1, 'neutra': 0})
    print("\nAgora os sentimentos estão como classes numéricas:\n")
    print(dadosLimposCSV.head())

    # 4º precisamos setar quais são as palavras de parada a serem utilizadas e gerar
    # o vetorTFIDF e a matrizTFIDF
    matrizTextoTFIDF = construirMatrizAtributos(dadosLimposCSV)

    # 5º precisamos converter a matriz para o tipo de dado que o classificador suporta
    # Aqui usamos a matrizModificada para ter um backup e usar a modificada para os testes
    matrizModificada = pd.DataFrame(matrizTextoTFIDF.toarray())  # não perder a original, criamos uma matriz densa dataframe
    print("Matriz modificada completa:\n", matrizModificada)
    # Converte a matrizModificada para esparsa que é o que o classificador precisa
    matrizModificada = scipy.sparse.csr_matrix(matrizTextoTFIDF)

    # 6º Definição dos conjuntos de treino e teste - Hold-out ou k-fold.
    print("Digite a opção de treinamento/teste:")
    print("[1] - Hold-out (70%/30%)")
    print("[2] - K-fold (K=10) (Usa o classificador ME)")
    opcaoModeloTeste = int(input())
    if opcaoModeloTeste<1 or opcaoModeloTeste>2:
        print("Opção inválida! Encerrando execução...")
        exit()
    elif opcaoModeloTeste == 1:
        x_treino,x_teste,y_treino,y_teste = separaDadoTreinoTesteHoldOut(matrizModificada,dadosLimposCSV,estadoInicial)
    elif opcaoModeloTeste == 2:
        separaDadoTreinoTesteKFold(matrizModificada,dadosLimposCSV)

    # 7º definição dos classificadores por meio do menu
    # Se o Hold-out foi usado, podemos chamar os classificadores
    if opcaoModeloTeste == 1:
        modeloClassificadorEscolhido = escolheClassificadorHoldOut(x_treino,x_teste,y_treino,y_teste)
    elif opcaoModeloTeste == 2:
        print("Execução finalizada!")
        exit()

    # Teste com dados novos
    #dadosNovosLimpos = pd.read_csv('comentarios/base_teste_pequeno_sem_rotulo.csv')
    #print(dadosNovosLimpos)
    #regLog = LogisticRegression()
    #regLog = regLog.fit(x_treino, y_treino)
    # save the model to disk
    #filename = 'finalized_model.sav'
    #pickle.dump(regLog, open(filename, 'wb'))
    # load the model from disk
    #loaded_model = pickle.load(open(filename, 'rb'))

    # new instances where we do not know the answer
    #X_novo_teste = construirMatrizAtributos(dadosNovosLimpos)
    #print("X_novo_teste:\n",X_novo_teste)
    # make a prediction
    # Y_novo_teste = regLog.predict(X_novo_teste)
    #print("Y_novo_teste:\n",Y_novo_teste)
    # show the inputs and predicted outputs
    #for i in range(len(X_novo_teste)):
    #    print("X=%s, Predicted=%s" % (X_novo_teste[i], Y_novo_teste[i]))


# Pressione o botão verde ou shift+F10 ou somente F10 para executar .
if __name__ == '__main__':
    main()

