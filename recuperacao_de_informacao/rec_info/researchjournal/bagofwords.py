import pandas as pd
import math
import numpy as np
import openpyxl


DATASET = 'static/arquivos/Dataset.xlsx'
TOKENS = 'static/arquivos/tokens.xlsx'
VOCABSENDDOCS = 'static/arquivos/vocabSend_docs.xlsx'
PLACESDOCS = 'static/arquivos/place_words.xlsx'



# Pre-processamento consulta
def pre_process_query(consultation) :
  #deixa todas as letras minúsculas
  consultation = consultation[0].lower()
  tokens = consultation.split()
  return tokens



# função para calcular a quantidade de vezes que uma palavra do vocabulário aparece no documento

def dic_of_count(vocab, doc) :
  corpus = []

  for i in range(0, len(doc)) :
    dic = dict.fromkeys(vocab, 0)
    for word in vocab:
      if word in doc[i] :
        dic[word] += 1
    corpus.append(dic.copy())
  return corpus


# Função para calcular TF (Quantidade de vezes que uma palavra ocorre no documento) / (Quantidade de palavras do documento)

def calcTF(dic_of_count, docs):
  corpus = []
  for i in range(0, len(docs)) :
    tf_dic = {}
    num_words_doc = len(docs[i])
    for word, count in dic_of_count[i].items():
      tf_dic[word] = count/float(num_words_doc)
    corpus.append(tf_dic.copy())
  return corpus # vetor de dicionários


# Função para calcular idf
def calcIDF(list_of_docs,vocab) :
  idf_dic = {}
  # tamanho da lista de docs (quantidade de documentos)
  N = len(list_of_docs)
  
  # Pega as palavras da primeira lista
  for word in vocab:
    num_docs_appears = 0
    # Conta quantas vezes uma palavra do vocabulário aparece na lista de documentos
    for doc in list_of_docs :
      if word in doc :
        num_docs_appears+=1
    # Calculo do idf para essa palavra
    if num_docs_appears > 0 :
      idf_dic[word] = N/(num_docs_appears)
    else :
      idf_dic[word] = 0

  return (idf_dic)


# Calculando o peso com w = tf x idf

def calcTFIDF(tf_bow, idfs) :
  tfidf = {}
  corpus = []
  for doc in tf_bow :
    for word  in idfs:
      if word in doc :
        tf = doc[word]
      else:
        tf = 0
      idf = idfs[word]
      tfidf[word] = tf*idf
    corpus.append(tfidf.copy())
  return(corpus)


# Função para calcular a similaridade

def sim(seach,doc) :
  sum = 0
  sum_seach = 0
  sum_doc = 0

  for i in seach:
    # Calcula o valor do númerador
    sum = (seach[i] * doc[i]) + sum
    # Calcula o valor que vai para a raiz de seach no denominado
    sum_seach = (seach[i]**2) + sum_seach
    # Calcula o valor que vai para a raiz de doc no denominado
    sum_doc = (doc[i]**2) + sum_doc
  # Calcula o valor do denominador
  raiz_sum_seach = math.sqrt(sum_seach)
  raiz_sum_doc = math.sqrt(sum_doc)
  denom = raiz_sum_seach * raiz_sum_doc
  similarity = sum / denom
  return similarity


# Função que retorna os documentos ordenados
def rank(dic_docs,dic_seach) :
  similarity_docs_with_seach = []
  recovered = []
  index = []
  threshold = 0
  for doc in dic_docs :
    value = sim(dic_seach,doc)
    similarity_docs_with_seach.append(value)
  threshold = (np.sum(similarity_docs_with_seach))/len(similarity_docs_with_seach)
  for i in range(0,len(similarity_docs_with_seach)):
    if similarity_docs_with_seach[i] > threshold:
      recovered.append([similarity_docs_with_seach[i],i])
  if len(recovered) > 1 :
    recovered_sort = sorted(recovered,reverse=True)
    for vector in recovered_sort :
      index.append(vector[1])
  else:
    index.append(recovered[0][1])

  return index


# Função que retorna os documentos ordenados da pesquisa

def list_docs(index_docs,dataset):
  result = []
  data = dataset.values.tolist()
  for i in index_docs :
    result.append(data[i])
  return result


# Função que recebe um caminho, lê o csv e transforma em lista vocabulário
def changecsvtolist2(path):
  data = pd.read_excel(path)
  data = data.values.tolist()
  data_aux = []
  for i in range(0,len(data)):
    # print(data[i][0])
    data_aux.append(data[i][0])
  return data_aux


# Função que recebe um caminho, lê o csv e transforma em lista
def changecsvtolist(path):
  data = pd.read_excel(path)
  data = data.values.tolist()
  return data



def metrics(path,tokens,vocab,result,quant_docs):


  places_w = changecsvtolist(path)
  

  relevante = []
  dic  =  dict.fromkeys(vocab,[])

  vecs = []
  for item in places_w :
    del item[0]
  
  
  for item in places_w :
    new_vec = []
    aux = []
    new_vec = [not math.isnan(number) for number in item]
    for j in range(0,len(new_vec)) :
      if new_vec[j] :
        aux.append(int(item[j]))
    vecs.append(aux)

  places_w =  vecs.copy()

  i = 0
  for word in vocab:
    dic[word] = places_w[i]
    i = i + 1
  

  # Calcular quantidade de documentos relevantes

  relevante_rec = []
  relevante = []

  for word in tokens: # tokens da query
    if word in vocab: # se estiver no vocabulário
      vec = dic[word] # recebe  o vetor de index de docs relevantes
      for item in vec:
        if item not in relevante:
          relevante.append(item)
      for index in result : # index dos docs recuperados
        if index in vec: # se o documento recuperado estiver como documento relevante
          if index not in relevante_rec:
            relevante_rec.append(index)

  precision = len(relevante_rec) / quant_docs
  recall = len(relevante_rec) / len(relevante)

  return precision,recall




def main(consultation):
  # pré-processa os documentos
  dataset = pd.read_excel(DATASET)
  #carregando os tokens
  tokens_docs = changecsvtolist(TOKENS)
  # cria o vocabulário com base nos documentos
  vocabSend_docs = changecsvtolist2(VOCABSENDDOCS)

  # dicionario de palavras 
  count_words_docs = dic_of_count(vocabSend_docs,tokens_docs)
  # calcular tf
  tf_docs = calcTF(count_words_docs,tokens_docs)
  # calcular idf
  idf_docs = calcIDF(tokens_docs,vocabSend_docs)

  # Calcular o peso
  weight_docs = calcTFIDF(tf_docs,idf_docs)

  # Processamento e retorna da consulta

  tokens_seach = pre_process_query([consultation])
  # tokens_seach 
  count_words_seach = dic_of_count(vocabSend_docs,[tokens_seach])
  # count_words_seach
  tf_seach = calcTF(count_words_seach,[tokens_seach])
  # tf_seach
  idf_seach = calcIDF([tokens_seach],vocabSend_docs)
  # idf_seach
  weight_seach = calcTFIDF(tf_seach,idf_seach)
  # weight_seach
  result = rank(weight_docs,weight_seach[0])
  # result
  docs_query = list_docs(result,dataset)

  precision,recall = metrics(PLACESDOCS,tokens_seach,vocabSend_docs,result,len(result))
  
  docs = []
  for doc in docs_query :
    docs.append(doc[0])
    
  return precision,recall,docs
