from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import pandas as pd
import numpy as np
import math

DATASET = 'static/arquivos/Dataset.xlsx'
TOKENS = 'static/arquivos/tokens_docs_2vec.xlsx'
MODEL = "static/arquivos/d2v.model"
VOCABSENDDOCS = 'static/arquivos/vocabSend_docs.xlsx'
PLACESDOCS = 'static/arquivos/place_words.xlsx'



# Pre-processamento consulta
def pre_process_query(consultation) :
  #deixa todas as letras minúsculas
  consultation = consultation[0].lower()
  tokens = consultation.split()
  return tokens


# Função para calculo da similaridade
def similarity(modelSave,tokens_docs,tokens_seach):
  seems = []
  i = 0
  for doc in tokens_docs:
    seems_cos = modelSave.similarity_unseen_docs(tokens_seach,doc)
    seems.append([seems_cos,i])
    i+=1
  seems = sorted(seems,reverse=True)

  return seems


# Função que retorna os documentos encontrados
def finds(seems,dataset):
  index = []
  i = 0
  docs_find = []
  data = dataset.values.tolist()
  for cos_vec in seems :
    if i < 20 :
      index.append(cos_vec[1])
      i = i +1
  for i in index: 
    docs_find.append(data[i][0])

  return index,docs_find



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


# Função que recebe um caminho, lê o csv e transforma em lista vocabulário
def changecsvtolist2(path):
  data = pd.read_excel(path)
  data = data.values.tolist()
  data_aux = []
  for i in range(0,len(data)):
    data_aux.append(data[i][0])
  return data_aux




def main(query):

  # Carregar o dataset 
  dataset = pd.read_excel(DATASET)
  tokens_docs = changecsvtolist(TOKENS)
  tagged_data = [TaggedDocument(words=_d, tags=[str(i)]) for i, _d in enumerate(tokens_docs)]

  # tirar os nan
  newList = []
  for item  in tokens_docs:
    newlist = [x for x in item if pd.isnull(x) == False]
    newList.append(newlist)

  tokens_docs = newList

  max_epochs=100
  alpha=0.001

  model = Doc2Vec(
    tagged_data, 
    vector_size=3000, 
    alpha=0.001,
    window=2, 
    epochs=max_epochs,
    min_alpha=0.00025,
    min_count=1, 
  )

  model.build_vocab(tagged_data)

  # treinamento
  # for epoch in range(max_epochs):
  #   model.train(
  #     corpus_iterable=tagged_data,
  #     epochs=model.epochs,
  #     start_alpha=model.alpha,
  #     total_examples=170,
  #     end_alpha=0.0001
  #   )

  # # Salvar o modelo
  # model.save(MODEL)

  # carregar o modelo que já foi treinado
  modelSave = Doc2Vec.load(MODEL)

  # Consulta
  
  # tokenização
  tokens_seach = pre_process_query([query])
  # tokens_seach

  # Resultado da consulta
  seems = similarity(modelSave,tokens_docs,tokens_seach)
  index,seems_find = finds(seems,dataset)

  # vocabulário com base nos documentos
  vocabSend_docs = changecsvtolist2(VOCABSENDDOCS)

  precision,recall = metrics(PLACESDOCS,tokens_seach,vocabSend_docs,index,len(index))

  return precision,recall,seems_find


