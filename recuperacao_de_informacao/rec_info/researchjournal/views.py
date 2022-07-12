from django.shortcuts import render
from researchjournal import bagofwords as bof
from django.utils.datastructures import MultiValueDictKeyError
from researchjournal import doc2vec as d2v

def index(request) :
    return render(request, "index.html")

def doc2vec(request) :
    return render(request, "index_doc2vec.html")



# Bag of words
def pesquisar(request) :
    reseach = ' '
    try: 
        reseach = request.POST['query']
    except MultiValueDictKeyError:
        print(MultiValueDictKeyError)

    data = {}
    data['item'] = reseach
    data['precision'],data['recall'],data['dados'] = bof.main(reseach)
    return render(request, "search.html",data)


# doc2vec
def pesquisar2(request) : 
    reseach2 = ' '
    try: 
        reseach2 = request.POST['query2']
    except MultiValueDictKeyError:
        print(MultiValueDictKeyError)

    data = {}
    data['item'] = reseach2
    data['precision'],data['recall'],data['dados'] = d2v.main(reseach2)
    
    return render(request, "search_doc2vec.html",data)