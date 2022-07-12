from django.urls import path
from . import views

urlpatterns = [
    path('',views.index, name='index'),
    path('doc2vec',views.doc2vec, name='doc2vec'),
    path('pesquisar', views.pesquisar, name='pesquisar'),
    path('pesquisar2', views.pesquisar2, name='pesquisar2'),
]