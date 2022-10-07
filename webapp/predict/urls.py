from django.urls import path

from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('runQuery', views.runQuery, name='runQuery'),
    path('runExampleQuery', views.runExampleQuery, name='runExampleQuery'),
    path('getAutoCompletes', views.getAutoCompletes, name='getAutoCompletes'),
]