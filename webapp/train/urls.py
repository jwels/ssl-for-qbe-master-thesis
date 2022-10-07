from django.urls import path

from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('dropModel', views.dropModel, name='dropModel'),
    path('selectModel', views.selectModel, name='selectModel'),
    path('newModel', views.newModel, name='newModel'),
    path('createNewModel', views.createNewModel, name='createNewModel'),
    path('debugView', views.debugView, name='debugView')
]