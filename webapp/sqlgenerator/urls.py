from django.urls import path

from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('generate', views.generate_queries_view, name='gensql'),
    path('gentraindata', views.generate_training_data_view, name='gentraindata'),
    path('dropall', views.drop_all, name='dropall'),
    path('dropalltrain', views.drop_all_trainingdata, name='dropalltrain')
]