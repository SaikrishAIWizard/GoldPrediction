from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='home'),
    path('signup/', views.signup_view, name='signup'),
    path('login/', views.login_view, name='login'),
    path('logout/', views.logout_view, name='logout'),
    path('train/', views.train_model, name='train_model'),
    path('forecast/', views.forecast, name='forecast'),
    path('graphs/', views.model_graphs, name='model_graphs'),
    path('Back_Testing/', views.Back_Testing, name='Back_Testing')  
]
