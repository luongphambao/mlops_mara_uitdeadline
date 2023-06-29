import json
from locust import HttpUser, TaskSet, task, between
import os 
import pandas as pd 
import random 
import json 
class Prob1(TaskSet):
    @task
    def predict(self):
        headers = {'content-type': 'application/json'}
        
        id_data = os.listdir("data/curl/phase-1/prob-1/")
        index=random.randint(0,len(id_data)-1)
        data=json.load(open("data/curl/phase-1/prob-1/"+id_data[index]))
        
        self.client.post("phase-1/prob-1/predict",json=data,headers=headers)
class Prob2(TaskSet):
    @task
    def predict(self):
        headers = {'content-type': 'application/json'}
        
        id_data = os.listdir("data/curl/phase-1/prob-2/")
        index=random.randint(0,len(id_data)-1)
        data=json.load(open("data/curl/phase-1/prob-2/"+id_data[index]))
        
        self.client.post("phase-1/prob-2/predict",json=data,headers=headers)

class LoadTest(HttpUser):
    tasks = [Prob1,Prob2]
    wait_time = between(0,1)
    host = "http://127.0.0.1:8000/"
    stop_timeout = 10

