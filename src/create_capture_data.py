import os 
import json 
import pandas as pd
import numpy as np
import sys 

phase_id="phase-3"
prob_id="prob-2"
data_path="data/captured_data/"+phase_id+"/"+prob_id+"/"
save_path="data/"+"data_captured_"+phase_id+"_"+prob_id+".csv"
list_df=[]
for file in os.listdir(data_path):
    if file.endswith(".json"):
        print(file)
        file_path=data_path+file
        with open(file_path,"r") as f:
            data=json.load(f)
        rows=data["rows"]
        columns=data["columns"]
        df=pd.DataFrame(rows,columns=columns)
        
        list_df.append(df)
df=pd.concat(list_df)
#reindex columns feature1 to feature41
df=df.reindex(columns=["feature"+str(i) for i in range(1,42)])
#drop duplicates
df=df.drop_duplicates()
#get 50000 samples
#df=df.sample(n=100000,random_state=1)

df.to_csv(save_path,index=False)
# model = classifier.fit(X_train, y_train, batch_size=128, epochs=200, verbose=1, validation_data=(X_test, y_test))
# classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'],learning_rate=0.01)
# #set Adam optimizer with learning rate 0.01
# import tensorflow as tf
# opt = tf.keras.optimizers.Adam(learning_rate=0.01)