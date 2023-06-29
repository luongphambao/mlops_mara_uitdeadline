import pandas as pd 
import numpy as np 


csv_path1="data/data_phase1_prob1_model-1.csv"
csv_path2="data/data_phase1_prob1_model-2.csv"
csv_path3="data/data_phase1_prob1_model-3.csv"
csv_path4="data/data_phase1_prob1_model-4.csv"


df1=pd.read_csv(csv_path1)
df2=pd.read_csv(csv_path2)
df3=pd.read_csv(csv_path3)
df4=pd.read_csv(csv_path4)

count=0
df=[]
for i in range(len(df1)):
    result1=df1["label_model"][i]
    result2=df2["label_model"][i]
    result3=df3["label_model"][i]
    result4=df4["label_model"][i]

    prob1=df1["prob"][i]
    prob2=df2["prob"][i]
    prob3=df3["prob"][i]
    prob4=df4["prob"][i]

    #check 6 models have the same result
    
    prob=(prob1+prob2+prob3+prob4)/4
    #get 20 features and prob
    row=df1.iloc[i]
    #replace value prob
    row["prob"]=prob
    min_prob=min(prob1,prob2,prob3,prob4)
    max_prob=max(prob1,prob2,prob3,prob4)
    abs_prob1=abs(prob1-prob) 
    if result1==result2 and result2==result3 and result3==result4 and abs_prob1<0.05 and min_prob>=0.9:
        
        print(min_prob,max_prob)
        
    else:
        if row["label_model"]==1:
            row["label_model"]=0
        else:
            row["label_model"]=1
    df.append(row)
df=pd.DataFrame(df)
df.to_csv("data/data_phase1_prob1_model-94.csv",index=False)


print(count)