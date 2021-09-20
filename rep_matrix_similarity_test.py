import pandas as pd
import numpy as np
import operator
from functools import reduce

#This file compute the top 5 similar movies from thrir embedding

def cosine_similarity(a, b):
    #print("a:{}".format(a))
    nominator = np.dot(a, b)
    
    a_norm = np.sqrt(np.sum(a**2))
    b_norm = np.sqrt(np.sum(b**2))
    
    denominator = a_norm * b_norm
    
    cosine_similarity = nominator / denominator
    
    return cosine_similarity

#read movie rep file
df = pd.read_csv('movie_rep_matrix_I_A.csv')
df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
df = df.drop_duplicates(subset=['movie_name']).reset_index()
#print(df)

#construct similarity matrix
sim_list = []
tmp = []
for i in range(0,100):
    if(df['rep'][i] != ''):
        for j in range(0,100):
            if(df['rep'][j] != ''):
                strI= df['rep'][i].replace('[[  ','').replace('[[ ','').replace('[[','').replace('  ]]','').replace(' ]]','').replace(']]','').replace('\n','').replace("  ",' ').replace(' ',',').replace(',,',',').replace(',,',',')
                strJ= df['rep'][j].replace('[[  ','').replace('[[ ','').replace('[[','').replace('  ]]','').replace(' ]]','').replace(']]','').replace('\n','').replace("  ",' ').replace(' ',',').replace(',,',',').replace(',,',',')
                #print(strI)
                strI = [float(item) for item in strI.split(',')]
                strJ = [float(item) for item in strJ.split(',')]
                tmp.append(cosine_similarity(np.array(strI, dtype='float64'),np.array(strJ, dtype='float64')))
            else:
                print("no embedding in j:{}".format(df['rep'][j]))
        sim_list.append(tmp)
        tmp = []
    else:
        print("no embedding in i:{}".format(df['rep'][i]))

#result to dataframe
sim_list_df = pd.DataFrame(sim_list,columns=df['movie_name'][0:100],index=df['movie_name'][0:100])

# #add satistical information
second_max = pd.DataFrame(np.sort(sim_list_df.values)[:,-2:], columns=['second_max','largest'])
second_max_df = second_max['second_max'].tolist()
columns =  sim_list_df.columns
index = sim_list_df.index
extracted = []
for i in range(0,100):
    for item in sim_list_df[columns[i]]:
        if(second_max_df[i]==item):
            extracted.append(index[sim_list_df[columns[i]]==item].values)

sim_list_df['second_max_key']=extracted
print(sim_list_df)
sim_list_df.to_csv('movie_rep_similarity_test_I_A.csv')