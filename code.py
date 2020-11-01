#%%
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import numpy as np
from collections import Counter
from tqdm import tqdm
from datetime import datetime,timezone,timedelta

def timestamp():
    dt1 = datetime.utcnow().replace(tzinfo=timezone.utc)
    dt2 = dt1.astimezone(timezone(timedelta(hours=8))) # 轉換時區 -> 東八區
    print(dt2)

def file_iter(_type):
    if _type == "q":
        for name in q_list:
            with open(query_path+name+'.txt') as f:
                yield f.readline()
    elif _type == "d":
        for name in d_list:
            with open(doc_path+name+'.txt') as f:
                yield f.readline()
#%%
B = 0.75
K1 = 4
K3 = 1000

#%%
timestamp()
doc_path = "../HW1 Vector Space Model/data/docs/"
query_path = "../HW1 Vector Space Model/data/queries/"

d_list = []
with open('../HW1 Vector Space Model/data/doc_list.txt', 'r') as d_list_file:
    for line in d_list_file:
        line = line.replace("\n", "")
        d_list += [line]

q_list = []
with open('../HW1 Vector Space Model/data/query_list.txt', 'r') as q_list_file:
    for line in q_list_file:
        line = line.replace("\n", "")
        q_list += [line]
#%%
#tf
list_tf = []
doc_list = []
query_list = []
for txt in tqdm(file_iter("q")):
    list_tf += [Counter(txt.split())]
    query_list += [txt]

for txt in tqdm(file_iter("d")):
    doc_list += [txt]

#%%
df_q_tf = pd.DataFrame(list_tf)
df_q_tf = df_q_tf.fillna(0)
np_q_tf = np.array(df_q_tf)
voc = list(df_q_tf.columns)
# df_q_tf

#%%
doc_len = []
np_d_tf = np.zeros([len(d_list), len(voc)])
for i, doc in tqdm(enumerate(doc_list)):
    doc_len += [len(doc)]
    for j, w in enumerate(voc):
        np_d_tf[i][j] += doc.count(w)
del df_q_tf
doc_avg_len = np.array(doc_len).mean()
np_d_tf

#%%
# df
np_df = np.count_nonzero(np_d_tf, axis=0) #n_i
np_df

#%%
# sim_array
sim_array = np.zeros([len(q_list), len(d_list)])
for q in tqdm(range(len(q_list))):
    for d in range(len(d_list)):
        for w in query_list[q].split():
            i = voc.index(w)
            d_tf = (K1+1)*np_d_tf[d][i] / (K1*((1-B) + B*doc_len[d]/doc_avg_len) + np_d_tf[d][i]) #####
            # q_tf = (K3+1)*np_q_tf[q][i] / (K3+np_q_tf[q][i])  ###
            idf = np.log(1+(len(d_list)-np_df[i]+0.5) / (np_df[i]+0.5))
            sim_array[q][d] += d_tf * np.power(idf, 2)
sim_array

#%%
from rank_bm25 import BM25Okapi
tokenized_corpus = [doc.split(" ") for doc in doc_list]

bm25 = BM25Okapi(tokenized_corpus, k1=4)
sim_array = [bm25.get_scores(tokenized_query.split()) for tokenized_query in query_list]
# sim_array

#%%
# output
with open('result.csv', 'w') as output_file:
    output_file.write("Query,RetrievedDocuments\n")
    for i, q_id in tqdm(enumerate(q_list)):
        output_file.write(q_id+',')
        sorted = np.argsort(sim_array[i])
        sorted = np.flip(sorted)
        for _, j in enumerate(sorted):
            output_file.write(d_list[j]+' ')
        output_file.write('\n')
timestamp()