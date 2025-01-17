import numpy as np
from collections import Counter
from tqdm import tqdm
from datetime import datetime,timezone,timedelta
from functools import reduce
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

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

parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
parser.add_argument("-B", default=0.75, type=float ,help=" ")
parser.add_argument("-K1", default=3.5, type=float ,help=" ")
parser.add_argument("-K3", default=1000, type=float ,help=" ")
parser.add_argument("-use_q_tf", default='F', type=str, choices=['T','F'], help="whether to use query's tf; will deactivate K3")
args = parser.parse_args()
str2bool = {'T':True, 'F':False}
args.use_q_tf = str2bool[args.use_q_tf]

B = args.B
K1 = args.K1
K3 = args.K3

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
# tf
list_q_tf = []
list_d_tf = []
doc_list = []
query_list = []
for txt in tqdm(file_iter("q")):
    list_q_tf += [Counter(txt.split())]
    query_list += [txt]

for txt in tqdm(file_iter("d")):
    list_d_tf += [Counter(txt.split())]
    doc_list += [txt]

# voc
voc = reduce(set.union, map(set, map(dict.keys, list_q_tf)))
voc = list(voc)

# df
doc_len = []
np_df = dict.fromkeys(voc, 0)
for i, doc in tqdm(enumerate(doc_list)):
    doc_len += [len(doc)]
    for w in voc:   
        np_df[w] += 1 if list_d_tf[i][w] > 0 else 0
doc_avg_len = np.array(doc_len).mean()

# sim_array
sim_array = np.zeros([len(q_list), len(d_list)])
for q in tqdm(range(len(q_list))):
    for d in range(len(d_list)):
        for w in query_list[q].split():
            d_tf = (K1+1)*list_d_tf[d][w] / (K1*((1-B) + B*doc_len[d]/doc_avg_len) + list_d_tf[d][w])
            if args.use_q_tf:
                q_tf = (K3+1)*list_q_tf[q][w] / (K3+list_q_tf[q][w])
            else:
                q_tf = 1
            idf = np.log(1+(len(d_list)-np_df[w]+0.5) / (np_df[w]+0.5))
            idf = np.power(idf, 2)
            sim_array[q][d] += d_tf * q_tf * idf

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