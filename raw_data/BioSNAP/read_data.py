import csv
import numpy as np

import pandas as pd
from tqdm import tqdm
import json
file = pd.read_csv('ChChSe-Decagon_polypharmacy.csv')
df = pd.DataFrame(file)

drugs = {}
side_effects = {}
edges = []

with open('ChChSe-Decagon_polypharmacy.csv', 'r') as csvfile:
    spamreader = csv.reader(csvfile)
    for row in spamreader:
        #print(row)# type of row: list
        x = row[0]
        y = row[1]
        z = row[2]
        drugs[x] = 1
        drugs[y] = 1
        if z not in side_effects:
            side_effects[z] = 1
        else:
            side_effects[z] += 1
        edges.append([x, y, z])

side_effect_filter = {}
edge_filter = []
drug_filter = {}
for s in side_effects:
    if 941 <= side_effects[s] <= 1967:
        side_effect_filter[s] = 1
    else:
        pass
for x,y,e in tqdm(edges):
    if e in side_effect_filter:
        edge_filter.append([x, y, e])
        drug_filter[x] = 1
        drug_filter[y] = 1

#1967 941
# tfeat = {}
# with open('drug_tfeat.txt', 'r') as f:
#     for line in f:
#         line = line.strip().split(',')
#         idx = line[0]
#         feat = list(map(float, line[1:]))
#         tfeat[idx] = feat
# print(tfeat)
# json.dump(tfeat, open('tfeat.json','w'))
# x = json.load(open('../ddi/node2id.json', 'r'))
# i = 0
# tfeats = np.zeros([len(x), 50])
# for y in x:
#     if y in tfeat:
#         #print(idx, y, tfeat[y])
#         idx = x[y]
#         tfeats[idx] = tfeat[y]
#         i+=1
# print(i, len(x))
#print(x,y,z,'\n')
print(len(drugs), len(side_effects), len(edges))
print(len(drug_filter), len(side_effect_filter), len(edge_filter))

z = sorted(list(side_effects.values()))
print(z[::-1][600:800])
print([x for x in drugs if x not in drug_filter])

with open('drugnames.txt', 'w') as f:
    for w in drug_filter:
        drugname = w[3:]
        f.write(drugname)
        f.write('\n')
with open('edges.txt', 'w') as f:
    for x, y, z in edge_filter:
        f.write('%s, %s, %s'%(x, y, z))
        f.write('\n')
with open('edges.json', 'w') as f:
    json.dump(list(side_effect_filter.keys()), f)

cid2smiles = {}
with open('cid2smiles.txt', 'r') as f:
    for lines in f:
        x = lines.strip().split('\t')
        # print(x)
        if len(x) == 2:
            cid, smiles = lines.strip().split('\t')
            cid2smiles[cid] = smiles
            # db2cid[db] = cid 
        else:
            cid =  lines.strip().split('\t')[0]
            # cid2smiles[cid] = None
print(len(cid2smiles), len([x for x in cid2smiles if cid2smiles[x] is not None]))

cid2db = {}
db2cid = {}
with open('cid2db.txt', 'r') as f:
    for lines in f:
        x = lines.strip().split('\t')
        if len(x) == 2:
            cid, db = lines.strip().split('\t')
            if cid in cid2smiles:
                cid2db[cid] = db
                db2cid[db] = cid 
        else:
            cid =  lines.strip().split('\t')[0]
            if cid in cid2smiles:
                cid2db[cid] = None

## calculate the matching
kbdrug = {}
with open('entity.json', 'r') as f:
    entity = json.load(f)
    for x in entity:
        if 'Compound' in x:
            idx = x.split('::')[-1]
            kbdrug[idx] = 1
src = list(db2cid.keys())
tgt = list(kbdrug.keys())
print(len(src), len(list(cid2db.keys())))
print(len([x for x in src if x in tgt]))

drug2id = {}
cid2id = {}
id2drug = {} # schema: id, cid, smiles, 
relations = {} # relation: id
id2relation = {} # relation: id
edges = []
edges_pair = {}
i = 0
j = 0
for x, y, z in edge_filter:
    #print(x,y,z)
    x_, y_ = str(int(x[3:])), str(int(y[3:]))
    if x_ not in cid2db or y_ not in cid2db:
        pass
    else:
        if x_ not in cid2id:
            cid2id[x_] = i 
            if x_ in cid2db:
                dbid = cid2db[x_]
                drug2id[dbid] = i
            else:
                dbid = None
            id2drug[i] = {'cid':x, 'db':dbid, 'smiles': cid2smiles[x_]}
            i += 1
        if y_ not in cid2id:
            cid2id[y_] = i 
            if y_ in cid2db:
                dbid = cid2db[y_]
                drug2id[dbid] = i
            else:
                dbid = None
            id2drug[i] = {'cid': y, 'db': dbid, 'smiles': cid2smiles[y_]}
            i += 1
        if z not in relations:
            relations[z] = j
            id2relation[j] = z
            j += 1
            edge_id = relations[z]
        else:
            edge_id = relations[z]
        edges.append([cid2id[x_], cid2id[y_], edge_id])
        if str(cid2id[x_])+','+str(cid2id[y_]) in edges_pair:
            edges_pair[str(cid2id[x_])+','+str(cid2id[y_])] += 1
        else:
            edges_pair[str(cid2id[x_])+','+str(cid2id[y_])] = 1
print(len(edges), len(edges_pair), np.mean(list(edges_pair.values())))
#print(edges, len(edges_pair))
print(len(cid2id))
with open('f_edges.txt', 'w') as f:
    for x, y, z in edges:
        f.write('%s, %s, %s'%(x, y, z))
        f.write('\n')
with open('id2drug.json', 'w') as f:
    json.dump(id2drug, f)
with open('drug2id.json', 'w') as f:
    json.dump(drug2id, f)
with open('cid2id.json', 'w') as f:
    json.dump(cid2id, f)
with open('id2relation.json', 'w') as f:
    json.dump(id2relation, f)
with open('relations.json', 'w') as f:
    json.dump(relations, f)
with open('edges2relation.json', 'w') as f:
    json.dump(edges_pair, f)
assert 0

assert 0
for i in tqdm(range(len(df))):
    document = df[i:i+1]
    x = document['# STITCH 1'][i]
    y = document['STITCH 2'][i]
    z = document['Polypharmacy Side Effect'][i]
    drugs[x] = 1
    drugs[y] = 1
    side_effects[z] = 1
    edges.append([x, y, z])
    #print(x,y,z,'\n')
print(len(drugs), len(side_effects))
with open('drugname.txt', 'w') as f:
    for w in drugs:
        drugname = w[3:]
        f.write(drugname)
        f.write('\n')