import numpy as np
from scipy.sparse import csc_matrix

def process_files_ddi(files, BKG_file,keeptrainone = False):
    entity2id = {}
    relation2id = {}

    triplets = {}
    kg_triple = []
    ent = 0
    rel = 0

    for file_type, file_path in files.items():
        data = []
        file_data = np.loadtxt(file_path)
        for triplet in file_data:
            triplet[0], triplet[1], triplet[2] = int(triplet[0]), int(triplet[1]), int(triplet[2])
            if triplet[0] not in entity2id:
                entity2id[triplet[0]] = triplet[0]
            if triplet[1] not in entity2id:
                entity2id[triplet[1]] = triplet[1]
            if  triplet[2] not in relation2id:
                if keeptrainone:
                    triplet[2] = 0
                    relation2id[triplet[2]] = 0
                    rel = 1
                else:
                    relation2id[triplet[2]] = triplet[2]
                    rel += 1

            # Save the triplets corresponding to only the known relations
            if triplet[2] in relation2id:
                data.append([entity2id[triplet[0]], entity2id[triplet[1]], relation2id[triplet[2]]])

        triplets[file_type] = np.array(data)
        
    triplet_kg = np.loadtxt(BKG_file)
    for (h, t, r) in triplet_kg:
        h, t, r = int(h), int(t), int(r)
        if h not in entity2id:
            entity2id[h] = h
        if t not in entity2id:
            entity2id[t] = t 
        # same id within train/valid/test and BKG_file does not mean same relation
        if rel+r not in relation2id:
            relation2id[rel+r] = rel+r
        kg_triple.append([h, t, r])
    kg_triple = np.array(kg_triple)
    id2entity = {v: k for k, v in entity2id.items()}
    id2relation = {v: k for k, v in relation2id.items()}
    # Construct the list of adjacency matrix each corresponding to each relation. Note that this is constructed from the train data and BKG data.
    adj_list = []
    for i in range(rel):
        idx = np.argwhere(triplets['train'][:, 2] == i)
        adj_list.append(csc_matrix((np.ones(len(idx), dtype=np.uint8), (triplets['train'][:, 0][idx].squeeze(1), triplets['train'][:, 1][idx].squeeze(1))), shape=(len(entity2id), len(entity2id))))
    for i in range(rel, len(relation2id)):
        idx = np.argwhere(kg_triple[:, 2] == i-rel)
        adj_list.append(csc_matrix((np.ones(len(idx), dtype=np.uint8), (kg_triple[:, 0][idx].squeeze(1), kg_triple[:, 1][idx].squeeze(1))), shape=(len(entity2id), len(entity2id))))
    return adj_list, triplets, entity2id, relation2id, id2entity, id2relation, rel

def process_files_decagon(files, triple_file, keeptrainone = False):
    entity2id = {}
    relation2id = {}

    triplets = {}
    triplets_mr = {}
    polarity_mr = {}
    kg_triple = []
    triplets_train = []
    rel = 0

    for file_type, file_path in files.items():
        data = []
        data_mr = []
        data_pol = []
        with open(file_path, 'r') as f:
            for lines in f:
                h, t, r, p = lines.strip().split('\t')
                h, t = int(h), int(t)
                p = int(p) # pos/neg edge
                list_r_onehot = list(map(int, r.split(',')))
                list_r = [0] if keeptrainone else [i for i, _ in enumerate(list_r_onehot) if _ == 1]  
                for s in list_r:
                    triplet = [h,t,s]
                    triplet[0], triplet[1], triplet[2] = int(triplet[0]), int(triplet[1]), int(triplet[2])
                    if triplet[0] not in entity2id:
                        entity2id[triplet[0]] = triplet[0]
                    if triplet[1] not in entity2id:
                        entity2id[triplet[1]] = triplet[1]
                    if triplet[2] not in relation2id:
                        if keeptrainone:
                            triplet[2] = 0
                            relation2id[triplet[2]] = 0
                            rel = 1
                        else:
                            relation2id[triplet[2]] = triplet[2]
                            rel += 1
                    # Save the triplets corresponding to only the known relations
                    if triplet[2] in relation2id :
                        data.append([entity2id[triplet[0]], entity2id[triplet[1]], relation2id[triplet[2]]])
                        if file_type == 'train' and p == 1:
                            triplets_train.append([entity2id[triplet[0]], entity2id[triplet[1]], relation2id[triplet[2]]])
                if keeptrainone:
                    data_mr.append([entity2id[triplet[0]], entity2id[triplet[1]], 0])
                else:
                    data_mr.append([entity2id[triplet[0]], entity2id[triplet[1]], list_r_onehot])
                data_pol.append(p)
        triplets_train = np.array(triplets_train)
        triplets[file_type] = np.array(data)#merged triplets (h,r,t)
        triplets_mr[file_type] = data_mr# triplets (h,r,[t1,t2,t3,....,tn]) ti=0/1
        polarity_mr[file_type] = np.array(data_pol)#whether the fake triplets

    assert len(entity2id) == 604
    if not keeptrainone:
        assert rel == 200
    else:
        assert rel == 1
    triplet_kg = np.loadtxt(triple_file)
    for (h, t, r) in triplet_kg:
        h, t, r = int(h), int(t), int(r)
        if h not in entity2id:
            entity2id[h] = h
        if t not in entity2id:
            entity2id[t] = t 
        # same id within train/valid/test and BKG_file does not mean same relation
        if rel+r not in relation2id:
            relation2id[rel+r] = rel + r
        kg_triple.append([h, t, r])
    kg_triple = np.array(kg_triple)
    id2entity = {v: k for k, v in entity2id.items()}
    id2relation = {v: k for k, v in relation2id.items()}

    # Construct the list of adjacency matrix each corresponding to eeach relation. Note that this is constructed from the train data and BKG data.
    adj_list = []
    for i in range(rel):
        idx = np.argwhere(triplets_train[:, 2] == i)
        adj_list.append(csc_matrix((np.ones(len(idx), dtype=np.uint8), (triplets_train[:, 0][idx].squeeze(1), triplets_train[:, 1][idx].squeeze(1))), shape=(len(entity2id), len(entity2id))))
    for i in range(rel, len(relation2id)):
        idx = np.argwhere(kg_triple[:, 2] == i-rel)
        adj_list.append(csc_matrix((np.ones(len(idx), dtype=np.uint8), (kg_triple[:, 0][idx].squeeze(1), kg_triple[:, 1][idx].squeeze(1))), shape=(len(entity2id), len(entity2id))))
    return adj_list, triplets, entity2id, relation2id, id2entity, id2relation, rel, triplets_mr, polarity_mr

