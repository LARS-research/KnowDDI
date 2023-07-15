import os
import numpy as np
import torch
import random
from sklearn import metrics
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score, average_precision_score,accuracy_score
from tqdm import tqdm
from utils.graph_utils import collate_dgl, move_batch_to_device_dgl
GLOBAL_SEED=1
GLOBAL_WORKER_ID=None


def init_fn(worker_id): 
    global GLOBAL_WORKER_ID
    GLOBAL_WORKER_ID = worker_id
    seed = GLOBAL_SEED + worker_id
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = False

class Evaluator_multiclass():
    """
    Drugbank
    """
    def __init__(self, params, classifier, data,is_test=False):
        self.params = params
        self.graph_classifier = classifier
        self.data = data
        self.global_graph = data.global_graph
        self.move_batch_to_device = move_batch_to_device_dgl
        self.collate_fn = collate_dgl
        self.num_workers = params.num_workers
        self.is_test = is_test
        self.eval_times = 0
        self.current_epoch = 0

    def eval(self):
        self.eval_times += 1
        scores = []
        labels = []
        self.current_epoch += 1
        dataloader = DataLoader(self.data, batch_size=self.params.batch_size, shuffle=False, num_workers=self.num_workers, collate_fn=self.collate_fn,worker_init_fn=init_fn)

        self.graph_classifier.eval()
        with torch.no_grad():
            for b_idx, batch in tqdm(enumerate(dataloader)):
                data, r_labels, polarity= self.move_batch_to_device(batch, self.params.device,multi_type=1)
                score = self.graph_classifier(data)


                label_ids = r_labels.to('cpu').numpy()
                labels += label_ids.flatten().tolist()
                scores += torch.argmax(score, dim=1).cpu().flatten().tolist() 

        auc = metrics.f1_score(labels, scores, average='macro')
        auc_pr = metrics.f1_score(labels, scores, average='micro')
        f1 = metrics.f1_score(labels, scores, average=None)
        kappa = metrics.cohen_kappa_score(labels, scores)
        return {'auc': auc, 'auc_pr': auc_pr, 'k':kappa}, {'f1': f1}

class Evaluator_multilabel():
    """
    BioSNAP
    """
    def __init__(self, params, classifier, data):
        self.params = params
        self.graph_classifier = classifier
        self.data = data
        self.global_graph = data.global_graph
        self.move_batch_to_device = move_batch_to_device_dgl
        self.collate_fn = collate_dgl
        self.num_workers = params.num_workers

    def eval(self):
        pred_class = {}
        dataloader = DataLoader(self.data, batch_size=self.params.batch_size, shuffle=False, num_workers=self.num_workers, collate_fn=self.collate_fn,worker_init_fn=init_fn)
        
        self.graph_classifier.eval()
        with torch.no_grad():
            for batch in tqdm(dataloader):
                data, r_labels, polarity = self.move_batch_to_device(batch, self.params.device,multi_type=2)
                score_pos = self.graph_classifier(data)

                m = nn.Sigmoid()
                pred = m(score_pos)
                labels = r_labels.detach().to('cpu').numpy() # batch * 200
                preds = pred.detach().to('cpu').numpy() # batch * 200
                polarity = polarity.detach().to('cpu').numpy()
                for (label, pred, pol) in zip(labels, preds, polarity):
                    for i, (l, p) in enumerate(zip(label, pred)):
                        if l == 1:
                            if i in pred_class:
                                pred_class[i]['pred'] += [p]
                                pred_class[i]['pol'] += [pol] 
                                pred_class[i]['pred_label'] += [1 if p > 0.5 else 0]
                            else:
                                pred_class[i] = {'pred':[p], 'pol':[pol], 'pred_label':[1 if p > 0.5 else 0]}
                                
        roc_auc = [ roc_auc_score(pred_class[l]['pol'], pred_class[l]['pred']) for l in pred_class]
        prc_auc = [ average_precision_score(pred_class[l]['pol'], pred_class[l]['pred']) for l in pred_class]
        ap =  [accuracy_score(pred_class[l]['pol'], pred_class[l]['pred_label']) for l in pred_class]
        return {'auc': np.mean(roc_auc), 'auc_pr': np.mean(prc_auc), 'f1': np.mean(ap)}, {"auc_all":roc_auc,"aupr_all":prc_auc, "f1_all":ap}

