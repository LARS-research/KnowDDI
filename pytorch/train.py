import os
import argparse
import random
import torch
import numpy as np
import logging
from warnings import simplefilter
from scipy.sparse import SparseEfficiencyWarning
from manager.trainer import Trainer
from manager.evaluator import Evaluator_multiclass, Evaluator_multilabel
from model.Classifier_model import Classifier_model
from utils.initialization_utils import initialize_experiment, initialize_model
from data_processor.datasets import SubgraphDataset
from data_processor.subgraph_extraction import generate_subgraph_datasets
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'


def process_dataset(params):
    params.db_path = os.path.join(params.main_dir, f'../data/{params.dataset}/digraph_hop_{params.hop}_{params.BKG_file_name}')

    if not os.path.isdir(params.db_path):
        generate_subgraph_datasets(params)
 
    train_data = SubgraphDataset(db_path=params.db_path,
                                db_name='train_subgraph',
                                raw_data_paths=params.file_paths,
                                add_traspose_rels=params.add_traspose_rels,
                                use_pre_embeddings=params.use_pre_embeddings,
                                dataset=params.dataset,
                                kge_model=params.kge_model,
                                dig_layer = params.num_dig_layers,
                                BKG_file_name=params.BKG_file_name)

    test_data = SubgraphDataset(db_path=params.db_path,
                                db_name='test_subgraph',
                                use_pre_embeddings=params.use_pre_embeddings,
                                dataset=params.dataset,
                                kge_model=params.kge_model,
                                ssp_graph=train_data.ssp_graph,
                                id2entity=train_data.id2entity,
                                id2relation=train_data.id2relation,
                                rel=train_data.num_rels,
                                global_graph=train_data.global_graph,
                                dig_layer = params.num_dig_layers,
                                BKG_file_name=params.BKG_file_name)

   
    valid_data = SubgraphDataset(db_path=params.db_path,
                                db_name='valid_subgraph',
                                use_pre_embeddings=params.use_pre_embeddings,
                                dataset=params.dataset,
                                kge_model=params.kge_model,
                                ssp_graph=train_data.ssp_graph,
                                id2entity=train_data.id2entity,
                                id2relation=train_data.id2relation,
                                rel=train_data.num_rels,
                                global_graph=train_data.global_graph,
                                dig_layer = params.num_dig_layers,
                                BKG_file_name=params.BKG_file_name)

    params.num_rels = train_data.num_rels  # only relations in dataset
    params.global_graph = train_data.global_graph.to(params.device)
    params.aug_num_rels = train_data.aug_num_rels  # including relations in BKG and self loop
    params.num_nodes = 35000
    logging.info(f"Device: {params.device}")
    logging.info(f" # Relations : {params.num_rels}, # Augmented relations : {params.aug_num_rels}")

    return train_data, valid_data, test_data

  
def main(params):
    simplefilter(action='ignore', category=UserWarning)
    simplefilter(action='ignore', category=SparseEfficiencyWarning)

    params.file_paths = {
        'train': os.path.join(params.main_dir, '../data/{}/{}.txt'.format(params.dataset, params.train_file)),
        'valid': os.path.join(params.main_dir, '../data/{}/{}.txt'.format(params.dataset, params.valid_file)),
        'test': os.path.join(params.main_dir, '../data/{}/{}.txt'.format(params.dataset, params.test_file))
    }
    train_data, valid_data, test_data = process_dataset(params)

    classifier = initialize_model(params, Classifier_model)
    
    valid_evaluator = Evaluator_multiclass(params, classifier, valid_data) if params.dataset == 'drugbank' \
        else Evaluator_multilabel(params, classifier, valid_data)
    test_evaluator = Evaluator_multiclass(params, classifier, test_data,is_test=True) if params.dataset == 'drugbank' \
        else Evaluator_multilabel(params, classifier, test_data)
    print(classifier)

    trainer = Trainer(params, classifier, train_data, valid_evaluator, test_evaluator)
    logging.info('start training...')
    trainer.train()
    

if __name__ == '__main__':
    
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="model params")
    """
    default params are best on drugbank
    best params on BioSNAP: --dataset=BioSNAP  --eval_every_iter=452 --weight_decay_rate=0.00001 --threshold=0.1 --lamda=0.5 --num_infer_layers=1 --num_dig_layers=3 --gsl_rel_emb_dim=24 --MLP_hidden_dim=24 --MLP_num_layers=3 --MLP_dropout=0.2
    """

    # global
    parser.add_argument('--seed', type=int, default=1111, help="seeds for random initial")
    parser.add_argument("--gpu", type=int, default=3, help="Which GPU to use?")
    parser.add_argument('--disable_cuda', action='store_true', help='Disable CUDA')
    parser.add_argument('--load_model', action='store_true', help='Load existing model?')
    parser.add_argument("--experiment_name", "-e", type=str, default="default", help="A folder with this name would be created to dump saved models and log files")

    # dataset
    parser.add_argument('--dataset', "-d", type=str, default='drugbank')
    parser.add_argument("--train_file", "-tf", type=str, default="train", help="Name of file containing training triplets")
    parser.add_argument("--valid_file", "-vf", type=str, default="valid", help="Name of file containing validation triplets")
    parser.add_argument("--test_file", "-ttf", type=str, default="test", help="Name of file containing validation triplets")
    parser.add_argument("--kge_model", type=str, default="TransE", help="Which KGE model to load entity embeddings from")
    parser.add_argument("--use_pre_embeddings", type=bool, default=False, help='whether to use pretrained KGE embeddings')
    parser.add_argument('--BKG_file_name', type=str, default='BKG_file')

    # extract subgraphs 
    parser.add_argument("--max_links", type=int, default=250000, help="Set maximum number of train links (to fit into memory)")
    parser.add_argument("--hop", type=int, default=2, help="Enclosing subgraph hop number")
    parser.add_argument("--max_nodes_per_hop", "-max_h", type=int, default=200, help="if > 0, upper bound the # nodes per hop by subsampling")
    parser.add_argument('--enclosing_subgraph', '-en', type=bool, default=True, help='whether to only consider enclosing subgraph')
    parser.add_argument('--add_traspose_rels', '-tr', type=bool, default=False, help='whether to append adj matrix list with symmetric relations')

    # trainer
    parser.add_argument("--eval_every_iter", type=int, default=526, help="Interval of iterations to evaluate the model")
    parser.add_argument("--save_every_epoch", type=int, default=10, help="Interval of epochs to save a checkpoint of the model")
    parser.add_argument("--early_stop_epoch", type=int, default=10, help="Early stopping patience")
    parser.add_argument("--optimizer", type=str, default="Adam", help="Which optimizer to use?")
    parser.add_argument("--lr", type=float, default=0.005, help="Learning rate of the optimizer")
    parser.add_argument("--lr_decay_rate", type=float, default=0.93, help="adjust the learning rate via epochs")
    parser.add_argument("--weight_decay_rate", type=float, default=1e-5)
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size")
    parser.add_argument("--num_epochs", "-ne", type=int, default=50,help="numer of epochs")
    parser.add_argument("--num_workers", type=int, default=32,help="Number of dataloading processes")
    
    # GraphSAGE params
    parser.add_argument("--emb_dim", "-dim", type=int, default=32,help="Entity embedding size")
    parser.add_argument("--num_gcn_layers", type=int, default=2,help="Number of GCN layers")
    parser.add_argument('--gcn_aggregator_type', type=str, choices=['mean', 'gcn', 'pool'], default='mean')
    parser.add_argument("--gcn_dropout", type=float, default=0.2,help="node_dropout rate in GCN layers")

    # gsl_Model params
    parser.add_argument("--num_infer_layers", type=int, default=3,help="Number of infer layers") 
    parser.add_argument("--num_dig_layers", type=int, default=3)
    parser.add_argument("--MLP_hidden_dim", type=int, default=16)
    parser.add_argument("--MLP_num_layers", type=int, default=2)
    parser.add_argument("--MLP_dropout", type=float, default=0.2)
    parser.add_argument("--func_num", type=int, default=1)
    parser.add_argument("--sparsify", type=int, default=1)
    parser.add_argument("--threshold", type=float, default=0.05)    
    parser.add_argument("--edge_softmax", type=int, default=1)
    parser.add_argument("--gsl_rel_emb_dim", type=int, default=32)
    parser.add_argument("--lamda", type=float, default=0.7)   
    parser.add_argument("--gsl_has_edge_emb", type=int, default=1)

    params = parser.parse_args()

    def set_seed(seed):
        np.random.seed(seed)
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ":16:8"
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.enabled = False
        torch.backends.cudnn.benchmark = False

    set_seed(params.seed)
    initialize_experiment(params, __file__)

    if not params.disable_cuda and torch.cuda.is_available():
        params.device = torch.device('cuda:%d' % params.gpu)
    else:
        params.device = torch.device('cpu')

    main(params)


