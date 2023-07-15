import os
import logging
import json
import torch


def initialize_experiment(params, file_name):
    """
    Makes the experiment directory, sets standard paths and initializes the logger
    """
    params.main_dir = os.path.join(os.path.relpath(os.path.dirname(os.path.abspath(__file__))), '..')
    global_exps_dir = os.path.join(params.main_dir, 'experiments')
    if not os.path.exists(global_exps_dir):
        os.makedirs(global_exps_dir)

    params.exp_dir = os.path.join(global_exps_dir, params.experiment_name)

    if not os.path.exists(params.exp_dir):
        os.makedirs(params.exp_dir)
    
    file_handler = logging.FileHandler(os.path.join(params.exp_dir, "log_train.txt"))
    
    logger = logging.getLogger()
    logger.addHandler(file_handler)

    logger.info('============ Initialized logger ============')
    logger.info('\t '.join('%s: %s' % (k, str(v)) for k, v
                          in sorted(dict(vars(params)).items())))
    logger.info('============================================')

    with open(os.path.join(params.exp_dir, "params.json"), 'w') as fout:
        json.dump(vars(params), fout)


def initialize_model(params, model):
    """
    relation2id: the relation to id mapping, this is stored in the model and used when testing
    model: the type of model to initialize/load
    load_model: flag which decide to initialize the model or load a saved model
    """
    if params.load_model and os.path.exists(os.path.join(params.exp_dir, 'best_graph_classifier.pth')):
        logging.info('Loading existing model from %s' % os.path.join(params.exp_dir, 'best_graph_classifier.pth'))
        classifier = torch.load(os.path.join(params.exp_dir, 'best_graph_classifier.pth')).to(device=params.device)
    else:
        logging.info('No existing model found. Initializing new model..')
        classifier = model(params).to(device=params.device)

    return classifier
