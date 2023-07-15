# Predicting Multi-typed Drug-Drug Interaction with Knowledge Subgraphs


## Contents

- [Overview](#overview)
- [Repo Contents](#repo-contents)
- [System Requirements](#system-requirements)
- [Installation Guide](#installation-guide)
- [Examples](#Examples)
- [Results](#results)
- [License](./LICENSE)
- [Issues](https://github.com/tata1661/KnowDDI-codes/issues)
- [Citation](#citation)

# Overview

Drug-drug interaction (DDI) prediction is a vital task in drug discovery. However, predicting DDI is hard as known DDI fact triplets are very rare. Here, we present KnowDDI to handle multi-typed DDI prediction by learning knowledge subgraphs from external knowledge graph. On a large, merged graph of DDI fact triplets and external knowledge graph which contains diverse organized information of biomedical entities, KnowDDI obtains generic node embeddings to encode the global topology. To leverage the relevant information to target drug pair, KnowDDI further extracts its directed drug pair-aware subgraph and refines its graph structure and node embeddings to be drug pair-aware. The resultant knowledge subgraph is more predictive of DDI types and can provide explaining paths to interpret the prediction results. Experimental results on benchmark datasets show that KnowDDI obtains better prediction performance, has better interpret ability, and suffers less from data sparsity than existing works. This repository contains the source code of KnowDDI.

# Repo Contents

- [data](./data): the pre-processed dataset of Drugbank and BioSNAP.
- [paddle](./paddle): the paddle version code of KnowDDI.
- [pytorch](./pytorch): the pytorch version code of KnowDDI.
- [raw_data](./raw_data): the origin dataset of Drugbank and BioSNAP.

# System Requirements

## Hardware Requirements

This repository requires only a standard computer with enough RAM to support the in-memory operations. We recommend that your computer contains a GPU.

## Software Requirements

### OS Requirements

The package development version is tested on *Linux*(Ubuntu 18.04) operating systems with CUDA 10.2.

### Python Dependencies
For the pytorch version, the environment required by the code is as follows.
```
python==3.7.15
pytorch==1.6.0
torchvision==0.7.0
cudatoolkit==10.2
lmdb==0.98
networkx==2.4
scikit-learn==0.22.1
tqdm==4.43.0
dgl-cu102==0.6.1
```

For the paddle version, the environment required by the code is as follows.
```
python==3.7.15
paddlepaddle-gpu==2.3.2
lmdb==0.98
networkx==2.4
scikit-learn==0.22.1
tqdm==4.43.0
pgl==2.2.4
```

# Installation Guide
For the pytorch version, please follow the commands below:
```
git clone git@github.com:tata1661/KnowDDI-codes.git
cd KnowDDI-codes
conda create -n KnowDDI_pytorch python=3.7
conda install pytorch==1.6.0 torchvision==0.7.0 cudatoolkit=10.2 -c pytorch
pip install dgl-cu102==0.6.1
pip install -r requirements.txt
cd pytorch
```

For the paddle version, please follow the commands below:
```
git clone git@github.com:tata1661/KnowDDI-codes.git
cd KnowDDI-codes
conda create -n KnowDDI_paddle python=3.7
python3 -m pip install paddlepaddle-gpu==2.4.1
pip install pgl==2.2.4
pip install -r requirements.txt
cd paddle
```

# Examples
The default parameters are the best on Drugbank dataset. To train and evaluate the model,you could run the following command.
```
python train.py -e Drugbank
```
Besides, to train and evaluate the model on BioSNAP dataset,you could run the following command.
```
python train.py -e BioSNAP --dataset=BioSNAP --eval_every_iter=452 --weight_decay_rate=0.00001 --threshold=0.1 --lamda=0.5 --num_infer_layers=1 --num_dig_layers=3 --gsl_rel_emb_dim=24 --MLP_hidden_dim=24 --MLP_num_layers=3 --MLP_dropout=0.2
```

## Dataset

We provide the dataset in the [data](data/) folder. 

| Data  | Source | Description
|-------|----------|----------|
| [Drugbank](./pytorch/data/drugbank/) | [This link](https://bitbucket.org/kaistsystemsbiology/deepddi/src/master/data/)| A drug-drug interaction network betweeen 1,709 drugs with 136,351 interactions.| 
| [TWOSIDES](./pytorch/data/BioSNAP/) | [This link](http://snap.stanford.edu/biodata/datasets/10017/10017-ChChSe-Decagon.html)| A drug-drug interaction network betweeen 645 drugs with 46221 interactions.|
| Hetionet | [This link](https://github.com/hetio/hetionet) | The knowledge graph containing 33,765  nodes  out  of  11  types  (e.g.,  gene,  disease,  pathway,molecular function and etc.) with 1,690,693 edges from 23 relation types after preprocessing (To ensure **no information leakage**, we remove all the overlapping edges  between  HetioNet  and  the  dataset).

We provide the mapping file between ids in our pre-processed data and their original name/drugbank id as well as a copy of hetionet data and their mapping file on [this link](./raw_data).


# Results
We provide [examples](./pytorch/experiments/) on two datasets with expected experimental results and running times.

# Citation

Please kindly cite this paper if you find it useful for your research. Thanks!

```

```

## Acknowledgement
The code framework is based on [SumGNN](https://github.com/yueyu1030/SumGNN).