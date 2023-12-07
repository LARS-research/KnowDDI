# Accurate and interpretable drug-drug interaction prediction enabled by knowledge subgraph learning


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

Background: Discovering potential drug-drug interactions (DDIs) is a long-standing challenge in clinical treatments and drug developments. Recently, deep learning techniques have been developed for DDI prediction. However, they generally require a huge number of samples, while known DDIs are rare.

Methods: In this work, we present KnowDDI, a graph neural network-based method that addresses the above challenge. KnowDDI enhances drug representations by adaptively leveraging rich neighborhood information from large biomedical knowledge graphs. Then, it learns a knowledge subgraph for each drug-pair to interpret the predicted DDI, where each of the edges is associated with a connection strength indicating the importance of a known DDI or resembling strength between a drug-pair whose connection is unknown. Thus, the lack of DDIs is implicitly compensated by the enriched drug representations and propagated drug similarities.

Results: Here we show the evaluation results of KnowDDI on two benchmark DDI datasets. Results show that KnowDDI obtains the state-of-the-art prediction performance with better interpretability. We also find that KnowDDI suffers less than existing works given a sparser knowledge graph. This indicates that the propagated drug similarities play a more important role in compensating for the lack of DDIs when the drug representations are less enriched.

Conclusions: KnowDDI nicely combines the efficiency of deep learning techniques and the rich prior knowledge in biomedical knowledge graphs. As an original open-source tool, KnowDDI can help detect possible interactions in a broad range of relevant interaction prediction tasks, such as protein-protein interactions, drug-target interactions and diseasegene interactions, eventually promoting the development of biomedicine and healthcare.

# Repo Contents

- [data](./data): the pre-processed dataset of Drugbank and BioSNAP.
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
@article{wang2023accurate,
  title={Accurate and interpretable drug-drug interaction prediction enabled by knowledge subgraph learning},
  author={Wang, Yaqing and Yang, Zaifei and Yao, Quanming},
  journal={arXiv preprint arXiv:2311.15056},
  year={2023}
}
```

## Acknowledgement
The code framework is based on [SumGNN](https://github.com/yueyu1030/SumGNN).