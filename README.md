# HGCN
## Hybrid Graph Convolutional Network with Online Masked Autoencoder for Robust Multimodal Cancer Prognosis


![Image text](https://github.com/lin-lcx/HGCN/blob/main/overview.png)

## Abstract

Cancer prognosis requires exploiting complementary multimodal information, e.g., pathological, clinical and genomic features, and it is even more challenging in clinical practices due to the incompleteness of patient’s multimodal data. However, the existing methods lack of sufficient excavating of the intra- and inter-modal interactions of multimodal data, and ignore the model inability or significant performance degradation caused by missing modalities. This paper proposes a novel hybrid graph convolutional network (i.e., HGCN) for multimodal cancer prognosis and introduces an online masked attention mechanism to tackle the missing modalities. Particularly, we pioneer modeling the patient’s multimodal data into flexible and interpretable multimodal graphs by modalrelated preprocessing methods. Furthermore, our elaborately designed HGCN combines the advantages of graph convolutional network (GCN) and hypergraph convolutional network (HCN), utilizing the node message passing and the hyperedge mixing to facilities intra-modal and inter-modal interactions of multimodal graphs, respectively. With the proposed HGCN, the potential of multimodal data can be better unleashed, leading to more reliable predictions of patient’s survival risk. More importantly, to handle the missing modalities, we proposed an online masked autoencoder paradigm embedded in HGCN, which cleverly generate the missing hyperedges information during network inference by learning the intrinsic dependence between modalities. Extensive experiments and analysis on six cancer cohorts from TCGA project show that our method clearly outperforms the state-of-the-art methods under both complete and missing modalities settings. In addition, abundant explanations of our model are provided, which may assist clinical management and potentially lead to biomarker discoveries.


## Train
After setting the parameters and save path in the file train.py, you can directly use the command line python train.py for training. The training process will be printed out, and the prediction results will be saved in the path.
