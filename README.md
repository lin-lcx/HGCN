# HGCN
## Hybrid Graph Convolutional Network with Online Masked Autoencoder for Robust Multimodal Cancer Survival Prediction

![Image text](https://github.com/lin-lcx/HGCN/blob/main/overview.png)

## Abstract

Cancer survival prediction requires exploiting complementary multimodal information (e.g., pathological, clinical and genomic features) and it is even more challenging in clinical practices due to the incompleteness of patient’s multimodal data. Existing methods lack of sufficient intra- and inter-modal interactions and suffer from severe performance degradation caused by missing modalities. This paper proposes a novel hybrid graph convolutional network, called HGCN, equipped with an online masked autoencoder paradigm for robust multimodal cancer survival prediction. Particularly, we pioneer modeling the patient’s multimodal data into flexible and interpretable multimodal graphs with modality-specific preprocessing. Our elaborately designed HGCN integrates the advantages of graph convolutional network (GCN) and hypergraph convolutional network (HCN), by utilizing the node message passing and the hyperedge mixing mechanism to facilities intra-modal and inter-modal interactions of multimodal graphs. With the proposed HGCN, the potential of multimodal data can be better unleashed, leading to more reliable predictions of patient’s survival risk. More important, to handle the missing modalities in clinical scenarios, we incorporate an online masked autoencoder paradigm into HGCN, which can capture the intrinsic dependence between modalities and seamlessly generate missing hyperedges for model inference. Extensive experiments and analysis on six cancer cohorts from TCGA project (e.g., KIRC, LIHC, ESCA, LUSC, LUAD and UCEC) show that our method significantly outperforms the state-of-the-arts under both complete and missing modal settings.

## Data processing
**Genomic profile** can be downloaded from the [cBioPortal](https://www.cbioportal.org/).The categorization of gene embeddings can be obtained from [MSigDB](https://www.gseamsigdb.org/gsea/msigdb/gene_families.jsp?ex=1).


**Pathological slide** and **clinical records** can be downloaded from the [GDC](https://portal.gdc.cancer.gov/).
**Clinical records** included in different trials are shown in **Supplementary**.

**cut_and_pretrain.py** gives the code for cutting patch and pre-training.

The detailed steps of processing data can be seen in **gendata.ipynb**,this file shows how to encapsulate a data into the format we want.The tool for building graph neural network is pytorch geometric.

## Data

We provide the [data and labels](https://drive.google.com/drive/folders/1PIyGLj9NUSj07b16GmJ-b7mp7A5j09D1?usp=share_link) (**patients, sur_and_time, all_data, seed_fit_split**) required for the experiment in the article, as well as a set of trained model parameters.



## Train
After setting the parameters and save path in the file train.py, you can directly use the command line **python train.py** for training. The training process will be printed out, and the prediction results will be saved in the path.

**Hyperparameter setting** and **Experimental environment** are shown in **Supplementary**.

The folder **data_split** contains the data division we used for the experiment. If you want to use it, you can modify the parameter **if_fit_split** to True.

## Citation
- If you found our work useful in your research, please consider citing our works(s) at:
```
@article{hou2023hybrid,
  title={Hybrid Graph Convolutional Network with Online Masked Autoencoder for Robust Multimodal Cancer Survival Prediction},
  author={Hou, Wentai and Lin, Chengxuan and Yu, Lequan and Qin, Jing and Yu, Rongshan and Wang, Liansheng},
  journal={IEEE Transactions on Medical Imaging},
  year={2023},
  publisher={IEEE}
}
```

