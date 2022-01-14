# MHCN
## Hybrid Intra- and Inter-modal Interactions with Masked Hypergraph Convolutional Network for Multimodal Cancer Prognosis


![Image text](https://github.com/qweghj123/H2-MIL/blob/main/overview.png)

## Abstract

Cancer prognosis requires utilizing complementary multimodal information, e.g., clinical, pathological and genetic features, for assessing the risk of mortality in patients, and it is even more challenging in clinical practices due to the incompleteness of patient’s multimodal data. This paper presents a hybrid intra-modal and inter-modal interactions framework for multimodal cancer prognosis and proposes a novel masked hypergraph convolutional network (i.e., MHCN) to tackle the missing modalities. Particularly, we pioneer modeling the patient’s multimodal data with a graph structure and design independent graph convolutional networks (GCNs) to prompt intra-modal information interaction. Furthermore, we extract the higher-order representation of individual modalities as hyperedges and mix up the hyperedges to facilitate the inter-modal interaction. More importantly, to handle the missing modalities, we design a novel masked autoencoder paradigm to generate the missing hyperedges during the network inference. With the proposed hybrid intra- and inter-modal interactions, the potential of multimodal data can be fully unleashed, leading to more reliable predictions. Extensive experiments on five cancer cohorts from the TCGA project shows that our method clearly outperforms the state-of-the-art methods under both complete and missing modalities settings.
