# Hyperparameter setting

 λo | λc | λp | λr | C | LR        | Dropout | Epoch | Batch size  
 ----- | ------  | ----- | ------ | ----- | ------ | ----- | ------ | ----- 
 5           | 1             | 5             | 5             | 512  |3e-5|  0.5     | 60    | 32                    


# Experimental environment

GPU        | CPU                     | PyTorch | PyTorch Geometric | CUDA | cuDNN 
 ---- | ----- | ------  | ----- | ------ | ----- 
RTX 3090 × 4 | Intel Xeon Silver 4210R | 1.7.1   | 1.6.3           | 11.1 | 8.0.4 

# Clinical records included in different trials

Dataset | Clinical records      
---- | -----                          
LIHC        | Race, Age\_at\_index, Gender, BMI, Tumor\_grade                                      
ESCA        |Race, Age\_at\_index, Gender, Alcohol\_history, Primary\_diagnosis, Site\_of\_resection\_or\_biopsy, Morphology, BMI, Cigarettes\_per\_day, Tumor\_grade
LUSC        |Race, Age\_at\_index, Gender, Prior\_malignancy, Site\_of\_resection, Pack\_years\_smoked, Years\_smoked                 
LUAD        | Race, Age\_at\_index, Gender, Morphology, Prior\_malignancy, Site\_of\_resection\_or\_biopsy         
UCEC        | Race, Age\_at\_index, Primary\_diagnosis, Morphology                                                                                                     
KIRC        | Race, Age\_at\_index, Gender, Prior\_malignancy, Pack\_years\_smoked, Years\_smoked              















