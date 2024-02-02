# Fact_group_29: Reproducibility study of explaining temporal graph models through an explorer-navigator framework

In this study, we replicated the main results in the paper: [Temporal Graph Networks for Deep Learning on Dynamic Graphs](https://arxiv.org/abs/2006.10637)

## Installation  

The project is launched as a python library, to run the code in this project, we first need to install this library. Go to the path where the 'setup.py' is in, and do the installation with the following command:  

```
cd  ~/workspace/TGNNEXPLAINER-PUBLIC/
pip install -e .
```

## Overview of running the code in this project

The overview of the steps can be find in the 'REAREADMEDME.md' in:  

```
~/workspace/GNNEXPLAINER-PUBLIC/tgnnexplainer/README.md
``` 

To replicate the results of this project, three parts of the tasks should be performed: data process, training corresponding models and running the explainers.  
This overview sample command of these three parts can be found in this 'README.md'. 

However, more detailed guidance for training and running explainers are in incorporated in the 'README.md' in corresponding child-folders. Following sections aims to provide a clear step-by step guidance to run the code with the link to the path of these sub-README.md files.  

### Data process

 This step contains:  
 
 1. Download real-world datasets, generate simulate datasets  

 2. Generate explain indexes  

 Details in:   
 ```  
  ~/workspace/TGNNEXPLAINER-PUBLIC/README.md
 ```

 ### Training the model

 #### TGAT model:  

```
~/workspace/TGNNEXPLAINER-PUBLIC/tgnnexplainer/xgraph/models/ext/tgat/README.md
```

Details in this README.md file:  

1. Introduction to this paper. And also some information related to the predecessor work and follow-up work.  

2. Detailed steps of from data process up to running the TGAT model are contained. The content of the processed data is given.  

3. Python envitonmental requirments is given in this file. For the installation convenience, we copy-pasted this information below:    

* python >= 3.7

* Dependency

```{bash}
pandas==0.24.2
torch==1.1.0
tqdm==4.41.1
numpy==1.16.4
scikit_learn==0.22.1
```

4. Sample command to run the code and general flags are given.   

5. Cite information is given at the end of this file. For convenience purposes, we copy-pasted this information below:   

```
@inproceedings{tgat_iclr20,
title={Inductive representation learning on temporal graphs},
author={da Xu and chuanwei ruan and evren korpeoglu and sushant kumar and kannan achan},
booktitle={International Conference on Learning Representations (ICLR)},
year={2020}
}
```

#### TGN model: 

```
~/workspace/TGNNEXPLAINER-PUBLIC/tgnnexplainer/xgraph/models/ext/tgn/README.md
```

Details in this README.md file:  

1. Short introduction to the paper with the link

2. Python environmental requirements:  

Dependencies (with python >= 3.7):

```{bash}
pandas==1.1.0
torch==1.6.0
scikit_learn==0.23.1
```

Worth noticing here that the requirements is not consistent with the TGAT model requirements. In out replication study, we used the requirements mentioned in TGAT root 'README.md' instead of this one. 

3. Data downloading and processing. Worth noticing here that for TGN model, the authors do not provide the command for simulated data. In out replication experiments, we copy pasted the processed data from TGAT model. 

4. Model training: sample commands for real-world dataset are given while the commands for the simulated datasets are missing. The command and parameter settings to run the simulated data is not clear, especially when there are multiple potential choices:   

In the 'tgn' folder, there are three training files: 'train_self_supervised.py', 'train_supervised.py' and 'train_simulate.py'.   

The relation of 'train_self_supervised.py' and 'train_supervised.py' is clear (for the real-world datasets), as stated in the 'README.md' in the 'tgn' folder, 'train_supervised.py' trains a dynamic node classification based on the self-supervised task.   

For the simulated data, the guidance is not clear enough. Our assumption is that the simulated data is trained by the 'train_simulate.py' file and based on the results from this training, dynamic node classification can be trained with 'train_supervised.py'. However, we found the simulated datasets can also be trained with 'train_self_supervised.py' file. 

5. Baselines and ablation study

For baselines, it gives sample commands to run with 'Jodie' and 'Dyrep'.  

For the ablation study, it gives sample commands to change '--embedding_module' or '--aggregator' settings.

This part is excluded from our replication study.  

6. General flags.

7. Cite information. Duplicated with the 'README.md' in 'tgat' folder, hence we don't put the details here. 

#### Copy checkpoint

This step is stated in the 'README.md' in '~/workspace/GNNEXPLAINER-PUBLIC/tgnnexplainer/README.md'.  

When the training is completed, we need to execute 'cpckpt.sh' to copy the checkpoints to the corresponding folder to prepare for the running of explainers in the following section with the command in terminal:

tgat:
```
cd  ~/workspace/TGNNEXPLAINER-PUBLIC/tgnnexplainer/xgraph/models/ext/tgat
./cpckpt.sh
```

tgn:
```
cd  ~/workspace/TGNNEXPLAINER-PUBLIC/tgnnexplainer/xgraph/models/ext/tgn
./cpckpt.sh
```


### Explainers

This step is stated in the 'README.md' in '~/workspace/GNNEXPLAINER-PUBLIC/tgnnexplainer/README.md'.  

```
cd  ~/workspace/TGNNEXPLAINER-PUBLIC/benchmarks/xgraph
./run.sh
``` 

Worth notice here, the author put the exucution of four explainer in one 'run.sh' file. Regarding the long GPU time for the T-GNNExplainer, we recommend to run the T-GNNExplainer separatly (in parallel, we can run the other 3 explainers) to shorten the total time consumption. Also, it is recommend to assign the task with longer expected running time if the code is executed in a '.job' file.  

As a reference, in our experiments, the longest time consumption to run the T-GNNExpainer is 25h (for reddit dataset), executed with the environment: 1 NVIDIA A100-SXM4-40GB GPU with 18 CPUs. 



## Experiments

In this study, we replicated the results by training the TGAT and TGN model with 4 datasets respectively, and running the 4 explainers to create a baseline results.  

To dive deeper, we modified the multi-layer perceptron (mlp) in the explainer, see the modifications in file:  

```
~/workspace/TGNNEXPLAINER-PUBLIC/tgnnexplainer/xgraph/method/other_baselines_tg.py
```  

Originally in the paper, a 2 layer MLP with 128 hidden units is applied. We modified it to 3 layers with 256 hidden units in the first layer and 128 hidden units in the second layer respectively.  

In the future study, it would be interesting to look into the structure of MLP and the impacts it gives to the model outcome and efficiency.  
 
Variations such as change the hidden unit to 64 or 32 might be tested. With given time limit and computational power, these experiments are excluded in our study. 

## Processing results

To process the results, open the notebook in the processing_results folder and run the code inside it to generate the graphs and process benchmark scores. Make sure to put the resulting output files of the model in the processing_results/Results folder or change the loading directory inside the notebook itself. The folder processing_results/Results is already filled with the results from our runs.
