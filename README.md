## UnNull: Unsupervised Range-Nullspace Learning for Multispectral Demosaicing and Reconstruction

## Multispectral demosaicing experiments

- ***Demosaicing Experiments Data Preparation***
1) Datasets for multispectral demosaicing experiments include: the **CAVE** dataset and the **Harvard** dataset. [Download link (Google Drive)](https://drive.google.com/drive/folders/1br3eJfxSp2pgY7PT15J6NturF7-ubLvw?usp=drive_link)
```
Then, put the downloaded data into the [../Dataset/] folder
Each .mat file includes:
                        A (Sensing matrix) [500x500x25]
                        Img (Ground truth) [500x500x25]
                        y = A.*Img (Measurements) [500x500]
```

2) Alternatively, you can generate simulation multispectral demosaicing data according to the [code](https://github.com/gtsagkatakis/Snapshot_Spectral_Image_demosaicing) (Run main.m and save 'A' and 'Img').

- ***Run Demosaicing Experiments***
```
cd UnNull_De
python main.py
```

- ***Check Our Demosaicing Results***
```
cd UnNull_De/Results
load 'Img' and 'img (demosaiced img)' variables
run metric.m
```


---
## Multispectral reconstruction experiments

- ***Reconstruction Experiments Data Preparation***
Datasets for multispectral reconstruction experiments include: the **CAVE** dataset. [Download link (Google Drive)](https://drive.google.com/drive/folders/1br3eJfxSp2pgY7PT15J6NturF7-ubLvw?usp=drive_link)
```
Then, put the downloaded data into the [../Dataset/] folder
Each .mat file includes:
                        A (Sensing matrix) [500x500x25]
                        Img (Ground truth) [500x500x31]
                        X (Spectral bands) [500x500x25]
                        y = A.*Img (Measurements) [500x500]

Note: The A_matrix.mat file is needed, which include:
                        F (Fileter response matrix) [25x500x500x31]
```


- ***Run Reconstruction Experiments***
```
cd UnNull_Recon
python main.py
```

- ***Check Our Recontruction Results***
```
cd UnNull_Recon/Results
load 'Img' and 'img (reconstructed img)' variables
run metric.m
```


---
- ***Code Description***
```
main.py                     : main code (select test scene)
func.py                     : code includes some useful functions
test_metric                 : code for metrics
run_code/MSFA_CAVE.py       : code for loading simu data and preprocessing
run_code/MSFA_RealData.py   : code for loading real meas and preprocessing
optim_code/model_MSFA.py    : code for running UnNull
models/model_loader.py      : code for defining network parameters
models/UN_Net.py            : code for constructing network 
models/common.py            : code includes some network blocks
models/basicblock.py        : code includes some network blocks
```