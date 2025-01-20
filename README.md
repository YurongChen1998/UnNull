## UnNull: Unsupervised Range-Nullspace Learning for Multispectral Demosaicing and Reconstruction

## Multispectral demosaicing experiments
- ***Demosaicing Experiments Data Preparation***
1) Datasets for multispectral demosaicing experiments include: the **CAVE** dataset and the **Harvard** dataset. [Download link (Google Drive)](https://drive.google.com/drive/folders/1br3eJfxSp2pgY7PT15J6NturF7-ubLvw?usp=drive_link)
```
Then, put the downloaded data into the [Dataset] folder
Each .mat file includes:
                        A (Sensing matrix) [500x500x25]
                        Img (Ground truth) [500x500x25]
                        y = A.*Img (Measurements) [500x500]
```

2) Alternatively, you can generate simulation multispectral demosaicing data according to the [code](https://github.com/gtsagkatakis/Snapshot_Spectral_Image_demosaicing) (Run main.m and save A/Img).



- ***Denoising Experiments***

```
cd Denoise
python main_*.py
(We have uploaded KAIST Scene01 data. You can run main_1_KAIST.py directly.)
```
