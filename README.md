# centroIDA

Run the following commands to install `tllib` and all the dependency.
```shell
python setup.py install
pip install -r requirements.txt
```
*TLlib* is an open-source and well-documented library for Transfer Learning. You need to install *TLlib* via `pip`.
```shell
pip install -i https://test.pypi.org/simple/ tllib==0.4
```

You also need to install timm to use PyTorch-Image-Models.
```
pip install timm
```
Following datasets can be downloaded automatically:

- [OfficeHome](https://www.hemanthdv.org/officeHomeDataset.html)
- [DomainNet](http://ai.bu.edu/M3SDA/)

Run the following command, conduct transfer task from Cl to Pr in the OfficeHome dataset.
```shell
CUDA_VISIBLE_DEVICES=0 python CCA-LSC.py data/office-home -d OfficeHome -s Cl -t Pr -a resnet50 --epochs 20 --bottleneck-dim 256 --lr 0.005 --log logs/CCA-LSC/OfficeHome_Cl2Pr
```

Run the following command, conduct transfer task from r to c in the DomainNet dataset.
```shell
CUDA_VISIBLE_DEVICES=0 python CCA-LSC.py data/domainnet -d DomainNet -s r -t c -a resnet50 --epochs 30 --bottleneck-dim 256 --lr 0.01 --log logs/CCA-LSC/DomainNet_r2c
```