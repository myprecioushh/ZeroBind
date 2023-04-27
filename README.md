# ZeroBind
This is the implementation of ZeroBind: A protein-specific meta-learning framework with subgraph matching for binding predictions of unseen proteins and drugs.
## Installation
ZeroBind is built on Python3, we recommend using a virtual conda environment as enviroment management for the installation of ZeroBind and its dependencies. The virtual environment can be created as follows:
```bash
conda create -n your_environment python=3.9
conda activate your_environment
```
Download the source code of ZeroBind from GitHub:
```bash
git clone https://github.com/myprecioushh/ZeroBind.git
```
Install ZeroBind dependencies as following:
```bash
conda install pytorch torchvision torchaudio cu102 -c pytorch
conda install pyg -c pyg
conda install lightning -c conda-forge
pip install lightning
conda install -c conda-forge rdkit
pip install graphein
pip install fair-esm
```
## Train
Multiple hyperparameters can be selected in meta.py. 
```bash
python metaentry.py  --batch_size=4  --num_workers=16 --num_inner_steps=5 --k_query=50
```
## Prediction
```bash
python metaentry.py  --test --num_workers=16 --k_query=50
```
## Online service
Online retrieval service and benchmark datasets are in [here](http://www.csbio.sjtu.edu.cn/bioinf/ZeroBind/index.html).

## License
This project is covered under the Apache 2.0 License.
