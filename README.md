# ZeroBind
This is the implementation of ZeroBind: A protein-specific meta-learning framework with subgraph matching for binding predictions of unseen proteins and drugs.
## Installation
we recommend using conda as enviroment management.
```bash
conda create -n your_environment python=3.9
conda activate your_environment
conda install pytorch torchvision torchaudio cu102 -c pytorch
conda install pyg -c pyg
conda install lightning -c conda-forge
pip install lightning
conda install -c conda-forge rdkit
pip install graphein
pip install fair-esm
```
## Train
```bash
python metaentry.py  --batch_size=4  --num_workers=16 --num_inner_steps=5 --k_query=50
```
