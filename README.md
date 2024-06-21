# PreciseADR

![PreciseADR Model](img%2Fmodel.png)

## Run the PreciseADR Model
1. Install the package
```bash
conda env create -f py3.yml
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.2.0+cu121.html
```
2. Evaluate the model
```shell
python -m main_eval 
```

## (Optional) Build the Dataset
See in `data/README.md`


## Cite