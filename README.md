# ProxFuse
--------------------------------------------------------------------------------

## Environment Setup
The codebase is implemented in Python 3.9. package versions used for development are below.
```
networkx                        3.2.1
numpy                           1.26.4
torch                           2.4.0+cu121
```

## Folder structure
- ./execution/ stores files that can be executed to generate outputs when we learn combination weights. For vast number of experiments, we use [GNU parallel](https://www.gnu.org/software/parallel/), which can be downloaded in command line and make it executable via:
```
wget http://git.savannah.gnu.org/cgit/parallel.git/plain/src/parallel
chmod 755 ./parallel
```

- ./joblog/ stores job logs from parallel. 
You might need to create it by 
```
mkdir joblog
```

- ./Output/ stores raw outputs (ignored by Git) from parallel.
You might need to create it by 
```
mkdir Output
```

- ./data/ stores raw and processed data sets.

- ./analysis_raw_data/ stores analysis raw data.

- ./saved_epoch_beta_error/, ./saved_models/, and ./saved_parameters/ are for learning graph weights.

- ./src/ stores source files.

## Reproduce results
1) First, get into the ./src/ folder:
```
cd src
```
Prepare name lists and initial graphs without hardcoded weights:
```
python initial_preparation.py
```

2) Then we learn graph weights iteratively by using learn_weights.py, we can get into the ./execution/ folder:
```
cd execution
```
To reproduce the results to be executed on GPU-0 for machine a.
```
bash a0.sh
```
The file ./src/chimps_learned_weights_analysis.ipynb provides an analysis notebook for learning weights for chimps, while ./src/syn_learned_weights_analysis.ipynb is for synthetic data.

3) After that, we prepare the full learned graphs. First, back to ./src/:
```
cd src
```
Prepare data:
```
python learned_graphs_preparation.py
```

4) We then conduct data analysis on the learned graphs.
```
python data_analysis.py
```

5) Similarity analysis.
```
python similarity_analysis.py
```

Note that if you are operating on CPU, you may delete the commands ``CUDA_VISIBLE_DEVICES=xx". You can also set you own number of parallel jobs, not necessarily following the j numbers in the .sh files, or use other GPU numbers.

--------------------------------------------------------------------------------