# OK_CLIP

Basic implementation of main analyses from the forthcoming manuscript:

### Fine-grained knowledge about manipulable objects is well-predicted by CLIP

By Jon Walbrin, Nikita Sossounov, Morteza Mahdiani, Igor Vaz, & Jorge Almeida

### Repo structure

``` 
root/
│
├── data/
│     ├── behavioural_dimensions/
│     │   ├── selected_dimensions.csv
│     ├── eighty_tools/
│     │   ├── [DNN features] # main experiment
│     ├── extra/
│     │   ├── [DNN features] # 2o extra unseen objects
│     ├── things/
│     │   ├── [DNN features] # things image set
|
├── functions/
│       ├── functions.py
│    
├── [scripts]  

```

### Setup

Install conda enivronment:

``` 
conda create -n ok_clip python=3.11 pip
conda activate ok_clip
pip install -r requirements.txt
```

### TO DO - FINISH

For ease, each analysis stage is implemented with a separate .py script where variables can be set at the top of the script

``` 
# --- Main analysis (run steps 1-3 in order)
1. oc_select_components.py # cross-validated component selection 
2. oc_predictions.py # generatate cross-validated predictions
3. oc_permutations.py # generate cross-validated permutations (optional)
oc_plot_bars.py # plot bar charts 
oc_plot_lines.py # plot line charts 

# --- Unique variance analyses (run steps 1 & 2 first)
oc_predictions_uv.py # generatate cross-validated cobined predictions for model pair(s)
oc_plot_lines_uv.py # plot line charts 

# THINGS analysis (run steps 1, 2 + optionally 3)




```






