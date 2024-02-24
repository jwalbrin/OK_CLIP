# OK_CLIP

Data and analysis code for the forthcoming manuscript:

### Fine-grained knowledge about manipulable objects is well-predicted by CLIP

By Jon Walbrin, Nikita Sossounov, Morteza Mahdiani, Igor Vaz, & Jorge Almeida

<img src="https://github.com/jwalbrin/OK_CLIP/blob/main/results_fig.png?raw=true" alt="GitHub Logo" width="640"/>

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
│     │   ├── [DNN features] # extra (20 unseen) objects
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

### Analysis

For ease, each analysis step is implemented with a separate .py script (variables can be set at the top of the script). 

``` 
# --- Mandatory steps (for each desired network)
1. oc_select_components.py    # cross-validated component selection 
-- This generates a data_object (class) that is required for each successive script
2. oc_predictions.py    # generatate cross-validated predictions 

# --- Main analysis 
oc_permutations.py    # generate cross-validated permutations (optional)
oc_plot_bars.py    # plot bar charts 
oc_plot_lines.py    # plot line charts 

# --- Unique variance analyses
oc_predictions_uv.py    # generatate cross-validated, combined predictions for specified model pair(s)
oc_plot_lines_uv.py    # plot line charts 

# --- THINGS objects analysis 
oc_permutations_things.py    # generate cross-validated permutations (optional)
oc_plot_bars_things.py

# --- Extra (20 unseen) objects analysis
oc_permutations_extra.py    # generate cross-validated permutations (optional)
oc_plot_bars_extra.py

```






