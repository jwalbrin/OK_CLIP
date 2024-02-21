# OK_CLIP

Basic implementation of main analyses from the forthcoming manuscript:

### Fine-grained knowledge about manipulable objects is well-predicted by CLIP

By Jon Walbrin, Nikita Sossounov, Morteza Mahdiani, Igor Vaz, & Jorge Almeida

### Setup

``` 
conda create -n ok_clip python=3.11 pip
conda activate ok_clip
pip install -r requirements.txt
```

### Data

Behavioural dimension scores (5 vision, 6 manipulation, 4 function dimensions)
- data/behavioural_dimensions/selected_dimensions.csv

Image representations for 80 tool set (for different networks)
- data/eighty_tools/...

Image representations for THINGS set 
- data/things/

Image representations for extra 20 set 
- data/extra/

### Scripts

For ease, each analysis stage is implemented with a separate .py script where variables can be set at the top of the script




