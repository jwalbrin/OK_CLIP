# OK_CLIP

Basic implementation of main analyses from the forthcoming manuscript:

"Fine-grained human object knowledge is well-captured by multimodal deep learning"

Jon Walbrin, Nikita Sossounov, Morteza Mahdiani, Igor Vaz, & Jorge Almeida
Data

a. Data/BehaviouralDimensions/SelectedDimensions.csv

    Dimension scores for 15 object knowledge dimensions described by Almeida et al. (2023)
        5 vision, 6 manipulation, 4 function dimensions b. Data/EightyTools/features.npy
    Extracted features of size: ((80 obects * 10 exemplars) * 1280)
        Extracted from the following model: vit_huge_patch14_224_clip.laion2b (obtained from https://pypi.org/project/timm/)

Scripts

a. Scripts/FeatureSelection/BestFeatSelect.py

    For given set(s) of k features, use PCA + RFE to identify best features/components

b. Scripts/Regression/ModelDims.py

    Cross-validated regression using best features (e.g. best 10 components identified with BestFeatSelect.py) for modelling each dimension

Set-up

´´´ python
conda create -n ok_clip python=3.11 pip
conda activate ok_clip
pip install -r requirements.txt
´´´