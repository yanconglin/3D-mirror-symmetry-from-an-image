# NeRD++: Improved 3D-mirror symmetry learning from a single image 

[NeRD++](https://arxiv.org/abs/2112.12579) is a follow-up work over [NeRD: Neural 3D Reflection Symmetry Detector](https://arxiv.org/abs/2105.03211), which aims to detect the dominant mirror symmetry plane from a single-view image. This work has been accepted at BMVC 2022.

[Yancong Lin](https://yanconglin.github.io/), and [Silvia Laura Pintea](https://silvialaurapintea.github.io/), and [Jan C. van Gemert](http://jvgemert.github.io/)

Vision Lab, Delft University of Technology, the Netherlands


Abstract: 
Many objects are naturally symmetric, and this symmetry can be exploited to infer unseen 3D properties from a single 2D image. Recently, NeRD is proposed for accurate 3D mirror plane estimation from a single image. Despite the unprecedented accuracy, it relies on large annotated datasets for training and suffers from slow inference. Here we aim to improve its data and compute efficiency. We do away with the computationally expensive 4D feature volumes and instead explicitly compute the feature correlation of the pixel correspondences across depth, thus creating a compact 3D volume. We also design multi-stage spherical convolutions to identify the optimal mirror plane on the hemisphere, whose inductive bias offers gains in data-efficiency. Experiments on both synthetic and real-world datasets show the benefit of our proposed changes for improved data efficiency and inference speed.

This repo is still in progress!

# Data preprocessing
Please refer to the baseline work [NeRD](https://github.com/zhou13/nerd#downloading-the-processed-datasets) for details.

# Train and test on ShapeNet or Pix3D datasets
Train: `python train.py -d 0 --identifier nerd++ config/config.yaml` 

Test: `python eval.py -d 0 --output result.npz path/config.yaml path/checkpoint.pth.tar`




# Checkpoints
To do

# Citation
```bash
@inproceedings{lin2021symmetry,
    author={Lin, Yancong and Pintea, Silvia L and van Gemert, Jan C},
    title = {{NeRD++}: Improved 3D-mirror symmetry learning from a single image},
    year = {2022},
    booktitle = {BMVC},
}
```
