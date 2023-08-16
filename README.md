# 6D-Pose-Estimation-for-Unseen-Categories

Welcome to the forefront of exploration in 6D pose estimation. This project ventures into uncharted territory, delving into the realm of functional mapping to expand the boundaries of 6D pose estimation across previously unseen object categories. The project's nucleus lies in constructing globally-informed correspondences using learned functional mapping, with a grand objective of enabling the estimation of 6D poses for entirely novel objects and categories. This trailblazing approach aims to redefine the benchmarks of 6D pose estimation, ushering in an era of heightened precision and adaptability that transcends conventional constraints.

## Prerequisites

Before running the script, make sure you have the following prerequisites installed:

- Python 3.x
- PyTorch
- TensorBoard logging
- Open3D
- Other dependencies as listed in the requirements file

You can install the required dependencies using the following command:

```bash
pip install -r requirements.txt
```
Alternatively, you can employ the provided conda environment.yaml file to install the dependencies using:

```bash
conda env create -f environment.yml
```
## Cloning the Repo
Clone the repo using the following command:
```bash 
git clone --recursive https://github.com/m13ammed/6D-Pose-Estimation-for-Unseen-Categories
```
## Generating Cache for the datasets
Efficient caching of laplacian estimates for both CAD models and back-projected depth images is paramount for expedited performance. Generate this cache by running the following command:
```bash
cd scripts && python generate_cache.py
```
Customize the caching specifics within the config/cache_gen.gin file according to your needs. For further insights into available options, refer to the README within the config folder.

## Training 
Use the following command to start the training: 
```bash
cd scripts && python train.py
```
The training process incorporates tensorboard logs for monitoring. Leverage the configuration file config/dpfm_orig.gin to fine-tune settings. This script seamlessly accommodates training with single or multiple datasets, offering options for both naive and spatial consistency filtering when mapping functional mappings to point correspondences.

A vital note: Within the caching directory (cache_dir/dataset_name/split), a /mapping_list.npz file is generated for accelerated loading. Be careful when transitioning from training to evaluation scripts with the same dataset; ensure that the mapping list is removed.

## Corrrespondce Evaluation
Use the following command to start the evauation: 
```bash
cd scripts && python eval.py
```
The evaluation process hinges on settings specified in the config/dpfm_orig.gin configuration file. Be sure to peruse the README within the config folder for detailed insights into available options. When transitioning from training to evaluation scripts with the same dataset, ensure the mapping list is cleared. The config file also offers the choice to export results as .pt files, which are pivotal for subsequent pose estimation steps.


## Pose Estimation using RANSAC 
Use the following command to generate the results:
``` bash
cd scripts && python test_RANSAC.py <path to the generated .pt results folder> <path to where to save the final results>
```


## Pose Estimation using TEASER++
First you need to install the python bindings:
``` bash
sudo apt install cmake libeigen3-dev libboost-all-dev
cd TEASER-plusplus && mkdir build && cd build
cmake -DTEASERPP_PYTHON_VERSION=<YOUR_PYTHON_VERSION> .. && make teaserpp_python
cd python && pip install .
```
Now use the following command to generate the results:
``` bash
cd scripts && python test_teaser_.py <path to the generated .pt results folder> <path to where to save the final results>
```
## Visualizations
We additionally provide functions for visualizations in the scripts/visualization.py file:
(Note: requires desktop env by the visualizer)

###mesh
draw_basis(verts, faces, evecs, output="mesh_basis.png", evecs_selection=range(25,30), crop=[0.2, 0.1, 0.2, 0])

###point cloud
draw_basis(verts, None, evecs, output="cloud_basis.png", evecs_selection=range(25,30), crop=[0.2, 0.1, 0.2, 0])

draw_features(CAD, PC, Obj, C_pred[0], overlap_score12, overlap_score21, use_feat1[0], use_feat2[0], offset=[0,-18,0]) #assume input has no batch dim

draw_correspondence(p_pred, Obj, offset=None, raw_CAD_down_sample=10000, models_path=None)
