# 6D-Pose-Estimation-for-Unseen-Categories - Configuration Guide

This README provides a detailed guide for configuring the 6D-Pose-Estimation-for-Unseen-Categories training pipeline using GIN configuration files. DPFM is a powerful method for 6D pose estimation for unseen categories. Follow the steps below to customize the configuration to your liking.

## Table of Contents

1. [Introduction](#introduction)
2. [Dataset Configuration](#dataset-configuration)
3. [Model Configuration](#model-configuration)
4. [Loss Configuration](#loss-configuration)
5. [Training Configuration](#training-configuration)
6. [Cache Generation Configuration](#cache-generation-configuration)
7. [Adjusting for Evaluation](#adjusting-for-evaluation)

## Introduction

The DPFM model is configured using GIN (Gin Is Not) configuration files. This guide walks you through configuring various aspects of the training pipeline. Below, you'll find details about dataset setup, model configuration, loss settings, training parameters, and evaluation adjustments.

## Dataset Configuration

Configure the datasets for training and evaluation using the `base_object_dataset` section in the GIN file. This section contains common parameters for all datasets and allows you to create specialized configurations for specific datasets.

### Common Dataset Parameters

- `mode`: Set the dataset mode (e.g., 'train_pbr', 'train', 'test', etc.).
- `num_samples`: Number of samples to use from the dataset (use -1 for all).
- `min_vis`: Minimum visibility of an object instance.
- `cache_dir`: Directory for caching dataset files.
- `LBO_pc`: Generate Laplacian-Beltrami Operator (LBO) for point clouds.

### Specialized Dataset Parameters

Use `scope/base_object_dataset` to create specialized dataset configurations.

- `render_data_name`: Specify the dataset name.
- `num_samples`: Number of samples to use from the dataset.
- `obj_take`: List of object IDs to include in the dataset.

### Training Datasets

For training with multiple datasets, use `prepare_train_datasets.datasets_list` to specify datasets used for training. Use multiple dataset configurations for multi-dataset training.

For example, for multi-dataset training with `train_lm`, `train_hb`, and `train_ycbv` datasets:
```
prepare_train_datasets.datasets_list = ([@train_lm/base_object_dataset,@train_hb/base_object_dataset, @train_ycbv/base_object_dataset])
```

For single-dataset training, specify only the relevant dataset configuration:
```
prepare_train_datasets.datasets_list = ([@train_lm/base_object_dataset])
```
## Model Configuration

Configure the DPFM model using the `DPFMNet` section in the GIN file.

- `cfg`: Path to a YAML configuration file containing model parameters.

## Loss Configuration

Configure the loss function used for training using the `DPFMLoss` section.

- `w_fmap`: Weight for the functional map loss.
- `w_acc`: Weight for the accuracy loss.
- `w_nce`: Weight for the NCE loss.
- `nce_t`: Temperature parameter for NCE loss.
- `nce_num_pairs`: Number of negative pairs for NCE loss.

## Training Configuration

Customize the training process using the `train_net` section.

- `model`: Instantiate the DPFM model using `DPFMNet`.
- `optimizer`: Select an optimizer (e.g., `RMSprop`, `Adam`, etc.).
- `criterion`: Specify the loss function using `DPFMLoss`.
- `decay_iter`: Iteration interval for learning rate decay.
- `decay_factor`: Factor for learning rate decay.
- `epochs`: Number of training epochs.
- `logging_dir`: Directory to save training logs and checkpoints.
- `comment`: Additional comment for Tensorboard folders.

Customize the functional mapping to point solver process using the `choose_fmap2pointmap_solver` section.
- `solver`: Choosing the solver. Currently @spacial_filtering_fmap2pointmap and @naive_fmap2pointmap are supported

### Additional Training Configurations

- `set_env_variables.pretrained_model`: Path for the model to load (for training and eval) or None. Does not load optimizer parameters.
- `set_env_variables.data_root`: Path to the original datasets.


## Cache Generation Configuration

To generate cache files for datasets, you can create a cache generation configuration using a `.gin` file. Here's an example configuration:

```ini
set_env_variables.data_root = "/data/unseen_object_data/tmp/full_dataset_setup"
base_object_dataset.render_data_name = "lmo"

set_env_variables.BS = 8  #batch size
set_env_variables.num_workers = 12

base_object_dataset.mode = 'test'
base_object_dataset.num_samples = 1000
base_object_dataset.min_vis = 0.3

base_object_dataset.cache_dir = '/data/unseen_object_data/tmp/cache_new'
base_object_dataset.LBO_pc = True
```

## Adjusting for Evaluation

For evaluation purposes, use the `eval/base_object_dataset` section.

- `render_data_name`: Specify the evaluation dataset name.
- `num_samples`: Number of samples for evaluation.
- `obj_take`: List of object IDs for evaluation.

Additionally one can set path to save results using the set_env_variables secition.
- `save_results`: path to save the results in. These files are then used to generate poses, if you dont want to save keep as None.

## Conclusion

By following this guide, you can configure the DPFM training pipeline to suit your specific requirements. Adjust dataset parameters, model settings, loss configurations, training parameters, and evaluation settings to achieve the best results for your 6D pose estimation tasks.