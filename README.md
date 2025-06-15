# Image Reconstruction: CT Denoising
Semester Project, EPFL Center for Imaging


This repository contains the code and results presented in the semester project report: **"Image Reconstruction: CT Denoising "**.

## Project Overview

The key results discussed in the report are available in the notebook `CT-reconstruction.ipynb`. This notebook presents experiments and evaluations of different denoising strategies (pre-processing FBP, post-processing FBP, and model-based with FBP as initial guess) for CT-reconstruction using both signal processing and machine learning approaches.

**Note:** The notebook also includes additional results, especially for the model-based approach, that go beyond those included in the final report.

## Repository Structure
### `CNC/`
Contains the implementation of the **CNC neural network regularizer** in both **JAX** and **PyTorch**. This model is used in the model-based approach for plug-and-play regularization.
/!\ **Note:** The JAX version is still under development and currently not usable.

### `data/`
Includes the input images used in the `CT-reconstruction.ipynb` notebook for testing.

### `maxim/`
Holds the pretrained **MAXIM neural network**, used for both **pre-processing** and **post-processing** denoising approaches.

### `CT-reconstruction.ipynb`
Main notebook of the project. It contains all the experimental results, including reconstructions, evaluations (CNR, SNR), and comparisons of all methods discussed in the report.

### `helper.py`
An utility script containing helper functions and algorithmic implementations used in the main notebook to improve readability and modularity. It also contains functions to generate synthetic datasets for training neural networks.

### `run_eval.py`
Includes evaluation functions specifically used for inference and metric computation with **MAXIM** model.

## Requirements
The project requires Python ≥ 3.11. Please see `requirements.txt` for the list of dependencies.

To install all dependencies:

```bash
pip install -r requirements.txt
```
Furthermore, To run the cells related to the **MAXIM** model in `CT-reconstruction.ipynb`, you need to manually download the pretrained checkpoint:

1. Download the model checkpoint from the following link:  
   [MAXIM Denoising Checkpoint – SIDD](https://console.cloud.google.com/storage/browser/gresearch/maxim/ckpt/Denoising/SIDD;tab=objects?inv=1&invt=Ab0MTQ&prefix=&forceOnObjectsSortingFiltering=false)

2. Rename the downloaded file to: ckpt_Denoising_SIDD.npz

3. Move it to the following directory in the repository: `./maxim/data/`

After this, you will be able to run the MAXIM-related inference and evaluation cells in the notebook.