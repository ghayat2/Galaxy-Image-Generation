# EGMP - Galaxy Image Generation

## Prerequisites
- Python version 3.6
- Tensorflow 1.13 (https://www.tensorflow.org/versions/r1.13/api_docs/python/tf)
- scikit-image / skimage (https://scikit-image.org/)
- scikit-learn / sklearn (https://scikit-learn.org/stable/)
- NumPy (https://www.numpy.org/)
- Pandas (https://pandas.pydata.org/)
- seaborn (https://seaborn.pydata.org/)
- PyWavelets / pywt (https://pywavelets.readthedocs.io/en/latest/)
- Matplotlib (https://matplotlib.org/)
- tqdm (https://github.com/tqdm/tqdm)
- Pillow (https://pillow.readthedocs.io/en/stable/)

## Authors
- Gabriel Hayat
- Mounir Amrani
- Philipe Andreu
- Emilien Pilloud 

| File | Description
| :--- | :----------
| [StyleGAN] | Main folder.
| [StyleGAN] | Main folder.
| [StyleGAN] | Main folder.
| [StyleGAN] | Main folder.
| [StyleGAN] | Main folder.
| [StyleGAN] | Main folder.
| [StyleGAN] | Main folder.
| [StyleGAN] | Main folder.
| [StyleGAN] | Main folder.
| [StyleGAN] | Main folder.
| [StyleGAN] | Main folder.
| [StyleGAN] | Main folder.
| [StyleGAN] | Main folder.
| [StyleGAN] | Main folder.

## Baselines

### 1) Manual Feature Regressors

In order to train the Regressors based on manually extracted features, one should run the following command: 

`python3 baseline_score_regressor.py --regressor_type=`

### 1) XGBoost Regressor on manually generated features


To train the XGBoost regressor on the manually generated features of the scored images dataset and create a file prediction.csv of the predicted scores of the query images, run the following command: 

`python3 main.py -regressor_type="Boost" -only_features=True`

This command will reproduce the result mentioned in the report. Note that this command can be ran with multiple flags, the main ones are described here:

- regressor type: The type of the regressor, options are Random Forest, Ridge, MLP, Boost (default: None,  use our main  model to output predictions)
- vae_encoded_images: True if the images of scored and query dataset were previously encoded by the vae model (default: False)
- only_features: Train the regressor only on manually crafted features (default: False)
- feature_dim: Number of manually crafted features (Default: 34)
- latent_dim: The dimension of the latent space of the vae (default: 100)

## run experiments
### Setup
To run experiments, images must have been generated and placed in the 
`./generated_images` folder with the following hierarchy structure:
```
generated_images/
          |
          ---> 64/                # images 64x64
          |     |
          |     ---> model_1/
          |     |
          |     ---> model_2/
          |     |
          |     ---> .../
          |
          ---> 1000/              # images 1000x1000
                |
                ---> model_1/
                |
                ---> model_2/
                |
               ---> .../

```

To ensure all the needed Python package, run `install_python_dependencies.sh` which
relies on pip to install the needed modules.

### Extract manual features
If the manual features are not computed yet, i.e. there is no manual_features folder or
it is empty, run `python extract_features.py -I True -m [max #images to extract]`.
The advised maximum number of images is 100 otherwise it can take some time for 
1000x1000 images.

The manual_features folder should look like that now:
```
 manual_features/
          |
          ---> 64/                # images 64x64
          |     |
          |     ---> model_1/
          |     |
          |     ---> model_2/
          |     |
          |     ---> .../
          |
          ---> 1000/              # images 1000x1000
                |
                ---> model_1/
                |
                ---> model_2/
                |
               ---> .../
          legend.json
```

### run the experiments
Now that the directories are setup, just run `python run_experiments.py` with the wanted
parameters. The default show be suited for the first run. A legend json file can be 
put in `manual_features` to specify a dict from feature number to feature name. 
The experiment results can be found in the specified out directory or in 
`experiment_results` by default.

### python scripts functions

- `downsample.py`

    given an input folder full of images 1000x1000, resize them using the max pooling
    keras layers and the appropriate padding for min/max range -1/1.
    
- `run_experiments.py`

    run all the experiments and save the figures
    
- `extract_features.py`

    extract manual features from image (sub)-directories
