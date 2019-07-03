# EGMP - Galaxy Image Generation

## Authors
- Gabriel Hayat
- Mounir Amrani
- Philipe Andreu
- Emilien Pilloud 

## Prerequisites
- argparse (https://docs.python.org/3/library/argparse.html)
- Matplotlib (https://matplotlib.org/)
- NumPy (https://www.numpy.org/)
- Pandas (https://pandas.pydata.org/)
- pathlib (https://docs.python.org/3/library/pathlib.html)
- patool (https://wummel.github.io/patool/)
- Pillow (https://pillow.readthedocs.io/en/stable/)
- Python version 3.6 (https://www.python.org/downloads/release/python-360/)
- PyWavelets / pywt (https://pywavelets.readthedocs.io/en/latest/)
- scikit-image / skimage (https://scikit-image.org/)
- scikit-learn / sklearn (https://scikit-learn.org/stable/)
- seaborn (https://seaborn.pydata.org/)
- Tensorflow 1.13 (https://www.tensorflow.org/versions/r1.13/api_docs/python/tf)
- tqdm (https://github.com/tqdm/tqdm)
- xgboost (https://xgboost.readthedocs.io/en/latest/python/python_intro.html)

##Files

Note: For each and every file, there is a `--help` flag one can set in order to visualise all the flags one may set in a file. For more information about each of the following, please run:

```
python3 [filename] --help
```

| File | Description |  Runtime (hh:mm:ss)
| :--- | :---------- | :----------
| baseline\_patches.py | Generative Model based on patches | 00:04:54
| baseline\_score\_regressor\_test.py | Tests the different Regressor architectures on the query set
| baseline\_score\_regressor.py | Applies different Regressor architectures to the Regression task | 00:10:00
| data.py | Image/Manual Feature loading and preprocessing
| DCGAN\_Scorer.py | Scoring Model based on the Discriminator obtained in the DCGAN model
| DCGAN.py | Deep Convolutional Generative Adversarial Network
| downsample.py | Given input images at 1000x1000 resolution, this resizes them using max pooling with appropriate padding for range \[-1, 1\].
| extract\_features.py | Extracts manually crafted features from the provided files
| generate\_feats.py | Generates manually crafted features from the provided images
| install\_python\_dependencies.sh | Executable to install all dependencies
| layers.py | Layers used to build the Models
| MCGAN.py | Manual Feature Conditionned Generative Adversarial Network
| run\_experiments.py | Run statistics on the provided images
| StackedSRM.py | Stacked SuperResolution Model
| test\_DCGAN\_scorer.py | Tests the DCGAN based scoring model on the query set
| test\_DCGAN\_SRM\_Scorer.py | Generation file for the DCGAN with scorer filtering
| train\_DCGAN\_for\_score.py | Training file for the DCGAN based scoring model
| train\_DCGAN.py | Training File for the DCGAN | 02:13:35 without minibatch, 
| train\_MCGAN.py | Training File for the MCGAN | 02:08:12
| train\_stackedSRM.py | Training File for the Stacked SRM
| utils.py | Utility functions
| data | Data folder
| &boxvr;&nbsp; labeled | Labeled Image Directory
| &boxvr;&nbsp; scored | Scored Image Directory
| &boxvr;&nbsp; query | Query Image Directory
| &boxvr;&nbsp; labeled.csv | Labeled Images' Labels
| &boxur;&nbsp; scored.csv | Scored Images' Scores

## Preparing datasets for training

The entire cosmology\_aux\_data\_170429 dataset should be be put into a directory called 'data/' as detailed above. Then, one should proceed and generate the 38 manual features for all the images in the dataset as follows:

```
> python3 generate_feats.py 
```

This will generate a new folder called "features" in which one will find The feature files, and id files in order to map them to the image they are from:

| File | Description
| :--- | :----------
| data | Data folder.
| &boxur;&nbsp; features | Features folder.
| &ensp;&ensp; &boxvr;&nbsp; labeled\_feats.gz | Labeled Features
| &ensp;&ensp; &boxvr;&nbsp; labeled\_feats\_ids.gz | Labeled Features ID Correspondences
| &ensp;&ensp; &boxvr;&nbsp; scored\_feats.gz | Scored Features
| &ensp;&ensp; &boxvr;&nbsp; scored\_feats\_ids.gz | Scored Features ID Correspondences
| &ensp;&ensp; &boxvr;&nbsp; query\_feats.gz | Query Features
| &ensp;&ensp; &boxur;&nbsp; query\_feats\_ids.gz | Query Features ID Correspondences

## Models

### A) Image Generation

#### 1) Patches

The model can be found in `baseline_patches.py`.

```
> python3 baseline_patches.py 
```
This generates a `LOG_PATCHES` folder with the following structure:

| File | Description
| :--- | :----------
| LOG\_PATCHES | Main folder.
| &boxur;&nbsp; generated_samples | Directory containing the generated samples

#### 2) DCGAN

The model can be found in `DCGAN.py`. 

```
> python3 train_DCGAN.py 
```

This generates a `LOG_DCGAN` folder with the following structure:

| File | Description
| :--- | :----------
| LOG\_DCGAN | Main folder.
| &boxur;&nbsp; [date-time] | Date and time designating each run
| &ensp;&ensp; &boxvr;&nbsp; checkpoints | Directory containing the last 5 checkpoints saved
| &ensp;&ensp; &boxvr;&nbsp; code.zip | Zip file containing the code used to produce the result
| &ensp;&ensp; &boxvr;&nbsp; output | Output generated by the run
| &ensp;&ensp; &boxur;&nbsp; test_samples | Directory containing the test sample images generated during training

#### 3) MCGAN

The model can be found in `MCGAN.py`.

```
> python3 train_MCGAN.py 
```

This generates a `LOG_MCGAN` folder with the following structure:

| File | Description
| :--- | :----------
| LOG\_MCGAN | Main folder.
| &boxur;&nbsp; [date-time] | Date and time designating each run
| &ensp;&ensp; &boxvr;&nbsp; checkpoints | Directory containing the last 5 checkpoints saved
| &ensp;&ensp; &boxvr;&nbsp; code.zip | Zip file containing the code used to produce the result
| &ensp;&ensp; &boxvr;&nbsp; output | Output generated by the run
| &ensp;&ensp; &boxur;&nbsp; test_samples | Directory containing the test sample images generated during training

### B) Image Scoring

#### 1) Manual Feature Regressors

In order to train the Regressors based on manually extracted features, one should run the following command: 

```
> python3 baseline_score_regressor.py --regressor_type=[reg_name]
```
Replace reg_name with one of the following arguments for different regressors:

| Argument | Description
| :--- | :----------
| Boost | XGBoost Regressor
| Ridge | Ridge Regressor
| Random\_Forest | Random Forest Regressor

This generates a `Regressor` folder with the following structure:

| File | Description
| :--- | :----------
| Regressor | Main folder.
| &boxur;&nbsp; predictions | 
| &ensp;&ensp; &boxur;&nbsp; predictions.csv | The Regressor's predictions on the query set
| &boxur;&nbsp; checkpoints | Directory containing the checkpoints saved

#### 2) Neural Network Regressors

The model can be found in `DCGAN.py` and `DCGAN_Scorer.py`. 

```
> python3 train_DCGAN_for_score.py 
```
This generates a `LOG_DCGAN_SCORER` folder with the following structure:

| File | Description
| :--- | :----------
| LOG\_DCGAN\_SCORER | Main folder.
| &boxur;&nbsp; [date-time] | Date and time designating each run
| &ensp;&ensp; &boxvr;&nbsp; checkpoints | Directory containing the last 5 checkpoints saved
| &ensp;&ensp; &boxvr;&nbsp; code.zip | Zip file containing the code used to produce the result
| &ensp;&ensp; &boxur;&nbsp; output | Output generated by the run

## Run Experiments
### Setup
To run experiments, images must have been generated and placed in the 
`./generated_images` folder with the following hierarchy structure:

| File | Description
| :--- | :----------
| generated_images | Generated Images folder.
| &boxvr;&nbsp; legend.json | Specifies correspondences between feature numbers and feature names
| &boxvr;&nbsp; 64 | 64x64 images
| &ensp;&ensp; &boxvr;&nbsp; model_1 | Folder containing the first model's images to run experiments on
| &ensp;&ensp; &boxvr;&nbsp; model_2 | Folder containing the second model's images to run experiments on
| &ensp;&ensp; &boxvr;&nbsp; ... | 
| &boxur;&nbsp; 1000 | 1000x1000 images
| &ensp;&ensp; &boxvr;&nbsp; model_1 | Folder containing the first model's images to run experiments on
| &ensp;&ensp; &boxvr;&nbsp; model_2 | Folder containing the second model's images to run experiments on
| &ensp;&ensp; &boxvr;&nbsp; ... | 


To ensure all the needed Python packages are installed, run the following which
relies on pip to install the required modules:

```
install_python_dependencies.sh
```

### Manual Feature Extraction
If the manual features are not computed yet, i.e. there is no manual_features folder or
it is empty, run: 

```
python3 extract_features.py -I True -m [max #images to extract]
```

The advised maximum number of images is 100 otherwise it can take some time for 
1000x1000 images.

The manual_features folder should now look as follows:

| File | Description
| :--- | :----------
| manual_features | Generated Images folder.
| &boxvr;&nbsp; legend.json | Specifies correspondences between feature numbers and feature names
| &boxvr;&nbsp; 64 | 64x64 images
| &ensp;&ensp; &boxvr;&nbsp; model_1 | Folder containing the first model's manual features
| &ensp;&ensp; &boxvr;&nbsp; model_2 | Folder containing the second model's manual features
| &ensp;&ensp; &boxvr;&nbsp; ... | 
| &boxur;&nbsp; 1000 | 1000x1000 images
| &ensp;&ensp; &boxvr;&nbsp; model_1 | Folder containing the first model's manual features
| &ensp;&ensp; &boxvr;&nbsp; model_2 | Folder containing the second model's manual features
| &ensp;&ensp; &boxvr;&nbsp; ... | 

### Experiment Execution
Now that the directories are setup, all that remains is to run:

```
python3 run_experiments.py
```
 
 A legend json file can be put in `manual_features` to specify a dict from feature number to feature name. 
The experiment results will be output to the specified out directory or in `experiment_results` by default.


<!--### 1) XGBoost Regressor on manually generated features


To train the XGBoost regressor on the manually generated features of the scored images dataset and create a file prediction.csv of the predicted scores of the query images, run the following command: 

`python3 main.py -regressor_type="Boost" -only_features=True`

This command will reproduce the result mentioned in the report. Note that this command can be ran with multiple flags, the main ones are described here:

- regressor type: The type of the regressor, options are Random Forest, Ridge, MLP, Boost (default: None,  use our main  model to output predictions)
- vae_encoded_images: True if the images of scored and query dataset were previously encoded by the vae model (default: False)
- only_features: Train the regressor only on manually crafted features (default: False)
- feature_dim: Number of manually crafted features (Default: 34)
- latent_dim: The dimension of the latent space of the vae (default: 100)-->