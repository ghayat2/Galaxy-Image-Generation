# Team EGMP - Galaxy Image Generation

## Authors
- Gabriel Hayat
- Mounir Amrani
- Philipe Andreu
- Emilien Pilloud 

## Getting Started

### Virtual environment
Install virtualenvwrapper if not already done:
```
pip install virtualenvwrapper
```

Create a new virtual environment:

```
mkvirtualenv -p python3.6 "env-name"
```

The virtual environment is by default activated. You can disable it and enable it using:

```
deactivate
workon "env-name"
```

Install pip requirements:

```
pip install -r requirements.txt
```
To remove the virtual environment, simply disable it as shown above then run:
```
rmvirtualenv "env-name"
```
## Preparing datasets for training

Please put the image folders `labeled`, `query` and `scored` and the csv files `labeled.csv` and `scored.csv` under the same directory called `data` that should be in the same directory as the `*.py` scripts. 
Then, to generate the 38 manual features for all the images in the `labeled`, `query` and `scored` datasets, please run:

```
python3 generate_feats.py 
```

This will generate a new folder called `features` under `data` in which you can find the features files and the corresponding ids files (used to map each feature vector to the id of the corresponding image). The final directory structure should be:

| File | Description
| :--- | :----------
| data | Data folder.
| &boxvr;&nbsp; labeled | Labeled Image Directory
| &boxvr;&nbsp; scored | Scored Image Directory
| &boxvr;&nbsp; query | Query Image Directory
| &boxvr;&nbsp; labeled.csv | Labeled Images' Labels
| &boxvr;&nbsp; scored.csv | Scored Images' Scores
| &boxur;&nbsp; features | Features folder.
| &ensp;&ensp; &boxvr;&nbsp; labeled\_feats.gz | Labeled Features
| &ensp;&ensp; &boxvr;&nbsp; labeled\_feats\_ids.gz | Labeled Features ID Correspondences
| &ensp;&ensp; &boxvr;&nbsp; query\_feats.gz | Query Features
| &ensp;&ensp; &boxvr;&nbsp; query\_feats\_ids.gz | Query Features ID Correspondences
| &ensp;&ensp; &boxvr;&nbsp; scored\_feats.gz | Scored Features
| &ensp;&ensp; &boxur;&nbsp; scored\_feats\_ids.gz | Scored Features ID Correspondences

## Running the models

### A) Generation models

#### 1) FullresGAN

The model definition can be found in `FullresGAN.py`. To train the model with the default parameters, simply run:

```
python3 train_FullresGAN.py 
```

This generates a `LOG_FullresGAN` folder with the following structure:

| File | Description
| :--- | :----------
| LOG\_FullresGAN | FullresGAN logs folder.
| &boxur;&nbsp; [date-time] | Date and time of the run
| &ensp;&ensp; &boxvr;&nbsp; checkpoints | Directory containing the last 5 checkpoints saved
| &ensp;&ensp; &boxvr;&nbsp; code.zip | Zip file containing the code used for the run
| &ensp;&ensp; &boxvr;&nbsp; output | Messages printed to standard output
| &ensp;&ensp; &boxur;&nbsp; test_samples | Directory containing the test sample images generated during training

#### 2) DCGAN

The model definition can be found in `DCGAN.py`. To train the model with the default parameters, simply run:

```
python3 train_DCGAN.py 
```

This generates a `LOG_DCGAN` folder with the following structure:

| File | Description
| :--- | :----------
| LOG\_DCGAN | DCGAN logs folder.
| &boxur;&nbsp; [date-time] | Date and time of the run
| &ensp;&ensp; &boxvr;&nbsp; checkpoints | Directory containing the last 5 checkpoints saved
| &ensp;&ensp; &boxvr;&nbsp; code.zip | Zip file containing the code used for the run
| &ensp;&ensp; &boxvr;&nbsp; output | Messages printed to standard output
| &ensp;&ensp; &boxur;&nbsp; test_samples | Directory containing the test sample images generated during training

#### 3) MCGAN

The model definition can be found in `MCGAN.py`. To train the model with the default parameters, simply run:

```
python3 train_MCGAN.py 
```

This generates a `LOG_MCGAN` folder with the following structure:

| File | Description
| :--- | :----------
| LOG\_MCGAN | MCGAN logs folder.
| &boxur;&nbsp; [date-time] | Date and time of the run
| &ensp;&ensp; &boxvr;&nbsp; checkpoints | Directory containing the last 5 checkpoints saved
| &ensp;&ensp; &boxvr;&nbsp; code.zip | Zip file containing the code used for the run
| &ensp;&ensp; &boxvr;&nbsp; output | Messages printed to standard output
| &ensp;&ensp; &boxur;&nbsp; test_samples | Directory containing the test sample images generated during training

#### 3) Stacked Super Resolution Model (SRM)

The model definition can be found in `StackedSRM.py`. To train the model with the default parameters, simply run:

```
python3 train_stackedSRM.py
```
Please make sure that you have enough GPU memory to run the model to avoid memory exceptions towards the end of training. Alternatively you can download the log directory containing model checkpoint (trained on Google Colab's Tesla T4 GPU with 16GB of memory for around 3h) from the following link:
```
https://polybox.ethz.ch/index.php/s/c7DHpbXWf76EDp1
```
Please extract the archive file and put the `LOG_SRM` folder in the same directory as the `*.py` scripts.<br />

When running the model training, this generates a `LOG_SRM` folder with the following structure:

| File | Description
| :--- | :----------
| LOG\_SRM | SRM logs folder.
| &boxur;&nbsp; [date-time] | Date and time of the run
| &ensp;&ensp; &boxvr;&nbsp; checkpoints | Directory containing the last 5 checkpoints saved
| &ensp;&ensp; &boxvr;&nbsp; code.zip | Zip file containing the code used for the run
| &ensp;&ensp; &boxvr;&nbsp; output | Messages printed to standard output
| &ensp;&ensp; &boxur;&nbsp; samples | Directory containing samples of the output of the SRM model on the training data.

### B) Image Scoring

#### 1) Manual Feature Regressors

In order to train the Regressors based on manually extracted features, you can simply run: 

```
python3 baseline_score_regressor.py --regressor_type <reg_name>
```
Replace <reg_name\> with one of the following arguments for different regressors:

| Argument | Description
| :--- | :----------
| Boost | XGBoost Regressor
| Ridge | Ridge Regressor
| Random\_Forest | Random Forest Regressor

This generates a `Regressor` folder with the following structure:

| File | Description
| :--- | :----------
| Regressor | Main folder.
| &boxur;&nbsp; [reg_name] | Trained Regressor Name
| &ensp; &boxvr;&nbsp; checkpoints | Directory containing the checkpoints saved
| &ensp; &boxur;&nbsp; predictions | 
| &ensp;&ensp; &boxur;&nbsp; predictions.csv | The Regressor's predictions on the query set


#### 2) DCGAN Scoring head:

The model definition can be found in `DCGAN_Scorer.py`. This model adds and trains a simple Feed-Forward Neural Network to the discriminator of the DCGAN model. So, since this model depends on a trained DCGAN model, please make sure to train the DCGAN model first then train the scoring head. <br />
To train this model using the latest trained DCGAN model and the default parameters, simply run: 

```
python3 train_DCGAN_for_score.py 
```
This generates a `LOG_DCGAN_SCORER` folder with the following structure:

| File | Description
| :--- | :----------
| LOG\_DCGAN\_SCORER | Main folder.
| &boxur;&nbsp; [date-time] | Date and time of the run
| &ensp;&ensp; &boxvr;&nbsp; checkpoints | Directory containing the last 5 checkpoints saved
| &ensp;&ensp; &boxvr;&nbsp; code.zip | Zip file containing the code used for the run
| &ensp;&ensp; &boxur;&nbsp; output | Messages printed to standard output

Once the training done, you can generate score predictions on the `query` dataset by running:
```
python3 test_DCGAN_scorer.py
```
This would load the latest trained DCGAN Scorer model and generate a `predictions` folder containing the `predictions.csv` file. The directory structure is as follows:

| File | Description
| :--- | :----------
| LOG\_DCGAN\_SCORER | Main folder.
| &boxur;&nbsp; [date-time] | Date and time of the run of the loaded model
| &ensp;&ensp; &boxur;&nbsp; predictions | Directory containing the predictions
| &ensp;&ensp;&ensp;&ensp; &boxur;&nbsp; [date-time] | Date and time of the score predictions generation
| &ensp;&ensp;&ensp;&ensp;&ensp;&ensp; &boxur;&nbsp; predictions.csv | Score predictions file


### C) Image Generation
#### 1) Generation using Patches from the labeled Dataset

To run this model, simply run:

```
python3 baseline_patches.py 
```
This generates a `LOG_PATCHES` folder with the following structure:

| File | Description
| :--- | :----------
| LOG\_PATCHES | Patches baseline logs folder.
| &boxur;&nbsp; [date-time] | Date and time of the run
| &ensp;&ensp; &boxvr;&nbsp; output | Messages printed to standard output
| &ensp;&ensp; &boxur;&nbsp; generated_samples | Directory containing the generated samples
| &ensp;&ensp;&ensp;&ensp;  &boxur;&nbsp; 1000 | Directory containing generated images of size 1000x1000

#### 2) Generation using GAN models (with optional Filtering using scorer model):
You can use the code in `test_GAN_SRM_Scorer.py` to execute the generation pipeline: this consits in generating an image using a GAN model, then upsampling the image using the SRM model (in case the generated image is 64x64), then optionally scoring the image with a chosen scoring model in order to filter or keep it. By default 100 images are generated. <br>

There are 2 strategies for deciding whether or not to keep an image when using a scorer:

- Keeping only images with a score above a certain threshold `t` (by default 3.0).
 - Scoring the images of the `labeled` dataset that represent galaxies (i,e labeled `1`), then taking the mean of these scores and adding a margin `m` (by default 0.25) to this mean in order to get a threshold `t`. If an image has a score below `t`, it is filtered, otherwise it is kept. 

To run the generation pipeline, simply run:
```
python3 test_GAN_SRM_Scorer.py --generator <GENERATOR> --scorer <SCORER> --use_threshold --threshold <THRESHOLD>
```
The option `--scorer` is optional and can be ommited. <br>
To use a margin on the mean score of the `labeled` galaxy images, replace `--use_threshold` by `--use_margin` and `--threshold <THRESHOLD>` by `--margin <MARGIN>`.

To view the list of allowed values for <GENERATOR\> and <SCORER\>, please run:
```
python3 test_GAN_SRM_Scorer.py --help
```
The generated images are stored in the folder `LOG_COMBINED` with the following structure:

| File | Description
| :--- | :----------
| LOG\_COMBINED | Logs folder.
| &boxur;&nbsp; [date-time] | Date and time of the run
| &ensp;&ensp; &boxvr;&nbsp; output | Messages printed to standard output
| &ensp;&ensp; &boxur;&nbsp; generated_samples | Directory containing the generated samples
| &ensp;&ensp;&ensp;&ensp;  &boxvr;&nbsp; 64 | Directory containing generated images of size 64x64 (if model generates 64x64 images)
| &ensp;&ensp;&ensp;&ensp;  &boxur;&nbsp; 1000 | Directory containing generated images of size 1000x1000

## Run Experiments
### Setup
To run experiments, images must have been generated and placed in the 
`./generated_images` folder with the following hierarchy structure:

| File | Description
| :--- | :----------
| generated_images | Generated Images folder.
| &boxvr;&nbsp; legend.json | (Optional) Specifies correspondences between feature indices and feature names
| &boxvr;&nbsp; model_1 | Directory with the name of the model
| &ensp;&ensp; &boxvr;&nbsp; 64 | Folder with the 64x64 images generated (optional if not generated)
| &ensp;&ensp; &boxur;&nbsp; 1000 | Folder with the 1000x1000 images generated
| &boxvr;&nbsp; model_2 | Directory with the name of the model
| &ensp;&ensp; &boxvr;&nbsp; 64 | Folder with the 64x64 images generated (optional if not generated)
| &ensp;&ensp; &boxur;&nbsp; 1000 | Folder with the 1000x1000 images generated
| &boxvr;&nbsp; ... | 

### Manual Feature Extraction
Compute the manual features on the generated images of all models under the folder `./generated_images` by running: 

```
python3 extract_features.py
```

The results are stored in `./manual_features` folder with a structure similar to the `./generated_images` folder:

| File | Description
| :--- | :----------
| manual_features | Generated Images folder.
| &boxvr;&nbsp; model_1 | Directory with the name of the model
| &ensp;&ensp; &boxvr;&nbsp; 64 | Folder with the features extracted on 64x64 images
| &ensp;&ensp; &boxur;&nbsp; 1000 | Folder with the features extracted on 1000x1000 images
| &boxvr;&nbsp; model_2 | Directory with the name of the model
| &ensp;&ensp; &boxvr;&nbsp; 64 | Folder with the features extracted on 64x64 images
| &ensp;&ensp; &boxur;&nbsp; 1000 | Folder with the features extracted on 1000x1000 images
| &boxvr;&nbsp; ... | 

### Experiment Execution
Now that the directories are setup and features extracted, all that remains is to run all experiments using:

```
python3 run_experiments.py -all
```
 
An optional path to a legend json file can be specified using the option `--legend <FILE_PATH>` to specify a dict from feature index to feature name. Otherwise, the feature index will be used as feature name.

The experiment results can be found uder the directory `./experiments_results` by default.

## Other options

Note: For many `*.py` files, you can get a some help information about the possible settable options and their descriptions by simply running:

```
python3 [filename] --help
```
## Files
| File | Description |  Runtime (hh:mm:ss)
| :--- | :---------- | :----------
| baseline\_patches.py | Generative Model based on patches | 00:04:54
| baseline\_score\_regressor\_test.py | Tests the different Regressor architectures on the query set | 00:05:00
| baseline\_score\_regressor.py | Applies different Regressor architectures to the Regression task | 00:10:00
| data.py | Image/Manual Feature loading and preprocessing | ------------
| DCGAN\_Scorer.py | Scoring Model based on the Discriminator obtained in the DCGAN model | ------------
| DCGAN.py | Deep Convolutional Generative Adversarial Network | ------------
| downsample.py | Given input images at 1000x1000 resolution, this resizes them using max pooling with appropriate padding for range \[-1, 1\] | Depends on number of files given
| extract\_features.py | Extracts manually crafted features from the provided files | Depends on number of files given
| generate\_feats.py | Generates manually crafted features from the provided images | 
| layers.py | Layers used to build the Models | ------------
| MCGAN.py | Manual Feature Conditionned Generative Adversarial Network | ------------
| run\_experiments.py | Run statistics on the provided images
| StackedSRM.py | Stacked SuperResolution Model | ------------
| test\_DCGAN\_scorer.py | Tests the DCGAN based scoring model on the query set
| test\_GAN\_SRM\_Scorer.py | Generation file for the DCGAN with scorer filtering | Depends on score value to filter according to 
| train\_DCGAN\_for\_score.py | Training file for the DCGAN based scoring model
| train\_DCGAN.py | Training File for the DCGAN | 02:13:35 
| train\_MCGAN.py | Training File for the MCGAN | 02:08:12
| train\_stackedSRM.py | Training File for the Stacked SRM
| utils.py | Utility functions | ------------

<!--### 1) XGBoost Regressor on manually generated features


To train the XGBoost regressor on the manually generated features of the scored images dataset and create a file prediction.csv of the predicted scores of the query images, run the following command: 

`python3 main.py -regressor_type="Boost" -only_features=True`

This command will reproduce the result mentioned in the report. Note that this command can be ran with multiple flags, the main ones are described here:

- regressor type: The type of the regressor, options are Random Forest, Ridge, MLP, Boost (default: None,  use our main  model to output predictions)
- vae_encoded_images: True if the images of scored and query dataset were previously encoded by the vae model (default: False)
- only_features: Train the regressor only on manually crafted features (default: False)
- feature_dim: Number of manually crafted features (Default: 34)
- latent_dim: The dimension of the latent space of the vae (default: 100)-->
