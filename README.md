# CIL-Galaxies

## Specs
* Python version 3.6
* Tensorflow 2.0.0-beta1

## Authors
- Gabriel Hayat
- Mounir Amrani
- Hidde Lycklama à Nijeholt
- Emilien Pilloud 

## Baselines
### 1) Random Forest on manually generated features


To train a Random Forest regressor on the manually generated features of the scored images dataset and create a file prediction.csv of the predicted scores of the query images, run the following command: 

`python3 main.py -regressor_type="Random Forest" -only_features=True`

This command will reproduce the result mentioned in the report. Note that this command can be ran with multiple flags, the main ones are described here:

- regressor type: The type of the regressor, options are default: None use our main  model to output predictions, Random Forest, Ridge, MLP
- vae_encoded_images: True if the images of scored and query dataset were previously encoded by the vae model (default: False)
- only_features: Train the regressor only on manually crafted features (default: False)
- feature_dim: Number of manually crafted features (Default: 34)
- latent_dim: The dimension of the latent space of the vae (default: 100)



