# CIL-Galaxies

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


