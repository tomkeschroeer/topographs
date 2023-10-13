# TopoGraphs

## Preprocessing

If you don't already have a suitable dataset you first have to run the preprocessing in order to obtain one. The data fed in has to contain, besides the input features, the following truth information:

on track level:
```
truthOriginLabel  
flavour
regression variable (e.g. hadron pt)
```
on jet level:
```
HadronConeExclExtendedTruthLabelID  
```
The needed fields to be filled in the config file are listed in the table below.
|field | description   |  
|---|---|  
| input | input files in .h5 format.|  
| output | output directory for the preprocessed files |  
| output_training | output directory for the model checkpoints and later for plots etc. All outputs that are produced during and after the training |  
| model_name | name of the training run that is used for weights and biases (wandb) |  
| preprocessing_file_name | name of the output file for the preprocessing |
| scale_dict | name of the scale dict where scaling and shifting parameters for the input features and regression variable for each jet flavour |  
| training_file_name | in the last step the preprocessing file will be split into a training, testing and validation file. Here, define the name of training file|
| testing_file_name | define the name of testing file |  
| validation_file_name | define the name of validation file |  
| njets | number of jets for the training file (total number, not per jet flavour!) |
| njets_val | number of jets for the validation file (total number, not per jet flavour!)  |
| njets_test | number of jets for the testing file (total number, not per jet flavour!)  |
| input_tracks_name | name of dataset group in .h5 file that includes the track inputs, the truthOriginLabel and regression variable(s) | 
| input_truth_name | name of dataset group in .h5 file that includes the flavour variable |
| input_jet_name | name of dataset group in .h5 file that includes the jet information (HadronConeExclExtendedTruthLabelID) |

to start the preprocessing run the following commands. First you have to calculate all the labels and save the inputs in the preprocessing file:
```
python preprocessing.py -c configs/config.yaml -p
```
Then, all preprocessed files are merged and the jet flavours saved:
```
python preprocessing.py -c configs/config.yaml -m
```
The scaling and shifting parameters for each input and separately for each jet flavour for the vertex labels are calculated:
```
python preprocessing.py -c configs/config.yaml -s
```
Apply scaling and shifting paramers:
```
python preprocessing.py -c configs/config.yaml -a
```
After the preprocessing you should have a training, a testing and a validation file.

## Training
