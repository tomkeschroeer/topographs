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
|field | description   | default value |
|---|---|---|
| input | input files in .h5 format.|  - 
| output | output directory for the preprocessed files | -
| output_training | output directory for the model checkpoints and later for plots etc. All outputs that are produced during and after the training | - |
| model_name | name of the training run that is used for weights and biases (wandb) | - |
| preprocessing_file_name | name of the output file for the preprocessing | preprocessed_ttbar_topograph.h5
| scale_dict | name of the scale dict where scaling and shifting parameters for the input features and regression variable for each jet flavour | scale_dict_ttbar.json |
| training_file_name | in the last step the preprocessing file will be split into a training, testing and validation file. Here, define the name of training file| training_ttbar_topographs.h5 |
| testing_file_name | define the name of testing file | testing_ttbar_topographs.h5 |
| validation_file_name | define the name of validation file | validation_ttbar_topographs.h5  |
| njets | number of jets for the training file (total number, not per jet flavour!) | 1_000_000 |
| njets_val | number of jets for the validation file (total number, not per jet flavour!)  | 500_000 |
| njets_test | number of jets for the testing file (total number, not per jet flavour!)  | 50_000 |
| input_tracks_name | name of dataset group in .h5 file that includes the track inputs, the truthOriginLabel and regression variable(s) | tracks_loose |
| input_truth_name | name of dataset group in .h5 file that includes the flavour variable | truth_hadrons |
| input_jet_name | name of dataset group in .h5 file that includes the jet information (HadronConeExclExtendedTruthLabelID) | jets |

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

for the training the following properties have to be defined:

|field | description | default value |
|---|---|---|
jet_types | jet flavours that should be used for the training | ["b"] |
small_net | dictionary that says for each jet flavour if a regression task has to be performed (False if regression should be performed) | {"b": False} |
|tracks_name| dataset name for track inputs | X_train_tracks  |
|edge_name| dataset name for edge labels |Y_edge |
|vertex_feat_name| dataset name for regression labels | Y_vertex_features |
|epochs| number of epochs to train for | 200 |
|lr| learning rate | 0.01 |
|use_sample_weights| If sample weights should be used for the edge training | True |
|loss_fac_edge| weighting factor for the edge loss | 1 |
|loss_fac_vert| weighting factor for the regression loss | 1 |
| edge_feature_network | see below | see below |
| edge_weight_network | see below | see below |
| vertex_network | see below | see below |

for the different parts of the networks you have to define the number of nodes:

|network | field | description | default value |  
|---|---|---|---|
| edge_feature_network | nodes | list of number of nodes per layer | [21, 20, 20, 30] |  
| edge_weight_network | nodes | list of number of nodes per layer | [21, 20, 20, 1] |  
| vertex_network | nodes | list of number of nodes per layer | [30, 30, 30, 30, 1] |  

To perform the training execute:  
```
python train -c configs/config.yaml
```

## Evaluation

Once the training is done, your training can be evaluated. The following parameters are all given below the "evaluation" parameter.  

| field | description | default value |  
|---|---|---|  
| model_file_numbers | list of the epoch number you would like to evaluate. | [1] |  
| ntracks | number of tracks to be considered. | 40 |  
| file_formats | list of file formats you want the plots to be saved | ["pdf", "png"] |  
| plot_efficiency | plot the efficiencies per epoch for the track-to-hadron assignment | {plot: False, recalculate: True} |  
| plot_effs_per_pt | plot the efficiencies for the track-to-hadron assignment as a function of pT| {plot: False}  |  
| plot_pt | plot the regressed variable as a function of the truth label | {plot: True} |  
| plot_conf_matrix | plot the confusion matrix for the track assignment efficiencies. | {plot: False} |  
| plot_preds_per_epoch | plot the track assignment prediction for each epoch specified by model_file_numbers. | {plot: False} |  
| plot_preds_scatter | plot the track assignment prediction as a function of the epoch | {plot: False} |  
| plot_saliency | plot the saliency map | {plot: False} |  
| plot_saliency_pertrack | plot the saliency map per track | {plot: False} |
| plot_saliency_pervar | plot the saliency map per variable| {plot: False} |
| plot_n_tracks_per_jet | plot the number of tracks per jet | {plot: False} |  
| plot_target_input_corr | plot correlation between the inputs and the regression variable | {plot: False} |  
| plot_linear_fit | plotting the distance of the vertex regession prediction to the diagonal | {plot: False} |  
| plot_track_origin | plot the truth direction of the tracks | {plot: False} |  
| plot_inputs | plot the inputs| {plot: False} |  
| plot_non_jet_tracks | plot the predicition of the track assignments for the tracks that do not originate from the corresponding hadron.| {plot: False} |  
| plot_vertex_labels | plot the truth regression labels | {plot: False} |  

To perform the evaluation you first need to collect the data per epoch. To do so execute:  
```
python evaluate -c configs/config.yaml --epoch 0
```
After having performed this for every epoch execute
```
python evaluate -c configs/config.yaml 
```
If you only want to make the plots that use the model of the epochs specified in `model_file_numbers` you only have to run the first command for these epochs. Otherwise you have to run it for all.
