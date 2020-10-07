# Main files

- dvrkTraining/dvrkSVRTraining.m - train the SVR models on the datasets
- optimization/optimalPoseCombinedGlobalSearchNormalized.m - generate the optimal pose on learned costmaps

- dVRKData/cone: contains the samples, models and optimized outputs from the optimization. 
  - samples/ <workspace_config> - contains the samples produced from simulator for learning the costmaps
  - saved_model/ - the learned regression models, distinguished by the DD_MM_YYYY_HH_numSamples_method.mat
  - pose/ - the optimized pose,  distinguished by DD_MM_YYYY_HH_numSamples_method 
  - hand_picked/ - hand-picked poses for illustrating the three metrics on the base positions;
  - optimized_poses/ -  the optimized poses, corresponding to base position D-G in Table II in the submitted paper to ICRA-RAL



# Model training and optimization

- Generate the samples with their three metrics in repo [fastron_experimental][https://github.com/ucsdarclab/fastron_experimental]; 

- Change the workspace for storing samples and corresponding parameters in dvrkTraining/dvrkSVRTraining.m; run the script, save the SVR models in dVRKData/cone/saved_model/ with the appropriate names (the datetime the model is trained);

- Change the model date-time in optimization/optimalPoseCombinedGlobalSearchNormalized.m, run the script and the optimized poses will be in dVRKData/cone/pose

- Use the evaluation benchmark in repo [fastron_experimental][https://github.com/ucsdarclab/fastron_experimental] to evaluate the poses. 

  

# File Directory

- artificialData: contains some datasets that were used for initial prototyping and testing;
- artificialDataTraining: matlab scripts for training and testing on synthetic data;
- datasetGeneration: files for generating 2D environments, including some GJK scripts;
- dVRKData: contains trained models, hand-picked and optimized base positions;
- figures: figures that were used in the paper;
- models: models that were used for training on the data;
- optimization: scripts used for optimizing on the costmap of the scores;
- plots: utility tools for plotting;
- results: tables for results; 
- scaling: platt scaling for calibrating the discriminative models;
- utility: some utility functions for dataset manipulations.



# Optimized Poses

```
Equal weights            							|  Boosting reachability               |  Boosting self-collision-free    | Boosting environment-collision-free
:-------------------------:							|:-------------------------:           |:-------------------------:       |:-------------------------:
![Optimized Poses with equal weights][https://github.com/jamesdi1993/fastron-confidence-score/tree/master/figures/final_poses/optimized_1_1_1.png]  |  ![Optimized Poses boosting reachability][https://github.com/jamesdi1993/fastron-confidence-score/tree/master/figures/final_poses/optimized_5_1_1.png] | ![Optimized Poses boosting self-collision-free][https://github.com/jamesdi1993/fastron-confidence-score/tree/master/figures/final_poses/optimized_1_5_1.png] | ![Optimized Poses boosting environment-collision-free][https://github.com/jamesdi1993/fastron-confidence-score/tree/master/figures/final_poses/optimized_1_1_5.png]
```





