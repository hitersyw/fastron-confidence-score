# matlab -nodisplay -nodesktop -r "generatePlots.m"
export PATH=/Applications/MATLAB_R2018b.app/bin/:$PATH
export PATH=/Users/jamesdi/Programs/anaconda3/bin/:$PATH
matlab -nodesktop -r "try, run('/Users/jamesdi/Dropbox/UCSD/Research/ARCLab/Code/ConfidenceScore/generatePlots.m'), catch, exit(1), end, exit(0);"
conda activate ConfidenceScore
python trainModels.py ./data/testset 50 1.0
conda deactivate
matlab -nodesktop -r "try, run('/Users/jamesdi/Dropbox/UCSD/Research/ARCLab/Code/ConfidenceScore/computeStats.m'), catch, exit(1), end"