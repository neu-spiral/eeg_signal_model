import os, sys
import scipy.io as sio
import numpy as np
import json, ast
import warnings
from copy import copy
from sklearn import metrics
import pyprog
from signal_model.ARXmodelFit import ARXmodelfit

CRED = '\033[31m'
CEND = '\033[0m'

# set directories including data and to save files
file_dir = "./data/FRPs"   # data directory (data should be in MATLAB)
fig_dir = "./figures/FRPs"    # path to save figures
model_dir = "./model/FRPs"    # path to save figures
list_filename = os.listdir(file_dir) # list of files' name in the data directory

# Set the simulator parameters
# Experiment paradigm including 'ERP' and 'FRP'
paradigm = "FRP"
multiChannel = True
# Define the program mode: 'simulator' or 'modelfitting'
mode = "modelfitting"
hyperParameterLearning = True
# Set the measurement channels
#channels = ['Fp1','Fp2','F3','F4','Fz','Fc1','Fc2','Cz','P1','P2','C1','C2',
           # 'Cp3','Cp4','P5','P6']
eegCh = range(2,17)     # number of eeg channels
nFold = 2          # number of fold in cross validation
# load filename and initialize the user from MATLAB file
filename = list_filename[0]
tmp = sio.loadmat(file_dir + '/' + filename)
channels = tmp['EEGChannels'][0]
data = dict()
data['timeseries'] = tmp['eeg_seq']
data['stimOnset'] = tmp['us']
data['targetOnset'] = tmp['ue']
hyperparameter = tmp['hyperparameters']   # range of hyperparameters for grid search
auc_user = tmp['auc'][0][0]
fs = tmp['fs'][0][0]
numTrial = tmp['numTrial'][0][0]
numSeq = tmp['numSeq'][0][0]
numSam = tmp['eeg_seq'].shape[1]

# print the user is and performance according to the trial-based (TB) model
indx = [i for i in range(len(filename)) if filename[i] == '_']
userID = filename[:indx[2]]

# Check te paradigm is correct or not
if paradigm != "FRP" and paradigm != "ERP":
    print CRED + "Please enter a valid paradigm e.g. FRP or ERP!" + CEND
    sys.exit()
# Print the user id and trial-based AUC value saved in the file
print '\n', 'User:', userID, '\n', 'TB_AUC:', auc_user, '\n', '\n', '\n'

# Run the EEG signal model based on the predfined mode
if mode == "simulator":
    simulator = ARXmodelfit(fs=fs, paradigm=paradigm, numTrial=numTrial,
                            numSeq=numSeq, numSamp=numSam,
                            hyperparameter=hyperparameter, channels=channels,
                            visualization=True)
elif mode == "modelfitting":
    modelObj = ARXmodelfit(fs=fs, paradigm=paradigm, numTrial=numTrial,
                            numSeq=numSeq, numSamp=numSam,
                            hyperparameter=hyperparameter, channels=channels,
                            orderSelection=True, visualization=True)
    auc, acc, parameters = modelObj.ARXmodelfit(data=data, nFold=nFold)
    ind = []
    print'AUC:',np.max(auc), '  ACC:', np.max(acc)
else:
    print CRED + mode, 'is NOT defined!' + CEND
    sys.exit()


