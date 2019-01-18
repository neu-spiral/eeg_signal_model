import os, sys, pickle
import scipy.io as sio
import numpy as np
from signal_model.ARXmodelFit import ARXmodelfit
from signal_model.meanBrainRespVisualiz import meanBrainRespVisualiz

CRED = '\033[31m'
CEND = '\033[0m'

# Set the simulator parameters
# Experiment paradigm including 'ERP' and 'FRP'
paradigm = "ERP"
saveFlag = True
# Define the program mode: 'simulator', 'modelfitting', 'visualization'
mode = "simulator"
fileNum = 0
hyperParameterLearning = True
# set directories including data and to save files
try:
    file_dir = "./data/"+paradigm+"s"       # data directory (data should be in MATLAB)
    fig_dir = "./figures/"+paradigm+"s/"    # path to save figures
    model_dir = "./model/"+paradigm+"s/"    # path to save model parameters and synthetic data
except:
    print CRED + 'Make sure you have all of the required folders and subfolders!' + CEND
try:
    # list of files' name in the data directory
    list_filename = os.listdir(file_dir)
    filename = list_filename[fileNum]
    tmp = sio.loadmat(file_dir + '/' + filename)
except:
    print CRED + 'Make sure data folder includes .mat data for the selected paradigm!' + CEND
eegCh = range(16)     # number of eeg channels
nFold = 5          # number of fold in cross validation
# load filename and initialize the user from MATLAB file
data = dict()
data['stimOnset'] = tmp['us']
data['targetOnset'] = tmp['ue']
data['timeseries'] = tmp['eeg_seq']
channels = tmp['EEGChannels'][0]
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
# Print the user id and AUC value saved in the file
print '\n', 'User:', userID, '\n','\n'
print 'Running in', mode, 'mode, under', paradigm, 'paradigm.', '\n'

# Run the EEG signal model based on the predefined mode
if mode == "simulator":
    with open(model_dir+userID+'_modelParam.p', "rb") as f:
        model_dic = pickle.load(f)
    f.close()
    try:
        parameters = model_dic["parameters"]
        hyperparameter = model_dic["hyperparameters"]
    except:
        print CRED + 'You need to learn the ARX model parameters and hyperparameters!' + CEND
        print CRED + 'First, run on \'modelfitting\' mode.' + CEND
        sys.exit()
    stimOnset = tmp['us']
    targetOnset = tmp['ue']
    modelObj = ARXmodelfit(fs=fs, paradigm=paradigm, numTrial=numTrial,
                            numSeq=numSeq, numSamp=numSam, channels=channels,
                            hyperparameter=hyperparameter, orderSelection=False)
    syn_data = modelObj.syntheticEEGseq(parameters, stimOnset, targetOnset)
    meanBrainRespVisualiz(syn_data, fs, paradigm, mode, fig_dir, userID)
    if saveFlag:
        with open(model_dir+userID+'_syntheticData.p', "wb") as f:
            pickle.dump(syn_data, f)
        f.close()

elif mode == "modelfitting":
    # timeseries of multi-channel EEG measurement (numChannel x numSample x numSequence)
    meanBrainRespVisualiz(data, fs, paradigm, mode, fig_dir, userID)
    # set a range of hyperparameters for grid search
    hyperparameter = tmp['hyperparameters']
    modelObj = ARXmodelfit(fs=fs, paradigm=paradigm, numTrial=numTrial,
                            numSeq=numSeq, numSamp=numSam, channels=channels,
                            hyperparameter=hyperparameter, orderSelection=True)
    auc, acc, parameters, hyperParam = modelObj.ARXmodelfit(data=data, nFold=nFold)
    print 'AUC:',np.mean(auc),u'\u00B1',np.std(auc)
    print 'ACC:', np.mean(acc),u'\u00B1',np.std(acc)
    if saveFlag:
        save_dic = {"parameters": parameters,
                    "hyperparameters": hyperParam,
                    "accuracy": auc,
                    "auc": acc}
        with open(model_dir+userID+'_modelParam.p', "wb") as f:
            pickle.dump(save_dic, f)
        f.close()

elif mode == "visualization":
    meanBrainRespVisualiz(data, fs, paradigm, mode)

else:
    print CRED + mode, 'is NOT defined!' + CEND
    sys.exit()





