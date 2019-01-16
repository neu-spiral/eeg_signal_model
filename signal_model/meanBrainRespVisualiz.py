import numpy as np
import matplotlib.pyplot as plt

CRED = '\033[31m'
CEND = '\033[0m'

def meanBrainRespVisualiz(data, fs, paradigm, mode, savingPath=[], userID=[]):
    """
        Plot the average brain responses under two experiment paradigm in
        RSVPKeyboard tasks including ERP and FRP
        Input Args:
             data: a dictionary including
                   timeseries - eeg signals, 3D matrix, numberChannel x numberSample x numberSequence
                   stimOnset - trigger information for all trials, 2D matrix, numberTrial x numberSequence
                   targetOnset - trigger information for target events, vector, 1 x numberSequence
             fs: sampling frequency (Hz), scaler real value, double
             pardigm: the experiment paradigm, either 'FRP' or 'ERP'
             mode: the signal model mode, which is either 'modelfitting' or 'simulator'
             saveingPath: the directory of saving figures
             userID: the user's ID (is already included in the file's name)
    """
    # Initialization
    eeg = data["timeseries"]
    trigOnsets = data["stimOnset"]
    targetOnsets = data["targetOnset"][0]
    ds = 100   # time shift that sets the visualization interval
    dt = fs-ds-1
    time = np.arange(0, dt+1., 1)/fs
    numSeq = eeg.shape[2]
    numTrial = trigOnsets.shape[0]
    numChannel = eeg.shape[0]
    fig = plt.figure()
    ax1 = fig.add_subplot(211)
    ax2 = fig.add_subplot(212, sharex=ax1, sharey=ax1)
    fig.subplots_adjust(hspace=.5)
    dy = .5 # setting ylim

    if paradigm=="FRP":
        frp_neg = np.zeros((numChannel, dt+1, numSeq))
        frp_pos = np.zeros((numChannel, dt+1, numSeq))
        i_neg = 0
        i_pos = 0
        # Extracting all traget-event related activities
        for s in range(numSeq):
            if targetOnsets[s] > 0:
                frp_neg[:,:,i_neg] = eeg[:,trigOnsets[0,s]-1:trigOnsets[0,s]+dt,s]
                i_neg +=1
            else:
                frp_pos[:,:,i_pos] = eeg[:,trigOnsets[0,s]-1:trigOnsets[0,s]+dt,s]
                i_pos +=1
        # Compute the average responses
        meanNegFRPs = np.mean(frp_neg[:,:,:i_neg],2)
        meanPosFRPs = np.mean(frp_pos[:,:,:i_pos],2)
        for ch in range(numChannel):
            ax1.plot(time, meanNegFRPs[ch,:])
            ax2.plot(time, meanPosFRPs[ch,:])
        ax1.set_title('Average -FRPs')
        ax1.set_ylabel(r'EEG ($\mu$V)')
        ax1.set_xlim([0,.6])
        ax1.set_ylim([np.min([np.min(meanNegFRPs),np.min(meanPosFRPs)])-dy,
                      np.max([np.max(meanNegFRPs),np.max(meanPosFRPs)])+dy])
        ax2.set_title('Average +FRPs')
        ax2.set_xlabel('Time (Sec)')
        ax2.set_ylabel(r'EEG ($\mu$V)')
    else:
        erp = np.zeros((numChannel, dt+1, numSeq))
        nonerp = np.zeros((numChannel, dt+1, numSeq*numTrial))
        m = 0
        # Extracting all traget-event related activities
        for s in range(numSeq):
            erp[:,:,s] = eeg[:,targetOnsets[s]-1:targetOnsets[s]+dt,s]
        m = 0
        # Extracting all nontraget-event related activities
        for s in range(numSeq):
            for t in range(numTrial):
                if trigOnsets[t,s]==targetOnsets[s]:
                    continue
                else:
                    nonerp[:,:,m] = eeg[:,trigOnsets[t,s]-1:trigOnsets[t,s]+dt,s]
                    m +=1
        # Compute the average responses
        meanERPs = np.mean(erp,2)
        meannonERPs = np.mean(nonerp[:,:,:m],2)
        for ch in range(numChannel):
            ax1.plot(time, meanERPs[ch,:])
            ax2.plot(time, meannonERPs[ch,:])
        ax1.set_title('Average ERPs')
        ax1.set_ylabel(r'EEG ($\mu$V)')
        ax1.set_xlim([0,.6])
        ax1.set_ylim([np.min([np.min(meanERPs),np.min(meannonERPs)])-dy,
                      np.max([np.max(meanERPs),np.max(meannonERPs)])+dy])
        ax2.set_title('Average non-ERPs')
        ax2.set_xlabel('Time (Sec)')
        ax2.set_ylabel(r'EEG ($\mu$V)')
    fig.show()
    # Save generated figures
    if savingPath:
        try:
            fig.savefig(savingPath+userID+mode+'.pdf', format='pdf', dpi=1000,
                                                     bbox_inches='tight')
        except:
            print CRED+'Can not save the file! The file',userID+mode+'.pdf',' might be open.'+CEND









