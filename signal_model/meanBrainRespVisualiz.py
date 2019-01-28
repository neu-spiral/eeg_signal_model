import numpy as np
import matplotlib.pyplot as plt

CRED = '\033[31m'
CEND = '\033[0m'


def mean_resp_visualization(data, fs, paradigm, mode, path_save=[], id_user=[]):
    """
        Plot the average brain responses under two experiment paradigm in
        RSVPKeyboard tasks including ERP and FRP
        Input Args:
             data: a dictionary including
                   timeseries - eeg signals, 3D matrix,
                        num_channel x num_sample x num_seq
                   stimOnset - trigger information for all trials, 2D matrix,
                        num_trial x num_seq
                   targetOnset - trigger information for target events, vector,
                        1 x num_seq
             fs: sampling frequency (Hz), scaler real value, double
             pardigm: the experiment paradigm, either 'FRP' or 'ERP'
             mode: the signal model mode,
                which is either 'modelfitting' or 'simulator'
             path_save: the directory of saving figures
             id_user: the user's ID (is already included in the file's name)
    """
    # Initialization
    eeg = data["timeseries"]
    onset_trig = data["stimOnset"]
    onset_target = data["targetOnset"][0]
    ds = 100  # time shift that sets the visualization interval
    dt = fs - ds - 1
    time = np.arange(0, dt + 1., 1) / fs
    num_seq = eeg.shape[2]
    num_trial = onset_trig.shape[0]
    num_channel = eeg.shape[0]
    fig = plt.figure()
    ax1 = fig.add_subplot(211)
    ax2 = fig.add_subplot(212, sharex=ax1, sharey=ax1)
    fig.subplots_adjust(hspace=.5)
    dy = .5  # setting ylim

    if paradigm == "FRP":
        frp_neg = np.zeros((num_channel, dt + 1, num_seq))
        frp_pos = np.zeros((num_channel, dt + 1, num_seq))
        i_neg = 0
        i_pos = 0
        # Extracting all traget-event related activities
        for s in range(num_seq):
            if onset_target[s] > 0:
                frp_neg[:, :, i_neg] = eeg[:, onset_trig[0, s] -
                                              1:onset_trig[0, s] + dt, s]
                i_neg += 1
            else:
                frp_pos[:, :, i_pos] = eeg[:, onset_trig[0, s] -
                                              1:onset_trig[0, s] + dt, s]
                i_pos += 1
        # Compute the average responses for negative and positive FRPs
        mean_frp_n = np.mean(frp_neg[:, :, :i_neg], 2)
        mean_frp_p = np.mean(frp_pos[:, :, :i_pos], 2)
        for ch in range(num_channel):
            ax1.plot(time, mean_frp_n[ch, :])
            ax2.plot(time, mean_frp_p[ch, :])
        ax1.set_title('Average -FRPs')
        ax1.set_ylabel(r'EEG ($\mu$V)')
        ax1.set_xlim([0, .6])
        ax1.set_ylim([np.min([np.min(mean_frp_n), np.min(mean_frp_p)]) - dy,
                      np.max([np.max(mean_frp_n), np.max(mean_frp_p)]) + dy])
        ax2.set_title('Average +FRPs')
        ax2.set_xlabel('Time (Sec)')
        ax2.set_ylabel(r'EEG ($\mu$V)')
    else:
        erp = np.zeros((num_channel, dt + 1, num_seq))
        nonerp = np.zeros((num_channel, dt + 1, num_seq * num_trial))
        m = 0
        # Extracting all traget-event related activities
        for s in range(num_seq):
            erp[:, :, s] = eeg[:, onset_target[s] - 1:onset_target[s] + dt, s]
        m = 0
        # Extracting all nontraget-event related activities
        for s in range(num_seq):
            for t in range(num_trial):
                if onset_trig[t, s] == onset_target[s]:
                    continue
                else:
                    nonerp[:, :, m] = eeg[:, onset_trig[t, s] -
                                             1:onset_trig[t, s] + dt, s]
                    m += 1
        # Compute the average responses
        mean_erp_p = np.mean(erp, 2)
        mean_erp_n = np.mean(nonerp[:, :, :m], 2)
        for ch in range(num_channel):
            ax1.plot(time, mean_erp_p[ch, :])
            ax2.plot(time, mean_erp_n[ch, :])
        ax1.set_title('Average ERPs')
        ax1.set_ylabel(r'EEG ($\mu$V)')
        ax1.set_xlim([0, .6])
        ax1.set_ylim([np.min([np.min(mean_erp_p), np.min(mean_erp_n)]) - dy,
                      np.max([np.max(mean_erp_p), np.max(mean_erp_n)]) + dy])
        ax2.set_title('Average non-ERPs')
        ax2.set_xlabel('Time (Sec)')
        ax2.set_ylabel(r'EEG ($\mu$V)')
    fig.show()
    # Save generated figures
    if path_save:
        try:
            fig.savefig(path_save + id_user + mode + '.pdf', format='pdf',
                        dpi=1000, bbox_inches='tight')
        except:
            # TODO: this is not the only possible error
            print CRED + 'Can not save the file! The file', \
                id_user + mode + '.pdf might be open.' + CEND
