# eeg_signal_model
This repository belongs to the generative EEG signal modeling project in Cognitive Systems Laboratory (CSL), at Northeastern University.

Electroencephalography (EEG) is an effective noninvasive measurement method to infer user intent in brain-computer interface (BCI) systems for control and communication, but so far these lack sufficient accuracy and speed due to low separability of class-conditional EEG feature distributions. Many factors impact system performance, including inadequate training datasets and models’ ignorance of the temporal dependency of brain responses to serial stimuli. 

Here, we propose a generative signal model for event related responses in EEG evoked with a rapid sequence of stimuli in BCI applications. The model describes EEG evoked by a sequence of stimuli as a superposition of impulse responses time-locked to stimuli corrupted with an autoregressive noise process. This model explicitly attempts to capture the temporal dependency of EEG signals evoked with rapid sequential stimulation. 

The performance of the signal model is assessed in the context of RSVP Keyboard, a language-model-assisted EEG-based BCI for typing [1]. The code can be used in three modes:

- modelfitting: to fix an ARX model to EEG sequences [2];
- simulator: to generate synthetic data according to the ARX model learned in the 'modelfitting' mode;
- visualization: to visualize the average brain responses over EEG trials.

There are two predefined experimental paradigms including Event-Related Potentials (ERPs) and Feedback-related Potentials (FRPs). More details about the signal model and experimental paradigms can be found in [3].





#### References

[1] Orhan, U, et al. **"RSVP keyboard: An EEG based typing interface."** *Proceedings of the... IEEE International Conference on Acoustics, Speech, and Signal Processing/sponsored by the Institute of Electrical and Electronics Engineers Signal Processing Society (ICASSP)*, 2012.

[2] Marghi YM, Gonzalez-Navarro P, et al. **"A Parametric EEG Signal Model for BCIs with Rapid-Trial Sequences."** *IEEE 40th Annual International Conference of the Engineering in Medicine and Biology Society (EMBC)*. IEEE, 2018. 

[3] Marghi YM, Gonzalez-Navarro P, et al. **"An Event-Driven AR-Process Model With Rapid Trial Sequences for EEG-based BCIs."** *IEEE Transactions on Neural Systems and Rehabilitation Engineering*, 2019. 

