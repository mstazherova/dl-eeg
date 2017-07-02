import mne
import numpy as np
# import matplotlib
# matplotlib.use('TkAgg')
raw = mne.io.read_raw_eeglab("data.import/CR_S00_T_DreamReports_EEG/CR_S00_T_DR_810_820.mat")
raw1 = mne.io.read_raw_eeglab("data.import/CR_S35_SN_MentalRehearsal_EEG/CR_S35_SN_MR_758_780.mat")
# raw.plot(block=True)
raw1.plot(block=True, duration=20.0, n_channels=23, scalings='auto')
#raw1.plot_psd()
#raw1.plot_psd(tmax=np.inf, average=False)
