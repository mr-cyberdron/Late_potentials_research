from dataclasses import dataclass
import matplotlib.pyplot as plt
import numpy as np
from similarity_methods.signal_similarity import cosine_similarity
from Frequency_tools.Filtering.AnalogFilters import AnalogFilterDesign
import neurokit2 as nk
from Withdrawal_tools import ColorPrint


@dataclass
class EcgRecord:
    Fs: list[int]
    Signals: np.array([float])
    Leads: list[str]
    Units: list[str]
    Metadata: dict[str:str]
    SAECG: list[np.array([float])] = None

    def saecg_count(self, all_leads_peaks_average=True):
        saecg_mass = []
        averaged_r_peaks_list = None
        if all_leads_peaks_average:
            averaged_r_peaks_list = self.average_peaks_by_all_leads(self.Signals, self.Fs[0])
        for signal, fs in zip(self.Signals, self.Fs):
            # import Frequency_tools.fft.FFT_tools
            # Frequency_tools.fft.FFT_tools.timefft_analisys(signal, fs, plotflag=True, log=True) #bp_filter=[30,80])
            complex_averaged = self.saecg_complex(signal, fs, r_peaks=averaged_r_peaks_list,
                                                  filt_50_hz=False, filt_100_hz=False, filt_60_hz=False,
                                                  filt_120_hz=False)
            if np.isnan(complex_averaged).all():
                complex_averaged = [0, 0, 0]
            else:
                complex_averaged = list(complex_averaged)

            saecg_mass.append(complex_averaged)
        self.SAECG = saecg_mass
        return self

    def ged_record_len(self):
        len_dict = {'Unit': 'sec'}
        for lead_name, signal, fs in zip(self.Leads, self.Signals, self.Fs):
            len_dict[lead_name] = len(signal) / fs
        return len_dict

    def remove_leads(self, *lead_to_remove: [str]):
        new_fs_mass = []
        new_signals_mass = []
        new_leads_mass = []
        new_units_mass = []
        for fs, signal, lead, unit in zip(self.Fs, self.Signals, self.Leads, self.Units):
            if lead in lead_to_remove:
                pass
            else:
                new_fs_mass.append(fs)
                new_signals_mass.append(signal)
                new_leads_mass.append(lead)
                new_units_mass.append(unit)

        self.Fs = new_fs_mass
        self.Signals = new_signals_mass
        self.Leads = new_leads_mass
        self.Units = new_units_mass
        return self

    @staticmethod
    def saecg_complex(signal: np.ndarray[float], fs: [int], r_peaks: [int] = None,
                      filt_50_hz=True, filt_100_hz=True, filt_60_hz=True, filt_120_hz=True) -> np.ndarray[float]:

        signal_filtered = EcgRecord.saecg_filter(signal, fs)
        if filt_50_hz:
            signal_filtered = EcgRecord.stnphaze_50_hz_filter(signal_filtered, fs)
        if filt_60_hz:
            signal_filtered = EcgRecord.stnphaze_60_hz_filter(signal_filtered, fs)
        if filt_120_hz:
            signal_filtered = EcgRecord.stnphaze_120_hz_filter(signal_filtered, fs)
        if filt_100_hz:
            signal_filtered = EcgRecord.stnphaze_100_hz_filter(signal_filtered, fs)
        if list(r_peaks):
            qrs_epochs = nk.ecg_segment(signal_filtered, sampling_rate=fs, rpeaks=r_peaks, show=False)
            plt.show()
        else:
            qrs_epochs = nk.ecg_segment(signal_filtered, sampling_rate=fs,
                                        rpeaks=EcgRecord.find_r_peaks_mass(signal_filtered), show=False)
        averaged_complex = EcgRecord.average_qrs_epochs(qrs_epochs, remove_outliers=True)

        return averaged_complex

    @staticmethod
    def remove_strange_cardiocycles(nk_qrs_epochs):
        filtered_mass = []
        tmp_averaged_complex = EcgRecord.average_qrs_epochs(nk_qrs_epochs)
        epochs_mass = EcgRecord.qrs_epochs_list(nk_qrs_epochs)
        for qrs_complex in epochs_mass:
            similarity = cosine_similarity([qrs_complex], tmp_averaged_complex)
            if similarity[0] > 0.9:
                filtered_mass.append(qrs_complex)
            else:
                ColorPrint('CardioCycle outlier removed (' + str(round(similarity[0] * 100, 2)) + ' %)').red()
                # plt.figure()
                # plt.plot(tmp_averaged_complex, color='orange')
                # plt.plot(qrs_complex, color='yellow')
                # plt.show()
        # cmap = iter(plt.cm.YlOrRd(np.linspace(0, 1, num=len(filtered_mass))))
        # for complex, color in zip(filtered_mass, cmap):
        #     plt.plot(complex, color=color)
        # plt.show()
        return filtered_mass

    @staticmethod
    def saecg_filter(signal, fs):
        signal_filtered = AnalogFilterDesign(signal, fs).hp(order=5, cutoff=1).zerophaze().butter() \
            .filtration()
        signal_filtered = AnalogFilterDesign(signal_filtered, fs).lp(order=5, cutoff=240).zerophaze().butter() \
            .filtration()
        return signal_filtered

    @staticmethod
    def stnphaze_50_hz_filter(signal, fs):
        signal_filtered = AnalogFilterDesign(signal, fs).notch(quality_factor=150, cutoff=50).zerophaze() \
            .filtration(show=False)
        return signal_filtered

    @staticmethod
    def stnphaze_100_hz_filter(signal, fs):
        signal_filtered = AnalogFilterDesign(signal, fs).notch(quality_factor=150, cutoff=100).zerophaze() \
            .filtration(show=False)
        return signal_filtered

    @staticmethod
    def stnphaze_60_hz_filter(signal, fs):
        signal_filtered = AnalogFilterDesign(signal, fs).notch(quality_factor=150, cutoff=60).zerophaze() \
            .filtration(show=False)
        return signal_filtered

    @staticmethod
    def stnphaze_120_hz_filter(signal, fs):
        signal_filtered = AnalogFilterDesign(signal, fs).notch(quality_factor=150, cutoff=120).zerophaze() \
            .filtration(show=False)
        return signal_filtered

    @staticmethod
    def qrs_epochs_list(epochs):
        qrs_nums = epochs.keys()
        complexes_mass = []
        for qrs_num in qrs_nums:
            signal = list(epochs[qrs_num]['Signal'].to_numpy())
            complexes_mass.append(signal)
        complexes_mass = np.array(complexes_mass)
        return complexes_mass

    @staticmethod
    def average_qrs_epochs(nk_epochs, remove_outliers=False):
        if remove_outliers:
            qrs_complexes = EcgRecord.remove_strange_cardiocycles(nk_epochs)
        else:
            qrs_complexes = EcgRecord.qrs_epochs_list(nk_epochs)
        qrs_complex_averaged = np.array(qrs_complexes).mean(axis=0)
        return qrs_complex_averaged

    @staticmethod
    def find_r_peaks_mass(ecg_signal):
        cleaned = nk.ecg_clean(ecg_signal, method="neurokit")
        _, peaks = nk.ecg_peaks(cleaned, method="kalidas2017")
        return peaks["ECG_R_Peaks"]

    @staticmethod
    def average_peaks_by_all_leads(signals, fs):
        r_peaks_for_each_lead = []
        for lead_signal in signals:
            r_peaks_for_each_lead.append(EcgRecord.find_r_peaks_mass(lead_signal))
        max_peak_value = np.max(np.concatenate(r_peaks_for_each_lead, axis=0))
        positions_vector = list(range(0, max_peak_value + int(np.round(fs / 2)), 1))
        all_positions_dict = {}
        for pos in positions_vector:
            all_positions_dict[pos] = 0
        for r_peaks_mass_lead in r_peaks_for_each_lead:
            for position in positions_vector:
                if position in r_peaks_mass_lead:
                    all_positions_dict[position] += 1
        results_filtered = AnalogFilterDesign(list(all_positions_dict.values()),
                                              fs).lp(order=5, cutoff=8).zerophaze().butter().filtration()
        _, averaged_r_peaks_responce = nk.ecg_peaks(results_filtered, method="neurokit")
        averaged_r_peaks_responce = averaged_r_peaks_responce["ECG_R_Peaks"]
        return averaged_r_peaks_responce
