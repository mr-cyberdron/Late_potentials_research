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
            complex_averaged = list(self.saecg_complex(signal, fs, r_peaks=averaged_r_peaks_list))
            saecg_mass.append(complex_averaged)
        self.SAECG = saecg_mass
        return self

    @staticmethod
    def saecg_complex(signal: np.ndarray[float], fs: [int], r_peaks: [int] = None,
                      filt_50_hz=True) -> np.ndarray[float]:

        signal_filtered = EcgRecord.saecg_filter(signal, fs)
        if filt_50_hz:
            signal_filtered = EcgRecord.stnphaze_50_hz_filter(signal_filtered, fs)
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
        signal_filtered = AnalogFilterDesign(signal_filtered, fs).lp(order=5, cutoff=250).zerophaze().butter() \
            .filtration()
        return signal_filtered

    @staticmethod
    def stnphaze_50_hz_filter(signal, fs):
        signal_filtered = AnalogFilterDesign(signal, fs).notch(quality_factor=80, cutoff=50).zerophaze() \
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
