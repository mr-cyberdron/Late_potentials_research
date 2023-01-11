from dataclasses import dataclass
import numpy as np
from Analyse_databases.modules.Analog_filters import AnalogFilterDesign
import neurokit2 as nk



@dataclass
class EcgRecord:
    Fs: list[int]
    Signals: np.array([float])
    Leads: list[str]
    Units: list[str]
    Metadata: dict[str:str]
    SAECG: list[np.array([float])] = None

    def saecg_count(self):
        saecg_mass = []
        for signal, fs in zip(self.Signals, self.Fs):
            complex_averaged = list(self.saecg_complex(signal, fs))
            saecg_mass.append(complex_averaged)
        self.SAECG = saecg_mass
        return self

    @staticmethod
    def saecg_filter(signal, fs):
        signal_filtered = AnalogFilterDesign(signal, fs).hp(order=5, cutoff=0.25).butter().zerophaze() \
            .filtration()
        signal_filtered = AnalogFilterDesign(signal_filtered, fs).lp(order=5, cutoff=250).butter().zerophaze() \
            .filtration()
        return signal_filtered

    @staticmethod
    def saecg_complex(signal: np.ndarray[float], fs: [int]) -> np.ndarray[float]:
        try:
            signal_filtered = EcgRecord.saecg_filter(signal, fs)
            qrs_epochs = nk.ecg_segment(signal_filtered, sampling_rate=fs, show=False)
            qes_nums = qrs_epochs.keys()
            qrs_complexes = []
            for qrs_num in qes_nums:
                signal = list(qrs_epochs[qrs_num]['Signal'].to_numpy())
                qrs_complexes.append(signal)
            qrs_complexes = np.array(qrs_complexes)
            averaged_complex = qrs_complexes.mean(axis=0)
        except:
            averaged_complex = [0,0,0]
        return averaged_complex
