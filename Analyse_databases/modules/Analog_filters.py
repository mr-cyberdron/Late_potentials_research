from scipy import signal
import numpy as np


class AnalogFilterDesign:
    # ----------In develop!--------------------#
    """
    Creation universal filter
    Example:
        filtered_signal = AnalogFilterDesign(sig,1000).lp(order=5, cutoff=25).bessel().zerophaze().filtration()
        filtered_signal2 = AnalogFilterDesign(sig,1000).bs(order=3, cutoff=[35, 43]).butter().zerophaze().filtration()
    """

    def __init__(self, input_signal, fs):
        self.Signal = input_signal
        self.Fs = fs
        self.Filer = None
        self.Type = None
        self.Order = None
        self.Cutoff = None
        self.Coefs = None
        self.Filer = signal.lfilter
        self.SignalFiltered = None

    def butter(self):
        b, a = signal.butter(self.Order,
                             self.Cutoff,
                             fs=self.Fs,
                             btype=self.Type,
                             analog=False)
        self.Coefs = {'b': b, 'a': a}
        return self

    def bessel(self):
        b, a = signal.bessel(self.Order,
                             self.Cutoff,
                             fs=self.Fs,
                             btype=self.Type,
                             analog=False)
        b = b * 300
        a = a * 300
        self.Coefs = {'b': b, 'a': a}
        return self

    def lp(self, order=None, cutoff=None):
        self.Order = order
        self.Cutoff = cutoff
        self.Type = 'lowpass'
        self.butter()
        return self

    def hp(self, order=None, cutoff=None):
        self.Order = order
        self.Cutoff = cutoff
        self.Type = 'highpass'
        self.butter()
        return self

    def bp(self, order=None, cutoff=None):
        self.Order = order
        self.Cutoff = cutoff
        self.Type = 'bandpass'
        self.butter()
        return self

    def bs(self, order=None, cutoff=None):
        self.Order = order
        self.Cutoff = cutoff
        self.Type = 'bandstop'
        self.butter()
        return self

    def zerophaze(self):
        self.Filer = signal.filtfilt
        self.Order = int(np.ceil(self.Order / 2))
        return self

    def filtration(self):
        b = self.Coefs['b']
        a = self.Coefs['a']
        self.SignalFiltered = self.Filer(b, a, self.Signal)
        return self.SignalFiltered
