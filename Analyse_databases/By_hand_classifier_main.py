import numpy as np
from Analyse_databases.modules.WFDB import WfdbParce
from class_record import EcgRecord
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import pylab
import os
import pandas as pd


def t_vectors(signals, fs) -> list[list[float]]:
    t_vectors_mass = []
    for signal, total_fs in zip(signals, fs):
        sig_units_num = len(signal)
        siglen_s = sig_units_num / total_fs
        t_vector = list(np.linspace(0, siglen_s, sig_units_num))
        t_vectors_mass.append(t_vector)
    return t_vectors_mass


def plot_ecg_record(signals, leads, units, fs_mass, title, filee, metadata) -> None:
    tvectors = t_vectors(signals, fs_mass)
    n_cols = 3
    expected_n_rows = int(np.ceil(len(signals) / n_cols))
    plt.rc('font', size=6)
    fig, axs = plt.subplots(nrows=expected_n_rows, ncols=n_cols, figsize=(14, 6))
    fig.canvas.manager.set_window_title(title)
    counter_rows = 0
    counter_cols = 0
    for signal, lead, unit, tvector in zip(signals, leads, units, tvectors):
        if counter_rows > (expected_n_rows - 1):
            counter_cols = counter_cols + 1
            counter_rows = 0
        axs[counter_rows, counter_cols].plot(tvector, signal, linewidth=0.8)  # , color= '#A40483')
        if lead == 'vx':
            axs[counter_rows, counter_cols].set_facecolor('#FFC0B5')
        if lead == 'vy':
            axs[counter_rows, counter_cols].set_facecolor('#FFF4B5')
        if lead == 'vz':
            axs[counter_rows, counter_cols].set_facecolor('#B9FFB5')
        axs[counter_rows, counter_cols].legend([str(lead) + '[' + str(unit) + ']'], loc='upper right')
        if counter_rows == (expected_n_rows - 1):
            axs[counter_rows, counter_cols].set_xlabel('[Sec]')

        counter_rows = counter_rows + 1

    plt.subplots_adjust(left=0.027, bottom=0.048, right=0.98, top=0.98, wspace=0.082, hspace=0.133)
    plt.show(block=False)

    fig2 = plt.figure(figsize=(7, 1))

    def tag_lvp(event):
        save_analyse_result('LVP')

    def tag_lap(event):
        save_analyse_result('LAP')

    def tag_lap_lvp(event):
        save_analyse_result('LVP_and_LAP')

    def tag_skip(event):
        save_analyse_result('none')

    def save_analyse_result(typee):
        dataline = {'file': [filee], 'type': [typee], 'metadata_json': [str(metadata)]}
        df_line = pd.DataFrame(data=dataline)
        old_df = pd.read_csv("logfile.csv")
        new_df = pd.concat([old_df, df_line])
        new_df.to_csv('logfile.csv', index=False)
        plt.close(fig)
        plt.close(fig2)

    button_LVP = Button(pylab.axes([0.005, 0, 0.25, 1]), 'Late ventricular potentials', color='#698EF1')
    button_LAP = Button(pylab.axes([0.26, 0, 0.25, 1]), 'Late atrial potentials', color='#DA69F1')
    button_LAP_and_LVP = Button(pylab.axes([0.515, 0, 0.25, 1]), 'Late atrial and ventricular potentials',
                                color='#EDF169')
    button_skip = Button(pylab.axes([0.77, 0, 0.25, 1]), 'Skip', color='#F16969')
    button_LVP.on_clicked(tag_lvp)
    button_LAP.on_clicked(tag_lap)
    button_LAP_and_LVP.on_clicked(tag_lap_lvp)
    button_skip.on_clicked(tag_skip)
    plt.show()


# ------------------------------------Input variables-------------------------------------------#
#ptb-diagnostic-ecg-database-1.0.0 https://physionet.org/content/ptbdb/1.0.0/
files_list_path = 'E:/Bases/PTB DATABASE/ptb-diagnostic-ecg-database-1.0.0/C/RECORDS'
file_path_header = 'E:/Bases/PTB DATABASE/ptb-diagnostic-ecg-database-1.0.0/ptb-diagnostic-ecg-database-1.0.0/'
# ----------------------------------------------------------------------------------------------#
with open(files_list_path, "r") as f:
    files_paths_parts = f.readlines()

if os.path.exists("logfile.csv") is False:
    print('not_exist')
    tmp_df = pd.DataFrame(data={'file': [], 'type': [], 'metadata_json': []})
    tmp_df.to_csv('logfile.csv', index=False)
else:
    print("logfile Exists")

files_list = list(pd.read_csv("logfile.csv")['file'].to_numpy())

file_counter = 1
for file in files_paths_parts:
    print(file)
    total_num = str(file_counter) + '/' + str(len(files_paths_parts))
    file_counter += 1
    print(total_num)
    if file not in files_list:
        full_record_path = file_path_header + file.replace('\n', '')
        wfdb_file_object = WfdbParce(full_record_path).read()
        record_object_data = EcgRecord(Fs=wfdb_file_object.Fs,
                                       Signals=wfdb_file_object.Signals,
                                       Leads=wfdb_file_object.Leads,
                                       Units=wfdb_file_object.Units,
                                       Metadata=wfdb_file_object.Metadata)
        record_object_data.saecg_count()
        print(record_object_data.Metadata)
        # signals_p = record_object_data.Signals
        signals_p = record_object_data.SAECG
        leads_p = record_object_data.Leads
        units_p = record_object_data.Units
        fs_mass_p = record_object_data.Fs
        img_tit = str(file_counter - 1) + ' ' + str(record_object_data.Metadata)
        plot_ecg_record(signals_p, leads_p, units_p, fs_mass_p, img_tit, file, record_object_data.Metadata)
    else:
        print('skipped')
