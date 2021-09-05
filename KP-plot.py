# diadaptasi dri JSD-plot_v2.3  (MZN 2019) download, analysis, and plot data di file terpisah

import matplotlib.pyplot as plt
import os
import pandas as pd
import numpy as np

from sunpy.timeseries import TimeSeries
from sunpy.time import TimeRange, parse_time
from sunpy.net import hek, Fido, attrs as a

wdir = r'C:\Users\Stargazers\PycharmProjects\magnetogram'
flare = pd.read_excel('KP-data_lengkap.xlsx')
ar = 12017
flare = 'C'
time = '2014.03.27'
time2 = '2014.03.29'
t_init=0 #start time (in hour UT t_init:00:00)
t_term=24 #end time (in hour UT [t_term:00:00-01:00:00])
t_step=1 #time step (hour +01:00:00 increment) 
time_str = (pd.to_datetime(time)).strftime('%Y-%m-%d')
time_str2 = (pd.to_datetime(time2)).strftime('%Y-%m-%d')
num = 2 # untuk nama file output, hanya ada 1 dan 2


def path_folder(chapter,waktu):
    nama_folder = str(ar)+'/'+chapter+'_'+str(waktu)+'/'
    os.makedirs(nama_folder, exist_ok=True)
    return(nama_folder)

tr = TimeRange([ time_str+' 00:00', time_str2+' 23:00'])
client = hek.HEKClient()
flares_hek = client.search(hek.attrs.Time(tr.start, tr.end),
                           hek.attrs.FL, hek.attrs.FRM.Name == 'SWPC')
n_sol_event = flares_hek.columns[0].size
'''
for z in range(flares_hek.columns[0].size): # z = nomor event
    #if flares_hek[z].get('fl_goescls')[0] == flare and flares_hek[z].get('ar_noaanum') == ar:
    if flares_hek[z].get('fl_goescls')[0] == flare:
        sol_event = flares_hek[z].get('SOL_standard')   
        peak_flare = flares_hek[z].get('event_peaktime')
        start_flare = flares_hek[z].get('event_starttime')
        end_flare = parse_time(flares_hek[z].get('event_endtime'))
        kelas_flare = flares_hek[z].get('fl_goescls')
        aregion = flares_hek[z].get('ar_noaanum')
        print('===============================')
        print('nomor event :',z, 'dari',n_sol_event-1)
        print('IAU standard nomor :',sol_event)
        print('ar :',aregion)
        print('kelas flare :',kelas_flare)
        print('mulai flare :',start_flare)
        print('puncak flare :',peak_flare)
        print('akhir flare :',end_flare)

kolom = ['time','sumexc_erg','sumerg_loc','sumJz_tot','sumJz_loc','umJzBloc','ratc']    
out_data = pd.read_csv(path_folder('output',time[8:10])+'OUTPUT_'+str(num)+'.txt', sep = ' ', names=kolom, index_col='time')
out_data['hour'] = np.arange(t_init,t_term,t_step)
def plot_out(data,k):
    fig = plt.figure(k)
    plt.plot(out_data['hour'],out_data[data],'.-')
    plt.xticks(np.arange(t_init,t_term,3))
    plt.title(data+'\n'+time_str)
    plt.xlabel('Time (hour in UT)')
    plot_folder = path_folder('plot_output',time[8:10])
    plt.grid(True)
    os.makedirs(plot_folder, exist_ok=True)
    path = os.path.join(plot_folder,kolom[i]+'_'+str(num)+'.png')
    fig.savefig(path,dpi=600)
    plt.close(fig)
for i in np.arange(1,7,1): plot_out(kolom[i],i)   
'''
    