# diadaptasi dari JSD-plot_v2.3.1  (MZN 2019) download, analysis, and plot data di file terpisah

import matplotlib.pyplot as plt
import os
import pandas as pd
from matplotlib import dates
import matplotlib
matplotlib.use('Agg')
from sunpy.net import hek
from datetime import datetime,timedelta
import matplotlib.ticker as ticker
import numpy as np

wdir = r'C:\Users\Stargazers\PycharmProjects\magnetogram'
flare = pd.read_excel('KP-data_lengkap.xlsx')
sharp_noaa = pd.read_csv('http://jsoc.stanford.edu/doc/data/hmi/harpnum_to_noaa/all_harps_with_noaa_ars.txt',sep=' ',index_col='HARPNUM')
kolom = ['datetime','free_E','free_E_loc','TOTUSJZ','TOTUSJZ_loc','TOTUSJH_loc','NC_ratio']
def ar_sharpnum(i):
    ar = int(flare.loc[i]['No.NOAA'])
    sharpnum = sharp_noaa[sharp_noaa['NOAA_ARS'].str.contains(str(ar))].index[0]
    return ar,sharpnum
aregion = ar_sharpnum(0)[0]
num = 2 # untuk nama file output, hanya ada 1 dan 2
data = pd.read_csv('{}/output_{}.csv'.format(str(aregion),num), names=kolom)
def date_time(x, y):
    date = datetime.strptime(data.loc[x]['datetime'], '%y/%m/%d %H:%M:%S')
    date_str = date.strftime('%Y.%m.%d_%H:%M:%S_TAI')
    time = date - timedelta(days=int(y))
    time_str = time.strftime('%Y.%m.%d_%H:%M:%S_TAI')
    waktu = time.strftime('%Y.%m.%d')
    return date_str, time, time_str, waktu
time = date_time(0,3)[3]
time2 = date_time(0,-3)[3]
# aregion = 12017
flare_class = 'X1.0'
# time = '2014.03.27'
# time2 = '2014.03.29'
# t_init=0 #start time (in hour UT t_init:00:00)
# t_term=24 #end time (in hour UT [t_term:00:00-01:00:00])
# t_step=1 #time step (hour +01:00:00 increment)
time_str = (pd.to_datetime(time)).strftime('%Y/%m/%d')
time_str2 = (pd.to_datetime(time2)).strftime('%Y/%m/%d')
waktu = [date_time(i,0)[1] for i in range(len(data))]
def path_folder(chapter,waktu):
    nama_folder = str(aregion)+'/'+chapter+'_'+str(waktu)+'/'
    os.makedirs(nama_folder, exist_ok=True)
    return(nama_folder)
def plot(parameter,data,x):
    fig,ax=plt.subplots()
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']
    ax.plot(waktu,data[parameter], 'bo',label='NOAA '+str(aregion))
    ax.xaxis.set_major_locator(dates.AutoDateLocator())
    # ax.xaxis.set_major_formatter(dates.DateFormatter('%H\n%b %d'))
    majors = ['-3','-2','-1','0','1','2','3']
    ax.xaxis.set_major_formatter(ticker.FixedFormatter(majors))
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.set_xlabel('Day (since {})'.format(flare.loc[x]['Tanggal & Waktu']))
    if parameter == 'free_E': 
        ax.set_ylabel(r'Free Energy (Ergs $cm^{-3}$)')
    elif parameter == 'TOTUSJH': 
        ax.set_ylabel('Total Current Helicity ($G^{2} m^{-1}$)')
    elif parameter == 'TOTUSJZ': 
        ax.set_ylabel(r'Vertical Current(Amperes)')
    else:
        ax.set_ylabel(r'|DC|/|RC|')
    # for i in range(n_flare):
    #     ar = flare_results[i]['ar_noaanum']
    #     if ar == aregion:
    #         peak_time = pd.to_datetime(flare_results[i]['event_peaktime'])
    #         #start_time = pd.to_datetime(flare_results[i]['event_starttime'].value)
    #         #end_time = pd.to_datetime(flare_results[i]['event_endtime'].value)
    #         goes_class = flare_results[i]['fl_goescls']
    #         if goes_class[0] == 'C':
    #             ax.axvline(peak_time,color='orange',alpha=0.7,label = goes_class)
    #             #ax.axvspan(start_time,end_time,color='orange',alpha=0.7,label = goes_class)
    #         elif goes_class[0] == 'M':
    #             ax.axvline(peak_time,color='red',alpha=0.7,label = goes_class)
    #             #ax.axvspan(start_time,end_time,color='red',alpha=0.7,label = goes_class)
    #         else:
    #             ax.axvline(peak_time,color='maroon',alpha=0.7,label = goes_class)
    #             #ax.axvspan(start_time,end_time,color='maroon',alpha=0.7,label = goes_class)
    #     else: print(i,'Bukan AR {} tapi {}'.format(str(aregion), ar))
    # ax.legend(loc='best', numpoints=1,framealpha=0.5,fontsize=5)
    #plt.grid(True)
    # print("Saving figure", parameter, "of",ar)
    fig.tight_layout()
    fig.savefig('{}/{}_{}'.format(str(aregion),parameter,str(num)))
    plt.close(fig)

client = hek.HEKClient()
event_type = 'FL'
flare_results = client.search(hek.attrs.Time(time_str, time_str2),
                          hek.attrs.EventType(event_type),
                          hek.attrs.FL.GOESCls > flare_class,
                          hek.attrs.OBS.Observatory == 'GOES')
n_flare = len(flare_results)
for i in range(n_flare):
    peak_time = pd.to_datetime(flare_results[i]['event_peaktime'])
    #start_time = pd.to_datetime(flare_results[i]['event_starttime'].value)
    #end_time = pd.to_datetime(flare_results[i]['event_endtime'].value)
    goes_class = flare_results[i]['fl_goescls']

# time_new = []
# for i in range(data.shape[0]):
#     time_new.append(datetime.strptime(data.time[i],'%y/%m/%d %H:%M:%S'))
# data['time_new'] = time_new
# data.index = data.time_new