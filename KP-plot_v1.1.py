# v2.3  (MZN 2019) download, analysis, and plot data di file terpisah

import matplotlib.pyplot as plt
import os
import pandas as pd
from matplotlib import dates

import sunpy
from sunpy.time import TimeRange
import datetime
from matplotlib.ticker import AutoMinorLocator

aregion = 12017
flare_class = 'C1'
time = '2014.03.27'
time2 = '2014.03.29'
t_init=0 #start time (in hour UT t_init:00:00)
t_term=24 #end time (in hour UT [t_term:00:00-01:00:00])
t_step=1 #time step (hour +01:00:00 increment) 
time_str = (pd.to_datetime(time)).strftime('%Y-%m-%d')
time_str2 = (pd.to_datetime(time2)).strftime('%Y-%m-%d')
num = 2 # untuk nama file output, hanya ada 1 dan 2


def path_folder(chapter,waktu):
    nama_folder = str(aregion)+'/'+chapter+'_'+str(waktu)+'/'
    os.makedirs(nama_folder, exist_ok=True)
    return(nama_folder)

def plot(parameter):
    fig,ax=plt.subplots()
    #prop_cycle = plt.rcParams['axes.prop_cycle']
    #colors = prop_cycle.by_key()['color']
    ax.plot(out_data.index,out_data[parameter],'.-',label='NOAA '+str(aregion))
    ax.xaxis.set_major_locator(dates.AutoDateLocator())
    ax.xaxis.set_major_formatter(dates.DateFormatter('%H\n%b %d'))
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.set_xlabel('Time (UT)')
    if parameter == 'free_E': 
        ax.set_ylabel(r'Free Energy (Ergs $cm^{-3}$)')
    elif parameter == 'TOTUSJH': 
        ax.set_ylabel('Total Current Helicity ($G^{2} m^{-1}$)')
    elif parameter == 'TOTUSJZ': 
        ax.set_ylabel(r'Vertical Current(Amperes)')
    else:
        ax.set_ylabel(r'|DC|/|RC|')
    for i in range(n_flare):
        ar = flare_results[i]['noaa_active_region']
        if ar == 12017:
            peak_time = pd.to_datetime(flare_results[i]['peak_time'].value)
            #start_time = pd.to_datetime(flare_results[i]['start_time'].value)
            #end_time = pd.to_datetime(flare_results[i]['end_time'].value)
            goes_class = flare_results[i]['goes_class']
            if goes_class[0] == 'C':
                ax.axvline(peak_time,color='orange',alpha=0.7,label = goes_class)
                #ax.axvspan(start_time,end_time,color='orange',alpha=0.7,label = goes_class)
            elif goes_class[0] == 'M':
                ax.axvline(peak_time,color='red',alpha=0.7,label = goes_class)
                #ax.axvspan(start_time,end_time,color='red',alpha=0.7,label = goes_class)
            else:
                ax.axvline(peak_time,color='maroon',alpha=0.7,label = goes_class)
                #ax.axvspan(start_time,end_time,color='maroon',alpha=0.7,label = goes_class)
        else: print(i,'Bukan AR 12017 tapi AR', ar)
    ax.legend(loc='best', numpoints=1,framealpha=0.5,fontsize=5)
    #plt.grid(True)
    print("Saving figure", parameter, "of",ar)
    fig.tight_layout()
    fig.savefig(str(aregion)+'/plot_output_all/'+str(aregion)+'_'+parameter+'_WIL'+str(num), dpi=600)
    plt.clf()
    plt.close(fig)

tr = TimeRange([ time_str+' 00:00', time_str2+' 23:00'])
flare_results =sunpy.instr.goes.get_goes_event_list(tr, goes_class_filter=flare_class)
n_flare = len(flare_results)

kolom = ['time','free_E','free_E_loc','TOTUSJZ','USFLUX','TOTUSJH','NC_ratio']   
data_27 = pd.read_csv(path_folder('output',27)+'OUTPUT_'+str(num)+'.txt', sep = ' ', names=kolom)
data_28 = pd.read_csv(path_folder('output',28)+'OUTPUT_'+str(num)+'.txt', sep = ' ', names=kolom)
data_29 = pd.read_csv(path_folder('output',29)+'OUTPUT_'+str(num)+'.txt', sep = ' ', names=kolom)
all_data = [data_27,data_28,data_29]
out_data = pd.concat(all_data,ignore_index=True)

time_new = []
for i in range(out_data.shape[0]):
    time_new.append(datetime.datetime.strptime(out_data.time[i],'%y/%m/%d %H:%M:%S'))
out_data['time_new'] = time_new   
out_data.index = out_data.time_new

    