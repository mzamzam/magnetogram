# diadaptasi dari JSD-plot_v2.3.2  (MZN 2019) download, analysis, and plot data di file terpisah

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

def cek_flare(z):
    kelas = flare_results[z]['goes_class']
    noaa = flare_results[z]['noaa_active_region']
    peak = flare_results[z]['peak_time']
    mulai = flare_results[z]['start_time']
    lokasi = flare_results[z]['goes_location']
    print('urutan:===== ',z)
    print('noaa :',noaa)
    print('kelas :',kelas)
    print('peak :',peak)
    print('start :',mulai)
    print('lokasi :',lokasi)  
    
def plot(parameter1='free_E_loc',parameter2='TOTUSJZ_loc',parameter3='TOTUSJH_loc',parameter4='NC_ratio'):
    fig,ax=plt.subplots(4,sharex=True,figsize=(8,15))
    #prop_cycle = plt.rcParams['axes.prop_cycle']
    #colors = prop_cycle.by_key()['color']
    axi = ax[0]
    axi.plot(out_data.index,out_data[parameter1],'g.-',label='AR '+str(aregion))
    axi.set_ylabel('Free Energy \n($x10^{23}$ $Erg$ $cm^{-1}$)',size=16)
    axi.tick_params(labelsize=18)
    
    axi = ax[1]
    axi.plot(out_data.index,out_data[parameter2],'g.-',label='AR '+str(aregion))
    axi.set_ylabel('Vertical Current \n($x10^{13}$ $Ampere$)',size=16)
    axi.tick_params(labelsize=18)
    
    axi=ax[2]
    axi.plot(out_data.index,out_data[parameter3],'g.-',label='AR '+str(aregion))
    axi.set_ylabel('Total Current Helicity\n ($G^{2} m^{-1}$)',size=16)
    axi.tick_params(labelsize=18)
    
    axi=ax[3]
    axi.plot(out_data.index,out_data[parameter4],'g.-',label='AR '+str(aregion))
    axi.set_ylabel(r'|DC|/|RC|',size=16) #r berfungsi semua simbol tidak ada artinya alpabet saja
    axi.xaxis.set_major_locator(dates.AutoDateLocator())
    axi.xaxis.set_major_formatter(dates.DateFormatter('%H\n%b %d'))
    axi.xaxis.set_minor_locator(AutoMinorLocator())
    axi.set_xlabel('Time (UT)',size=16)
    axi.tick_params(labelsize=16)
    nums = [3,4,5,6,7,9,10,15,16,17,19,20,21,22,23] #cek manual karena ada flare 12014 dan 12010 dianggap 12017
    for i in nums:        
        peak_time = pd.to_datetime(flare_results[i]['peak_time'].value)
        #start_time = pd.to_datetime(flare_results[i]['start_time'].value)
        #end_time = pd.to_datetime(flare_results[i]['end_time'].value)
        goes_class = flare_results[i]['goes_class']
        for axi in ax.flatten():
            if goes_class[0] == 'C':
                axi.axvline(peak_time,color='purple')
                #ax.axvspan(start_time,end_time,color='orange',alpha=0.7,label = goes_class)
            elif goes_class[0] == 'M':
                axi.axvline(peak_time,color='gold',label = goes_class)
                #ax.axvspan(start_time,end_time,color='red',alpha=0.7,label = goes_class)
            else:
                axi.axvline(peak_time,color='blue',label = goes_class)
                #ax.axvspan(start_time,end_time,color='maroon',alpha=0.7,label = goes_class)
            axi.legend(loc='best', numpoints=1,framealpha=0.5,fontsize=10)      
    #plt.grid(True)
    print("Saving figure of",aregion)
    fig.tight_layout()
    fig.savefig(str(aregion)+'/plot_output_all/'+str(aregion)+'_WIL00'+str(num), dpi=600)
    plt.clf()
    plt.close(fig) 

tr = TimeRange([ time_str+' 00:00', time_str2+' 23:00'])
flare_results =sunpy.instr.goes.get_goes_event_list(tr, goes_class_filter=flare_class)
n_flare = len(flare_results)

kolom = ['time','free_E','free_E_loc','TOTUSJZ','TOTUSJZ_loc','TOTUSJH_loc','NC_ratio']   
out_data = pd.read_csv(str(aregion)+'/output_all/OUTPUT_'+str(num)+'_filter.txt', sep = ' ', names=kolom)

time_new = []
for i in range(out_data.shape[0]):
    time_new.append(datetime.datetime.strptime(out_data.time[i],'%y/%m/%d %H:%M:%S'))
out_data['time_new'] = time_new   
out_data.index = out_data.time_new

    