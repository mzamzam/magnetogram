# diadaptasi dri JSD-get_v2.3  (MZN 2019) download, analysis, and plot data di file terpisah

import os.path
import numpy as np
import drms
import pandas as pd
from datetime import datetime, timedelta
start_time = datetime.now()
wdir = r'C:\Users\Stargazers\PycharmProjects\magnetogram'
flare = pd.read_excel('{}\KP-data_lengkap.xlsx'.format(wdir))
def tanggal_jsoc(i):
    date = datetime.strptime(flare.loc[i]['Tanggal & Waktu'],'%Y-%m-%d T%H:%M')
    tanggal = date.strftime('%Y.%m.%d_%H:%M:%S_TAI')
    return tanggal,date
urutan = 0
ar = int(flare.loc[urutan]['No.NOAA'])
series = 'hmi.sharp_cea_720s'
sharp_noaa = pd.read_csv('http://jsoc.stanford.edu/doc/data/hmi/harpnum_to_noaa/all_harps_with_noaa_ars.txt',sep=' ',index_col='HARPNUM')
sharpnum =  sharp_noaa[sharp_noaa['NOAA_ARS'].str.contains(str(ar))].index[0]
segments = ['Bp', 'Bt', 'Br']
kwlist = ['T_REC', 'LON_FWT','OBS_VR','CRPIX1', 'CRPIX2', 'CDELT1', 'CDELT2', 'CRVAL1', 'CRVAL2']
c = drms.Client(email = 'mzn5412@gmail.com') #verbose untuk menampilkan status downloading
# kk = c.query('%s[%d]' % (series, sharpnum), key=kwlist) #saat kenampakan di earth-side
def path(chapter,time):
    nama_folder = str(ar)+'/'+chapter+'_'+str(time[8:10])
    os.makedirs(nama_folder, exist_ok=True)
    return(nama_folder)
def unduh_fits(j,z):
    time = (tanggal_jsoc(j)[1] - timedelta(hours = int(z))).strftime('%Y.%m.%d_%H:%M:%S_TAI')
    k = c.query('%s[%d][%s]' % (series, sharpnum,time), key=kwlist, rec_index=True)
    #Find the record that is clostest to the central meridian, by using the minimum of the patch's absolute longitude:
    rec_cm = k.LON_FWT.idxmin()
    t_cm = drms.to_datetime(k.T_REC[rec_cm])
    # print(rec_cm, '@', k.LON_FWT[rec_cm], 'deg')
    t_cm_str = t_cm.strftime('%Y%m%d_%H%M%S_TAI')
    os.chdir(wdir)
    k.to_csv(path('meta',time)+'/k_'+t_cm_str[9:15]+'.csv',index_label='query')
    os.chdir(path('Fits',time))
    fname_mask = '{series}.{sharpnum}.{tstr}.{segment}.fits'
    fnames = {
        s: fname_mask.format(
            series=series, sharpnum=sharpnum, tstr=t_cm_str, segment=s)
        for s in segments}
    download_segments = []
    for w, v in fnames.items():
        if not os.path.exists(v):
            os.chdir(wdir)
            download_segments.append(w)
            print('{} terunduh.'.format(v))
        else: print('{} sudah ada di folder.'.format(v))
    if download_segments:
        exp_query = '%s{%s}' % (rec_cm, ','.join(download_segments))
        r = c.export(exp_query)
        r.download(path('Fits',time))
# for m in [1,-1]:
#     for i in np.arange(0,6*m,1*m):
#         try: unduh_fits(urutan,i)
#         except KeyError:
#             print('{} tidak ada data.'.format(str(i)))
#             continue
print('time consume: ',datetime.now()-start_time)