# v2.3  (MZN 2019) download, analysis, and plot data di file terpisah

import os.path
import numpy as np
import drms
import pandas as pd
from datetime import datetime
from astropy.io import fits
start_time = datetime.now()
#########DOWNLOAD DATA (this part is modified from SUNPY, (drms) should be installed in advance)#######################
series = 'hmi.sharp_cea_720s'
# 12017 at 28-29 Maret 2014
ar = 12017
sharp_noaa = pd.read_csv('http://jsoc.stanford.edu/doc/data/hmi/harpnum_to_noaa/all_harps_with_noaa_ars.txt',sep=' ',index_col='HARPNUM')
sharpnum =  sharp_noaa[sharp_noaa['NOAA_ARS'].str.contains(str(ar))].index[0]
segments = ['Bp', 'Bt', 'Br']
kwlist = ['T_REC', 'LON_FWT','OBS_VR','CRPIX1', 'CRPIX2', 'CDELT1', 'CDELT2', 'CRVAL1', 'CRVAL2']
c = drms.Client(email = 'mzn5412@gmail.com') #verbose untuk menampilkan status downloading
#kk = c.query('hmi.sharp_cea_720s[7117][2017.09.03_05:00:00]', key=kwlist, rec_index=True)
def read_fits_data(fname):
    """Reads FITS data and fixes/ignores any non-standard FITS keywords."""
    hdulist = fits.open('fits_29/'+fname)
    hdulist.verify('silentfix+warn')
    return hdulist[1].data
def path(chapter):
    nama_folder = str(ar)+'/'+chapter+'_'+str(time[8:10])
    os.makedirs(nama_folder, exist_ok=True)
    return(nama_folder)

t_init=0#start time (in hour UT t_init:00:00)
t_term=24 #end time (in hour UT [t_term:00:00-01:00:00])
t_step=1

for i in np.arange(t_init,t_term,t_step): # untuk tgl 27, 15 tidak ada data jadi akan ada pesan eror
    time = '2014.03.28_'+str(i).zfill(2)+':00:00'
    k = c.query('%s[%d][%s]' % (series, sharpnum,time), key=kwlist, rec_index=True)
    #Find the record that is clostest to the central meridian, by using the minimum of the patch's absolute longitude:
    rec_cm = k.LON_FWT.idxmin()
    k_cm = k.loc[rec_cm]
    t_cm = drms.to_datetime(k.T_REC[rec_cm])
    print(rec_cm, '@', k.LON_FWT[rec_cm], 'deg')
    print('Timestamp:', t_cm)
    t_cm_str = t_cm.strftime('%Y%m%d_%H%M%S_TAI')
    k.to_csv(path('meta')+'/k_'+t_cm_str[9:15]+'.csv',index_label='query')
    fname_mask = '{series}.{sharpnum}.{tstr}.{segment}.fits'
    fnames = {
        s: fname_mask.format(
            series=series, sharpnum=sharpnum, tstr=t_cm_str, segment=s)
        for s in segments}
    download_segments = []
    for w, v in fnames.items():
        if not os.path.exists(v):
            download_segments.append(w)
    if download_segments:
        exp_query = '%s{%s}' % (rec_cm, ','.join(download_segments))
        r = c.export(exp_query)
        dl = r.download(path('Fits'))

print('time consume: ',datetime.now()-start_time)