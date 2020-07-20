# diadaptasi dari JSD-analysis_v2.3  (MZN 2019) download, analysis, and plot data di file terpisah

from __future__ import division, print_function
import os.path
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from astropy.io import fits
import drms
from mpl_toolkits.axes_grid1 import make_axes_locatable
import pandas as pd
from datetime import datetime,timedelta
start_time = datetime.now()
wdir = r'C:\Users\Stargazers\PycharmProjects\magnetogram'
flare = pd.read_excel('KP-data_lengkap.xlsx')
ar = int(flare.loc[0]['No.NOAA'])
series = 'hmi.sharp_cea_720s'
sharp_noaa = pd.read_csv('http://jsoc.stanford.edu/doc/data/hmi/harpnum_to_noaa/all_harps_with_noaa_ars.txt',sep=' ',index_col='HARPNUM')
segments = ['Bp', 'Bt', 'Br']
def tanggal_jsoc(i):
    date = datetime.strptime(flare.loc[i]['Tanggal & Waktu'],'%Y-%m-%d T%H:%M')
    tanggal = date.strftime('%Y.%m.%d_%H:%M:%S_TAI')
    return tanggal,date
time = (tanggal_jsoc(0)[1] - timedelta(hours = int(0))).strftime('%Y.%m.%d_%H:%M:%S_TAI')
def path_folder(ar,chapter,waktu):
    nama_folder = str(ar)+'/'+chapter+'_'+str(waktu)+'/'
    os.makedirs(nama_folder, exist_ok=True)
    return(nama_folder)
def read_fits_data(fname):
    """Reads FITS data and fixes/ignores any non-standard FITS keywords."""
    hdulist = fits.open(fname)
    hdulist.verify('silentfix+warn')
    return hdulist[1].data
def fnames():
    os.chdir(wdir)
    sharpnum =  sharp_noaa[sharp_noaa['NOAA_ARS'].str.contains(str(ar))].index[0]
    os.chdir(path_folder(ar,'meta',time[8:10]))
    k = pd.read_csv(os.listdir()[0],index_col = 'query')
    rec_cm = k.LON_FWT.idxmin()
    k_cm = k.loc[rec_cm]
    t_cm = drms.to_datetime(k.T_REC[rec_cm])
    t_cm_str = t_cm.strftime('%Y%m%d_%H%M%S_TAI')
    os.chdir(wdir)
    os.chdir(path_folder(ar,'Fits',time[8:10]))
    fname_mask = '{series}.{sharpnum}.{tstr}.{segment}.fits'
    fnames = {
        s: fname_mask.format(
            series=series, sharpnum=sharpnum, tstr=t_cm_str, segment=s)
        for s in segments}
    return fnames,k_cm,t_cm_str
#1 read data
bphi = read_fits_data(fnames()[0]['Bp'])
bth = read_fits_data(fnames()[0]['Bt'])
brad = read_fits_data(fnames()[0]['Br'])
#2 Plotting Bz in CEA coordinate ###
k_cm = fnames()[1]
ny_brad, nx_brad = brad.shape
xmin = (1 - k_cm.CRPIX1) * k_cm.CDELT1 + k_cm.CRVAL1
xmax = (nx_brad - k_cm.CRPIX1) * k_cm.CDELT1 + k_cm.CRVAL1
ymin = (1 - k_cm.CRPIX2) * k_cm.CDELT2 + k_cm.CRVAL2
ymax = (ny_brad - k_cm.CRPIX2) * k_cm.CDELT2 + k_cm.CRVAL2

def zeros(nx_brad,ny_brad):
    return np.zeros((nx_brad,ny_brad),float)
bx,by,bz,bxp,byp,bzp = [zeros(nx_brad,ny_brad)]*6

bx = bphi
by = -bth
bz = brad
extent = (xmin - abs(k_cm.CDELT1) / 2, xmax + abs(k_cm.CDELT1) / 2,
          ymin - abs(k_cm.CDELT2) / 2, ymax + abs(k_cm.CDELT2) / 2)

ys = abs(ymax - ymin) / 2
xs = abs(xmax - xmin) / 2
def rc(cmap):
    plt.rc('mathtext', default='regular')
    plt.rc('image', origin='lower', interpolation='nearest', cmap=cmap)
t_cm_str = fnames()[2]
def img_id(name,t_cm_str):
    return (name + '_' + str(ar) + '_' + t_cm_str[6:] + '.png', name)
def savefigs(img_id, title, data, xlabel, ylabel, cb_label, extent, a):  # a == 0 khusus bz
    os.chdir(wdir)
    if a == 0:
        rc('gray')
    else:
        rc('seismic')
    fig = plt.figure()
    plt.title(title)
    ax = plt.gca()
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if a == 0:
        im_mag = plt.imshow(data, extent=extent, vmin=-1, vmax=1)
        plt.text(extent[0] + .3, extent[2] + .3, str(ar) + '_' + str(t_cm_str), dict(size=6))
    else:
        im_mag = plt.imshow(data)
        plt.text(5, 5, str(ar) + '_' + str(t_cm_str), dict(size=6))
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="2.2%", pad=0.1)
    cbar = plt.colorbar(im_mag, cax=cax)
    cbar.set_label(cb_label, size=8)
    cbar.ax.tick_params(labelsize=8)
    if a != 0:
        plt.clim(-1500, 1500)
    else:
        plt.clim(None, None)
    nama_folder_gambar = path_folder(ar,time[0:4], time[5:7]) + '/' + img_id[1]
    os.makedirs(nama_folder_gambar, exist_ok=True)
    path = os.path.join(path_folder(ar,time[0:4], time[5:7]), img_id[1], img_id[0])
    fig.savefig(path)
    plt.close(fig)
savefigs(img_id('Bz',t_cm_str),'',brad/1e3,'CEA-deg','CEA-deg','$B_{\mathrm{rad}}$ [kG]',extent,0)
print(datetime.now()-start_time)
