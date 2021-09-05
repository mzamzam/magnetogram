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
wdir = r'C:\Users\Stargazers\PycharmProjects\magnetogram'
flare = pd.read_excel('KP-data_lengkap.xlsx')
segments = ['Bp', 'Bt', 'Br']
sharp_noaa = pd.read_csv('http://jsoc.stanford.edu/doc/data/hmi/harpnum_to_noaa/all_harps_with_noaa_ars.txt',sep=' ',index_col='HARPNUM')
# sharp_noaa = pd.read_csv('{}/harps.txt'.format(wdir),sep= ' ',index_col='HARPNUM')
# i = urutan data excel.
# j = selisih hari (... h-1,h0,h1 ...)
# m = data ke- [0,1,2,3]
def ar_sharpnum(i):
    ar = int(flare.loc[i]['No.NOAA'])
    sharpnum = sharp_noaa[sharp_noaa['NOAA_ARS'].str.contains(str(ar))].index[0]
    return ar,sharpnum
def date_time(i,j):
    date = datetime.strptime(flare.loc[i]['Tanggal & Waktu'],'%Y-%m-%d T%H:%M')
    date_str = date.strftime('%Y.%m.%d_%H:%M:%S_TAI')
    time = date - timedelta(days=int(j))
    time_str = time.strftime('%Y.%m.%d_%H:%M:%S_TAI')
    return date_str,time,time_str
def path_folder(chapter,i,j):
    nama_folder = '{}/{}_{}/'.format(str(ar_sharpnum(i)[0]),chapter,str(date_time(i,j)[1].day).zfill(2))
    os.makedirs(nama_folder, exist_ok=True)
    return(nama_folder)
def read_fits_data(fname):
    """Reads FITS data and fixes/ignores any non-standard FITS keywords."""
    hdulist = fits.open(fname)
    hdulist.verify('silentfix+warn')
    return hdulist[1].data
def fnames(i,j,m):
    os.chdir(wdir)
    os.chdir(path_folder('meta',i,j))
    k = pd.read_csv(os.listdir()[m],index_col = 'query')
    rec_cm = k.LON_FWT.idxmin()
    k_cm = k.loc[rec_cm]
    t_cm = drms.to_datetime(k.T_REC[rec_cm])
    t_cm_str = t_cm.strftime('%Y%m%d_%H%M%S_TAI')
    os.chdir(wdir)
    os.chdir(path_folder('Fits',i,j))
    fname_mask = '{series}.{sharpnum}.{tstr}.{segment}.fits'
    fname = {
        s: fname_mask.format(
            series=series, sharpnum=ar_sharpnum(i)[1], tstr=t_cm_str, segment=s)
        for s in segments}
    # os.chdir(wdir)
    return fname,k_cm,t_cm_str,t_cm
def zeros(nx_brad,ny_brad):
    return np.zeros((nx_brad,ny_brad),float)
def rc(cmap):
    plt.rc('mathtext', default='regular')
    plt.rc('image', origin='lower', interpolation='nearest', cmap=cmap)
def img_id(name,i,j,m):
    return ('{}_{}_{}.png'.format(name,str(ar_sharpnum(i)[0]),fnames(i,j,m)[2][6:]), name)
def savefigs(name, title, data, cb_label, extent, a,i,j,m):  # a == 0 khusus bz
    if a == 0:
        rc('gray')
    else:
        rc('seismic')
    fig = plt.figure()
    plt.title(title)
    ax = plt.gca()
    plt.axis('off')
    if a == 0:
        im_mag = plt.imshow(data, extent=extent, vmin=-1, vmax=1)
        plt.text(extent[0] + .3, extent[2] + .3, '{}_{}'.format(str(ar_sharpnum(i)[0]),str(fnames(i,j,m)[2])), dict(size=6))
    else:
        im_mag = plt.imshow(data)
        plt.text(5, 5, '{}_{}'.format(str(ar_sharpnum(i)[0]),str(fnames(i,j,m)[2])), dict(size=6))
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="2.2%", pad=0.1)
    cbar = plt.colorbar(im_mag, cax=cax)
    cbar.set_label(cb_label, size=8)
    cbar.ax.tick_params(labelsize=8)
    if a != 0:
        plt.clim(-1500, 1500)
    else:
        plt.clim(None, None)
    nama_folder_gambar = '{}/{}_{}/{}/'.format(str(ar_sharpnum(i)[0]),str(date_time(i,j)[1].year),str(date_time(i,j)[1].month).zfill(2),img_id(name,i,j,m)[1])
    os.chdir(wdir)
    os.makedirs(nama_folder_gambar, exist_ok=True)
    path = os.path.join(os.getcwd(),nama_folder_gambar, img_id(name,i,j,m)[0])
    fig.savefig(path)
    plt.close(fig)
def cea(i,j,m):
    #1 read data
    bphi = read_fits_data(fnames(i,j,m)[0]['Bp'])
    bth = read_fits_data(fnames(i,j,m)[0]['Bt'])
    brad = read_fits_data(fnames(i,j,m)[0]['Br'])
    #2 Plotting Bz in CEA coordinate ###
    k_cm = fnames(i,j,m)[1]
    ny_brad, nx_brad = brad.shape
    xmin = (1 - k_cm.CRPIX1) * k_cm.CDELT1 + k_cm.CRVAL1
    xmax = (nx_brad - k_cm.CRPIX1) * k_cm.CDELT1 + k_cm.CRVAL1
    ymin = (1 - k_cm.CRPIX2) * k_cm.CDELT2 + k_cm.CRVAL2
    ymax = (ny_brad - k_cm.CRPIX2) * k_cm.CDELT2 + k_cm.CRVAL2

    bx,by,bz,bxp,byp,bzp = [zeros(nx_brad,ny_brad)]*6

    bx = bphi
    by = -bth
    bz = brad
    extent = (xmin - abs(k_cm.CDELT1) / 2, xmax + abs(k_cm.CDELT1) / 2,
              ymin - abs(k_cm.CDELT2) / 2, ymax + abs(k_cm.CDELT2) / 2)

    ys = abs(ymax - ymin) / 2
    xs = abs(xmax - xmin) / 2
    os.chdir(wdir)
    return bx,by,bz,extent,bxp,byp,bzp,ys,xs,bphi,bth,brad
def bz_cont(i,j,m):
    return savefigs('Bz','',cea(i,j,m)[2]/1e3,'$B_{\mathrm{z}}$ [kG]',cea(i,j,m)[3],0,i,j,m)
### CALCULATE POTENTIAL FIELD USING FOURIER METHOD (BASED ON ALISSANDRAKIS, 1981)###
# Search for the maximum values
def energy(x,y,m):
    bx,by,bz,extent,bxp,byp,bzp,ys,xs,bphi,bth,brad = [cea(x,y,m)[i] for i in range(12)]
    maxv = [bphi.max(), bth.max(), brad.max()]
    norm = (max(maxv))
    # Normalization
    bxn = bx / norm
    byn = by / norm
    bzn = bz / norm
    # Discrete element
    nx = bz.shape[0]  # number of pixel in x-direction
    ny = bz.shape[1]  # number of segment in y-direction
    # Here we changed the definition of x and y, which is horizontal for y and vertical for x
    # Length of the data in x and y-axis is normalized by y-axis size
    dy = 1 / ny
    dx = dy
    pi2 = 2 * np.math.pi
    # define normalized length unit. This is applied only for the calculation of potential field.
    yl = 1
    xl = (nx / ny) * yl
    # discretization of axis
    xc = np.zeros((nx), float)
    xc[0] = 0
    for i in range(nx - 1):
        xc[i + 1] = xc[i] + dx
    yc = np.zeros((ny), float)
    yc[0] = 0
    for j in range(ny - 1):
        yc[j + 1] = yc[j] + dy
    # Define wave numbers
    kx,ky,kz = [np.zeros((i), float) for i in [nx,ny,(nx,ny)]]
    nxh = int(nx / 2) + (nx % 2 > 0)
    for i in range((nxh)):
        kx[i] = (i / xl) * pi2
    for i in range(nxh + 1, nx):
        kx[i] = ((i - nx) / xl) * pi2
    nyh = int(ny / 2) + (ny % 2 > 0)
    for j in range((nyh)):
        ky[j] = (j / yl) * pi2
    for j in range(nyh + 1, ny):
        ky[j] = ((j - ny) / yl) * pi2
    for i in range(nx):
        for j in range(ny):
            kz[i, j] = np.sqrt((kx[i] ** 2) + (ky[j] ** 2))
    ima = 1j
    null = 0
    # Calculate Fourier components
    bxz = np.fft.fft2(bx)
    byz = np.fft.fft2(by)
    bzz = np.fft.fft2(bz)
    # debug
    ffxp,ffyp = [np.zeros((nx, ny), complex)]*2
    brfx,brfy = [np.zeros((nx, ny), float)]*2
    # Calculate potential field
    for i in range(nx):
        for j in range(ny):
            if kz[i, j] != 0:
                ffyp[i, j] = -ima * (kx[i] / kz[i, j]) * bzz[i, j]  # because we have changed the index notation, kx refers to the wavelength of By
                ffxp[i, j] = -ima * (ky[j] / kz[i, j]) * bzz[i, j]  # because we have changed the index notation, ky refers to the wavelength of Bx
                brfx[i, j] = np.real(ffxp[i, j])
                brfy[i, j] = np.real(ffyp[i, j])
    fbxp,fbyp = [np.zeros((nx, ny), complex)]*2
    # FFT Inversion
    fbxp,fbyp,fbzp = [np.fft.ifft2(i) for i in [ffxp,ffyp,bzz]]
    bxp = bx  # np.zeros((nx,ny))
    byp = by  # np.zeros((nx,ny))
    bzp = bz  # np.zeros((nx,ny))
    bxp = np.real(fbxp)
    byp = np.real(fbyp)
    bzp = np.real(fbzp)
    ### CALCULATE MAGNETIC ENERGY ####
    dt = np.dtype([('vect', np.float64, (2,))])
    Bot,Bpt,Bnt = [np.zeros((nx, ny), dtype=dt)]*3
    Brt_mag,Bnt_mag,Bot_mag,Bpt_mag = [np.zeros((nx, ny), float)]*4  # total flux
    cos_shear,shear_ang,shear_deg,exc_erg,denum = [np.zeros((nx, ny), float)]*5
    pixr = 0.36442476  # 1 pixel in SHARP data equals to 0.36442476 Mm
    pixrcm = pixr * (10 ** 8)  # convert to cm unit
    xr = nx * pixrcm  # x-length in cm unit
    yr = ny * pixrcm  # y-length in cm unit
    Ar = yr * xr  # total area in square-cm
    dAr = pixrcm ** 2  # area-element in square-cm-unit
    # Define the horizontal observed and potential field as vectors and calculate their intensities
    for i in range(nx):
        for j in range(ny):
            Bot[i, j] = [bx[i, j], by[i, j]]  # only note (useless)
            av = np.array([bx[i, j], by[i, j]])
            Bot_mag[i, j] = np.linalg.norm(av)
            Bpt[i, j] = [bxp[i, j], byp[i, j]]  # only note (useless)
            bv = np.array([bxp[i, j], byp[i, j]])
            Bpt_mag[i, j] = np.linalg.norm(bv)
            cv = np.subtract(av, bv)
            Bnt_mag[i, j] = np.linalg.norm(cv)
            Brt_mag[i, j] = np.sqrt((Bot_mag[i, j] ** 2) + (bz[i, j] ** 2))
            # Calculate shear angle between observed and potential field
            denum[i, j] = (Bot_mag[i, j] * Bpt_mag[i, j])
            if denum[i, j] != 0:
                cos_shear[i, j] = (av @ bv) / denum[i, j]
            shear_ang[i, j] = np.arccos(cos_shear[i, j])
            shear_deg[i, j] = np.math.degrees(shear_ang[i, j])
            # Calculate proxy of the photospheric magnetic energy
            exc_erg[i, j] = (Bnt_mag[i, j] ** 2) * dAr / (8 * np.math.pi)
    exc_ergn = np.zeros((nx, ny), float)
    for i in range(nx):
        for j in range(ny - 1):
            if abs(bzn[i, j]) >= 0.1:
                exc_ergn[i, j] = exc_erg[i, j]
            else:
                exc_ergn[i, j] = 0.0
    pot_erg,tot_erg = [np.zeros((nx, ny), float)]*2
    for i in range(nx):
        for j in range(ny):
            pot_erg[i, j] = (Bpt_mag[i, j] ** 2) * dAr / (8 * np.math.pi)
            tot_erg[i, j] = (Bot_mag[i, j] ** 2) * dAr / (8 * np.math.pi)
    shear_degn = np.zeros((nx, ny), float)
    for i in range(nx):
        for j in range(ny - 1):
            if abs(bzn[i, j]) >= 0.1:
                shear_degn[i, j] = shear_deg[i, j]
            else:
                shear_degn[i, j] = 0.0
    #######Calculate vertical current density in the photosphere###############
    # J_z = dB_y/dx - dB_x/dy (Using 2nd order accuracy center finite difference/ 5-point stencils)
    #######################
    Jz,Jzn,dbxdy,dbxdyr,dbydx,dbydxr,dbyn,dbyr,dbxn,dbxr = [np.zeros((nx, ny))]*10
    # dB_y/dx
    for i in range(nx):
        for j in range(ny - 2):
            dbyn[i, j] = -byn[i, j + 2] + 8 * byn[i, j + 1] - 8 * byn[i, j - 1] + byn[i, j - 2]
            dbyr[i, j] = -by[i, j + 2] + 8 * by[i, j + 1] - 8 * by[i, j - 1] + by[i, j - 2]
    dbydxr = dbyr / (12 * pixrcm)  # in Gauss/cm unit (real value)
    # dB_x/dy
    for i in range(nx - 2):
        for j in range(ny):
            dbxn[i, j] = -bxn[i + 2, j] + 8 * bxn[i + 1, j] - 8 * bxn[i - 1, j] + bxn[i - 2, j]
            dbxr[i, j] = -bx[i + 2, j] + 8 * bx[i + 1, j] - 8 * bx[i - 1, j] + bx[i - 2, j]
    dbxdyr = dbxr / (12 * pixrcm)  # in Gauss/cm unit (real value)
    #Vertical Current (Jz)
    # Plot vertical current in mA per m^2 unit
    mu = np.math.pi * 4 * 10 ** (-7)  # unit Wb/A m
    Jz = dbydxr - dbxdyr  # in Gauss/cm unit (real value)
    Jz = Jz * 10 ** (-4) * 100  # In SI unit (Tesla /m or Wb/m3 , 1 Tesla is 1 Wb/m2)
    Jz = (Jz * 1000) / (mu)  # in milli-Ampere per square-meter unit

    fig12 = plt.figure()
    rc('gray')
    plt.imshow(bz, alpha=0.9)
    rc('seismic')
    # con_bz = plt.contour(bz,levels=[0]) #contour PIL
    Jzt = np.zeros((nx, ny))
    ##Plot only for the region with Bz over a threshold
    for i in range(nx):
        for j in range(ny - 1):
            if abs(bz[i, j]) >= 100:
                Jzt[i, j] = Jz[i, j]
            else:
                Jzt[i, j] = 0.0

    plt.title('Vertical Current Contour', fontsize=16)
    plt.contour(Jzt, levels=[-20, 20],linewidths=0.5)  # red and blue contours represent 20 mA/m^2 and -20 10 mA/m^2 , respectively
    plt.axis('off')
    ar = ar_sharpnum(x)[0]
    t_cm_str = fnames(x,y,m)[2]
    plt.text(5, 5, '{}_{}'.format(str(ar), str(t_cm_str)), dict(size=6))
    nama_folder_gambar = '{}/{}_{}/{}/'.format(str(ar), str(date_time(x, y)[1].year),
                                               str(date_time(x, y)[1].month).zfill(2), img_id('vertical_current', x, y, m)[1])
    os.chdir(wdir)
    os.makedirs(nama_folder_gambar, exist_ok=True)
    path = os.path.join(os.getcwd(), nama_folder_gambar, img_id('vertical_current_', x, y, m)[0])
    fig12.savefig(path)
    plt.close(fig12)

    JzB,Jzpos,Jzneg = [np.zeros((nx, ny))]*3
    for i in range(nx):
        for j in range(ny):
            JzB[i,j] = Jzt[i,j]*brad[i,j]*mu*10
            if JzB[i,j] >=0:
                Jzpos[i,j]=JzB[i,j]
            else:
                Jzneg[i,j] = JzB[i,j]
    tot_dc = sum(map(sum,Jzpos))
    tot_rc = sum(map(sum,Jzneg))
    sumJzB = sum(map(sum,JzB))
    if sumJzB >= 0:
        ratc = abs(tot_dc) / abs(tot_rc)
        ratcr = abs(tot_rc) / abs(tot_dc)
        print('Positive helicity is more dominant')
    else:
        ratc = abs(tot_dc) / abs(tot_rc)
        ratcr = abs(tot_rc) / abs(tot_dc)
        print('Negative helicity is more dominant')
    print('DC/RC', ratc)
    print('RC/DC', ratcr)
    ###Calculate gradient of magnetic field (forward finite difference 1-order accuracy)
    dbznx,dbzny,dbox,dboy,dbtotx,dbtoty,gradBz,gradBh,gradBtot = [np.zeros((nx, ny))]*9
    for i in range(nx - 1):
        for j in range(ny):
            dbznx[i, j] = bzn[i + 1, j] - bzn[i, j]
            dbox[i, j] = Bot_mag[i + 1, j] - Bot_mag[i, j]
            dbtotx[i, j] = Brt_mag[i + 1, j] - Brt_mag[i, j]
    for i in range(nx):
        for j in range(ny - 1):
            dbzny[i, j] = bzn[i, j + 1] - bzn[i, j]
            dboy[i, j] = Bot_mag[i, j + 1] - Bot_mag[i, j]
            dbtoty[i, j] = Brt_mag[i, j + 1] - Brt_mag[i, j]
    ## Summation of parameters
    sumexc_erg = sum(map(sum, exc_erg))  # Total Free energy
    sumJz_tot = sum(map(sum, abs((Jzt * 1.328054057010576e11) / 1000)))  # Total unsigned vertical current (Ampere)
    sumbrad = sum(map(sum, abs(brad * 1.328054057010576e15)))  # Total unsigned flux (Mx)
    data = {'datetime':t_cm_str,'total_unsigned_flux':sumbrad,'total_unsigned_vertical_current':sumJz_tot,
            'total_free_energy':sumexc_erg,'total_current_helicity':sumJzB,'dc_rc':ratc}
    df = pd.DataFrame([data],columns=['datetime','total_unsigned_flux','total_unsigned_vertical_current',
                                      'total_free_energy','total_current_helicity','dc_rc'])
    df.to_csv('{}/{}/output.csv'.format(wdir,str(ar)),mode='a',index=False,header=False)
    return exc_erg
def savefig_con(name, title, data, bz_data, cb_label, a, b,i,j,m):
    fig = plt.figure()
    plt.title(title, fontsize=16)
    ax = plt.gca()
    plt.axis('off')
    rc('gray')
    plt.imshow(cea(i,j,m), alpha=0.9)
    if a == 0:  # a != 0 khusus bukan proxy
        plt.contour(bz_data, levels=[-500, 500], linewidths=0.5, colors=((1, 1, 0), 'green'))
        if b == 0:
            rc('seismic')  # b != 0 khusus shear angle
        else:
            plt.text(5, 5, '{}_{}'.format(str(ar_sharpnum(i)[0]),str(fnames(i,j,m)[2])), dict(size=6))
            plt.rc('image', origin='lower', interpolation='nearest', cmap='Reds')
        im = plt.imshow(data)
        plt.text(5, 5, '{}_{}'.format(str(ar_sharpnum(i)[0]),str(fnames(i,j,m)[2])), dict(size=6, color='w'))
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="2.2%", pad=0.1)
        cbar = plt.colorbar(im, cax=cax)
        cbar.set_label(cb_label, size=8)
        cbar.ax.tick_params(labelsize=8)
    else:
        rc('seismic')
        plt.contour(data, levels=[50000])
        plt.text(5, 5, '{}_{}'.format(str(ar_sharpnum(i)[0]),str(fnames(i,j,m)[2])), dict(size=6))
    nama_folder_gambar = '{}/{}_{}/{}/'.format(str(ar_sharpnum(i)[0]), str(date_time(i, j)[1].year), str(date_time(i, j)[1].month).zfill(2), img_id(name, i, j, m)[1])
    os.chdir(wdir)
    os.makedirs(nama_folder_gambar, exist_ok=True)
    path = os.path.join(os.getcwd(), nama_folder_gambar, img_id(name, i, j, m)[0])
    fig.savefig(path)
    plt.close(fig)
def free_energy(i,j,m):
    savefig_con('Free_energy','Free Energy (Proxy)',energy(i,j,m),cea(i,j,m)[2],'$E_{\mathrm{pot}} [erg/cm^{\mathrm{3}}$]',0,0,i,j,m)
# savefig_con(img_id('Free_energy'),'Free Energy (Proxy)',exc_erg,bz,'','','$E_{\mathrm{pot}} [erg/cm^{\mathrm{3}}$]',0,0,)
# savefig_con(img_id('Free_energy_contour'),'Free Energy',exc_ergn,bz,'$$E_{\mathrm{pot}} [erg/cm^{\mathrm{3}}$]',1,0)
# savefig_con(img_id('Potential_energy'),'Potential Energy (Proxy)',pot_erg,bz,'','','$E_{\mathrm{pot}} [erg/cm^{\mathrm{3}}$]',0,0)
# savefig_con(img_id('Total_energy'),'Total Energy (Proxy)',tot_erg,bz,'','','$E_{\mathrm{pot}} [erg/cm^{\mathrm{3}}$]',0,0)

# savefig_con(img_id('Shear_angle_degree'),'Shear Angle',shear_deg,bz,'','','Angle (degree)',0,1)
# savefig_con(img_id('Shear_angle_degree_filter'),'Shear Angle',shear_degn,bz,'','','Angle (degree)',0,1)
def saving(param,i):
    start_time = datetime.now()
    for j in np.arange(-3,4,1):
        for m in [0,1,2,3]:
            try: param(i,j,m)
            except IndexError: continue
    print(datetime.now() - start_time)
