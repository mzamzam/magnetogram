# diadaptasi dari JSD-analysis_v2.3  (MZN 2019) download, analysis, and plot data di file terpisah

from __future__ import division, print_function
import os.path
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
import drms
from mpl_toolkits.axes_grid1 import make_axes_locatable
import pandas as pd
from datetime import datetime,timedelta
import matplotlib
matplotlib.use('Agg')
start_time = datetime.now()
#########DOWNLOAD DATA (this part is modified from SUNPY, (drms) should be installed in advance)#######################
wdir = r'C:\Users\Stargazers\PycharmProjects\magnetogram'
flare = pd.read_excel('KP-data_lengkap.xlsx')
series = 'hmi.sharp_cea_720s'
sharp_noaa = pd.read_csv('http://jsoc.stanford.edu/doc/data/hmi/harpnum_to_noaa/all_harps_with_noaa_ars.txt',sep=' ',index_col='HARPNUM')
segments = ['Bp', 'Bt', 'Br']
def ar_sharpnum(i):
    ar = int(flare.loc[i]['No.NOAA'])
    sharpnum = sharp_noaa[sharp_noaa['NOAA_ARS'].str.contains(str(ar))].index[0]
    return ar,sharpnum
x = 11
for y in np.arange(-3,4,1):
    ar = ar_sharpnum(x)[0]
    sharpnum = ar_sharpnum(x)[1]
    def date_time(x,y):
        date = datetime.strptime(flare.loc[x]['Tanggal & Waktu'],'%Y-%m-%d T%H:%M')
        date_str = date.strftime('%Y.%m.%d_%H:%M:%S_TAI')
        time = date - timedelta(days=int(y))
        time_str = time.strftime('%Y.%m.%d_%H:%M:%S_TAI')
        waktu = time.strftime('%Y.%m.%d')
        return date_str,time,time_str,waktu
    time = date_time(x,y)[3]
    def rc(cmap):
        plt.rc('mathtext', default='regular')
        plt.rc('image', origin='lower', interpolation='nearest', cmap=cmap)
    def img_id(name):
        return (name + '_' + str(ar) + '_' + t_cm_str[6:] + '.png', name)
    def filename(nama):
        return (nama + '_' + str(ar) + '_' + t_cm_str[6:], nama)
    def savefigs(img_id, title, data, xlabel, ylabel, cb_label, extent, a):  # a == 0 khusus bz
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
        nama_folder_gambar = path_folder(time[0:4], time[5:7]) + '/' + img_id[1]
        os.makedirs(nama_folder_gambar, exist_ok=True)
        path = os.path.join(path_folder(time[0:4], time[5:7]), img_id[1], img_id[0])
        fig.savefig(path)
        plt.clf()
        plt.close(fig)
    def savefig_con(img_id, title, data, bz_data, xlabel, ylabel, cb_label, a, b):
        fig = plt.figure()
        plt.title(title, fontsize=16)
        ax = plt.gca()
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        rc('gray')
        plt.imshow(bz, alpha=0.9)
        if a == 0:  # a != 0 khusus bukan proxy
            plt.contour(bz_data, levels=[-500, 500], linewidths=0.5, colors=((1, 1, 0), 'green'))
            if b == 0:
                rc('seismic')  # b != 0 khusus shear angle
            else:
                plt.text(5, 5, str(ar) + '_' + str(t_cm_str), dict(size=6))
                plt.rc('image', origin='lower', interpolation='nearest', cmap='Reds')
            im = plt.imshow(data)
            plt.text(5, 5, str(ar) + '_' + str(t_cm_str), dict(size=6, color='w'))
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="2.2%", pad=0.1)
            cbar = plt.colorbar(im, cax=cax)
            cbar.set_label(cb_label, size=8)
            cbar.ax.tick_params(labelsize=8)
        else:
            rc('seismic')
            plt.contour(data, levels=[50000])
            plt.text(5, 5, str(ar) + '_' + str(t_cm_str), dict(size=6))
        nama_folder_gambar = path_folder(time[0:4], time[5:7]) + '/' + img_id[1]
        os.makedirs(nama_folder_gambar, exist_ok=True)
        path = os.path.join(path_folder(time[0:4], time[5:7]), img_id[1], img_id[0])
        fig.savefig(path)
        plt.clf()
        plt.close(fig)
    def read_fits_data(fname):
        """Reads FITS data and fixes/ignores any non-standard FITS keywords."""
        hdulist = fits.open(fname)
        hdulist.verify('silentfix+warn')
        return hdulist[1].data
    def path_folder(chapter, waktu):
        nama_folder = str(ar) + '/' + chapter + '_' + str(waktu) + '/'
        os.makedirs(nama_folder, exist_ok=True)
        return (nama_folder)
    # def save_2d(nama, data):
    #     nama_folder_raw = path_folder('raw', time[8:10]) + '/' + filename(nama)[1]
    #     os.makedirs(nama_folder_raw, exist_ok=True)
    #     path = os.path.join(nama_folder_raw, filename(nama)[0])
    #     f = open(path + ".2d", "wb")
    #     f.write(data)
    #     f.close()
    for ii in range(4):
        # Find the record that is clostest to the central meridian, by using the minimum of the patch's absolute longitude:
        try: k = pd.read_csv('{}/meta_{}/{}'.format(ar,time[8:10],os.listdir('{}/meta_{}'.format(ar,time[8:10]))[ii]), index_col='query') # 0 = m
        except IndexError:break
        rec_cm = k.LON_FWT.idxmin()
        k_cm = k.loc[rec_cm]
        t_cm = drms.to_datetime(k.T_REC[rec_cm])
        print('Timestamp:', t_cm)
        t_cm_str = t_cm.strftime('%Y%m%d_%H%M%S_TAI')
        fname_mask = '{series}.{sharpnum}.{tstr}.{segment}.fits'
        fnames = {
            s: fname_mask.format(
                series=series, sharpnum=sharpnum, tstr=t_cm_str, segment=s)
            for s in segments}
        ############### ANALYSIS PROGRAM STARTS HERE (Johan Muhamad, LAPAN, June 2019)  #########
        # bphi =np.zeros((nx,ny),float)
        # bth  =np.zeros((nx,ny),float)
        # brad =np.zeros((nx,ny),float)

        # Read data
        bphi = read_fits_data(path_folder('fits', time[8:10]) + fnames['Bp'])
        bth = read_fits_data(path_folder('fits', time[8:10]) + fnames['Bt'])
        brad = read_fits_data(path_folder('fits', time[8:10]) + fnames['Br'])

        ## Plotting Bz in CEA coordinate ###
        ny_brad, nx_brad = brad.shape
        xmin = (1 - k_cm.CRPIX1) * k_cm.CDELT1 + k_cm.CRVAL1
        xmax = (nx_brad - k_cm.CRPIX1) * k_cm.CDELT1 + k_cm.CRVAL1
        ymin = (1 - k_cm.CRPIX2) * k_cm.CDELT2 + k_cm.CRVAL2
        ymax = (ny_brad - k_cm.CRPIX2) * k_cm.CDELT2 + k_cm.CRVAL2

        bx = np.zeros((nx_brad, ny_brad), float)
        by = np.zeros((nx_brad, ny_brad), float)
        bz = np.zeros((nx_brad, ny_brad), float)

        bxp = np.zeros((nx_brad, ny_brad), float)
        byp = np.zeros((nx_brad, ny_brad), float)
        bzp = np.zeros((nx_brad, ny_brad), float)

        # Rename the parameters
        bx = bphi
        by = -bth
        bz = brad

        # if abs(180 - k_cm.CROTA2) < 0.1:
        #    bphi = bphi[::-1, ::-1]
        #    bth = bth[::-1, ::-1]
        #    brad = brad[::-1, ::-1]
        #    xmin, xmax = -xmax, -xmin
        #    ymin, ymax = -ymax, -ymin
        # else:
        #    raise RuntimeError('CROTA2 = %.2f value not supported.' % k_cm.CROTA2)
        extent = (xmin - abs(k_cm.CDELT1) / 2, xmax + abs(k_cm.CDELT1) / 2,
                  ymin - abs(k_cm.CDELT2) / 2, ymax + abs(k_cm.CDELT2) / 2)

        # Plotting

        ys = abs(ymax - ymin) / 2
        xs = abs(xmax - xmin) / 2

        savefigs(img_id('Bz'),'',brad/1e3,'CEA-deg','CEA-deg','$B_{\mathrm{rad}}$ [kG]',extent,0)

        ### CALCULATE POTENTIAL FIELD USING FOURIER METHOD (BASED ON ALISSANDRAKIS, 1981)###

        # Search for the maximum values
        maxv = [bphi.max(), bth.max(), brad.max()]
        norm = (max(maxv))

        # Normalization
        bxn = bx / norm
        byn = by / norm
        bzn = bz / norm

        # Discrete element
        nx = bz.shape[0]  # number of pixel in x-direction
        ny = bz.shape[1]  # number of segment in y-direction
        # print(nx, ny)  # Here we changed the definition of x and y, which is horizontal for y and vertical for x
        # Length of the data in x and y-axis is normalized by y-axis size
        dy = 1 / ny
        dx = dy
        # print(dy,dx)

        pi2 = 2 * np.math.pi
        # debug
        # print(pi2)

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
        kx = np.zeros((nx), float)
        # debug
        # print('kx=',kx[10])
        ky = np.zeros((ny), float)

        kz = np.zeros((nx, ny), float)

        nxh = int(nx / 2) + (nx % 2 > 0)

        for i in range((nxh)):
            kx[i] = (i / xl) * pi2
        for i in range(nxh + 1, nx):
            kx[i] = ((i - nx) / xl) * pi2
        # print(kx[10],kx[200])

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
        ffxp = np.zeros((nx, ny), complex)
        ffyp = np.zeros((nx, ny), complex)
        brfx = np.zeros((nx, ny), float)
        brfy = np.zeros((nx, ny), float)

        # Calculate potential field
        for i in range(nx):
            for j in range(ny):
                if kz[i, j] != 0:
                    ffyp[i, j] = -ima * (kx[i] / kz[i, j]) * bzz[
                        i, j]  # because we have changed the index notation, kx refers to the wavelength of By
                    ffxp[i, j] = -ima * (ky[j] / kz[i, j]) * bzz[
                        i, j]  # because we have changed the index notation, ky refers to the wavelength of Bx
                    brfx[i, j] = np.real(ffxp[i, j])
                    brfy[i, j] = np.real(ffyp[i, j])

        # print(ffxp.shape)
        fbxp = np.zeros((nx, ny), complex)
        fbyp = np.zeros((nx, ny), complex)
        # print(fbxp.shape)

        # FFT Inversion
        fbxp = np.fft.ifft2(ffxp)
        fbyp = np.fft.ifft2(ffyp)
        fbzp = np.fft.ifft2(bzz)

        bxp = bx  # np.zeros((nx,ny))
        byp = by  # np.zeros((nx,ny))
        bzp = bz  # np.zeros((nx,ny))

        bxp = np.real(fbxp)
        byp = np.real(fbyp)
        bzp = np.real(fbzp)

        # bxp.flags['C_CONTIGUOUS']

        # print(fbxp.shape)
        # print('fbxp=',fbxp[180,360])

        # print(bxp.shape)

        # Plotting potential horizontal-field components

        # print('real.fbxp=', bxp[180, 360])
        # savefigs(img_id('Bx'),'Bx',bx,'','','Gauss','',1)
        # savefigs(img_id('Bxp'),'Bxp',byp,'','','Gauss','',1)
        # savefigs(img_id('By'),'By',by,'','','Gauss','',1)
        # savefigs(img_id('Byp'),'Byp',bxp,'','','Gauss','',1)

        ### CALCULATE MAGNETIC ENERGY ####

        dt = np.dtype([('vect', np.float64, (2,))])
        Bot = np.zeros((nx, ny), dtype=dt)
        Bpt = np.zeros((nx, ny), dtype=dt)
        Bnt = np.zeros((nx, ny), dtype=dt)

        Brt_mag = np.zeros((nx, ny), float)  # total flux
        Bnt_mag = np.zeros((nx, ny), float)
        Bot_mag = np.zeros((nx, ny), float)
        Bpt_mag = np.zeros((nx, ny), float)

        cos_shear = np.zeros((nx, ny), float)
        shear_ang = np.zeros((nx, ny), float)
        shear_deg = np.zeros((nx, ny), float)
        exc_erg = np.zeros((nx, ny), float)
        denum = np.zeros((nx, ny), float)

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

        pot_erg = np.zeros((nx, ny), float)
        tot_erg = np.zeros((nx, ny), float)
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

        savefig_con(img_id('Free_energy'),'Free Energy (Proxy)',exc_erg,bz,'','','$E_{\mathrm{pot}} [erg/cm^{\mathrm{3}}$]',0,0)
        # savefig_con(img_id('Free_energy_contour'),'Free Energy',exc_ergn,bz,'$$E_{\mathrm{pot}} [erg/cm^{\mathrm{3}}$]',1,0)
        # savefig_con(img_id('Potential_energy'),'Potential Energy (Proxy)',pot_erg,bz,'','','$E_{\mathrm{pot}} [erg/cm^{\mathrm{3}}$]',0,0)
        # savefig_con(img_id('Total_energy'),'Total Energy (Proxy)',tot_erg,bz,'','','$E_{\mathrm{pot}} [erg/cm^{\mathrm{3}}$]',0,0)

        savefig_con(img_id('Shear_angle_degree'),'Shear Angle',shear_deg,bz,'','','Angle (degree)',0,1)
        # savefig_con(img_id('Shear_angle_degree_filter'),'Shear Angle',shear_degn,bz,'','','Angle (degree)',0,1)

        #######Calculate vertical current density in the photosphere###############
        # J_z = dB_y/dx - dB_x/dy (Using 2nd order accuracy center finite difference/ 5-point stencils)
        #######################
        Jz = np.zeros((nx, ny))
        Jzn = np.zeros((nx, ny))
        dbxdy = np.zeros((nx, ny))
        dbxdyr = np.zeros((nx, ny))
        dbydx = np.zeros((nx, ny))
        dbydxr = np.zeros((nx, ny))
        dbyn = np.zeros((nx, ny))
        dbyr = np.zeros((nx, ny))
        dbxn = np.zeros((nx, ny))
        dbxr = np.zeros((nx, ny))

        # dB_y/dx

        for i in range(nx):
            for j in range(ny - 2):
                dbyn[i, j] = -byn[i, j + 2] + 8 * byn[i, j + 1] - 8 * byn[i, j - 1] + byn[i, j - 2]
                dbyr[i, j] = -by[i, j + 2] + 8 * by[i, j + 1] - 8 * by[i, j - 1] + by[i, j - 2]

        dbydx = dbyn / (12 * dx)  # normalized
        dbydxr = dbyr / (12 * pixrcm)  # in Gauss/cm unit (real value)

        # dB_x/dy

        for i in range(nx - 2):
            for j in range(ny):
                dbxn[i, j] = -bxn[i + 2, j] + 8 * bxn[i + 1, j] - 8 * bxn[i - 1, j] + bxn[i - 2, j]
                dbxr[i, j] = -bx[i + 2, j] + 8 * bx[i + 1, j] - 8 * bx[i - 1, j] + bx[i - 2, j]

        dbxdy = dbxn / (12 * dy)  # normalized
        dbxdyr = dbxr / (12 * pixrcm)  # in Gauss/cm unit (real value)

        ##Vertical Current (Jz)
        Jzn = dbydx - dbxdy
        im_bz = null
        im_jzn = null

        # Plot vertical current in mA per m^2 unit
        mu = np.math.pi * 4 * 10 ** (-7)  # unit Wb/A m
        Jz = dbydxr - dbxdyr  # in Gauss/cm unit (real value)
        Jz = Jz * 10 ** (-4) * 100  # In SI unit (Tesla /m or Wb/m3 , 1 Tesla is 1 Wb/m2)
        Jz = (Jz * 1000) / (mu)  # in milli-Ampere per square-meter unit

        fig12 = plt.figure()
        im_bz = null
        im_jzt = null
        rc('gray')
        im_bz = plt.imshow(bz, alpha=0.9)
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
        im_jzt = plt.contour(Jzt, levels=[-20, 20],
                             linewidths=0.5)  # red and blue contours represent 20 mA/m^2 and -20 10 mA/m^2 , respectively
        # plt.axis('off')
        plt.text(5, 5, str(ar) + '_' + str(t_cm_str), dict(size=6))
        nama_folder_gambar = path_folder(time[0:4], time[5:7]) + '/' + img_id('vertical_current_real')[1]
        os.makedirs(nama_folder_gambar, exist_ok=True)
        path = os.path.join(nama_folder_gambar, img_id('vertical_current')[0])
        fig12.savefig(path)
        plt.clf()
        plt.close(fig12)

        ### CREATE RAW FILES OF BX, BY, BXP, BYP, and BZ
        # save_2d('Bx',bx)
        # save_2d('By',by)
        # save_2d('Bz',bz)

        bxpp = np.ascontiguousarray(bxp, dtype=np.float64)
        bypp = np.ascontiguousarray(byp, dtype=np.float64)

        # save_2d('Bxp',bypp)
        # save_2d('Byp',bxpp)

        # Debug only

        s = 0
        ss = 0
        # print('Debugging for x=', s, 'and y=', ss)
        # print('Bot=', Bot[s, ss])
        # print('bx=', bx[s, ss])
        # print('by=', by[s, ss])
        # print('Bpt=', Bpt[s, ss])
        # print('Bnt=', Bnt[s, ss])
        # print('Bot_mag=', Bot_mag[s, ss])
        # print('Bpt_mag=', Bpt_mag[s, ss])
        # print('Bnt_mag=', Bnt_mag[s, ss])
        # print('denum=', denum[s, ss])
        # print('cos_shear=', cos_shear[s, ss])
        # print('shear_deg=', shear_deg[s, ss])

        #### Analyze local region (AR SCALE) ####

        # Define local boundary for x axis (in pixel)
        nxl1 = 100  # 300   #100  # x minimum
        nxl2 = 600  # 500   #600  # x maximum

        # Define local boundary for yaxis (in pixel)
        nyl1 = 250  # 300 #250 # y minimum
        nyl2 = 500  # 450 #500 # y maximum

        # Define new parameters for local boundaries
        Jz_loc = Jzt[nyl1:nyl2, nxl1:nxl2]  # vertical current
        Bz_loc = brad[nyl1:nyl2, nxl1:nxl2]  # B_rad or Bz

        nloc, yloc = Jz_loc.shape

        ## Plotting parameters for specific region in CEA coordinate ###
        xminl = (nxl1 - k_cm.CRPIX1) * k_cm.CDELT1 + k_cm.CRVAL1
        xmaxl = (nxl2 - k_cm.CRPIX1) * k_cm.CDELT1 + k_cm.CRVAL1
        yminl = (nyl1 - k_cm.CRPIX2) * k_cm.CDELT2 + k_cm.CRVAL2
        ymaxl = (nyl2 - k_cm.CRPIX2) * k_cm.CDELT2 + k_cm.CRVAL2

        extent2 = (xminl - abs(k_cm.CDELT1) / 2, xmaxl + abs(k_cm.CDELT1) / 2,
                   yminl - abs(k_cm.CDELT2) / 2, ymaxl + abs(k_cm.CDELT2) / 2)

        # savefigs(img_id('Bz_local'),'',Bz_loc/1e3,'CEA-deg','CEA-deg','$B_{\mathrm{rad}}$ [kG]',extent2,0)

        erg_loc = np.zeros((nloc, yloc), float)
        erg_loc = exc_erg[nxl1:nxl2, nyl1:nyl2]  ## Free energy local

        # Calculate Direct and Return Currents for local region
        JzBloc = np.zeros((nloc, yloc), float)
        Jzpos = np.zeros((nloc, yloc), float)
        Jzneg = np.zeros((nloc, yloc), float)

        for i in range(nloc):
            for j in range(yloc):
                JzBloc[i, j] = Jz_loc[i, j] * Bz_loc[i, j] * mu * 10  # This is proportional to the current helicity
                if JzBloc[i, j] >= 0:
                    Jzpos[i, j] = JzBloc[i, j]  # direct/return current (depend on which one is more dominant)
                else:
                    Jzneg[i, j] = JzBloc[i, j]  # return/direct current (depend on which one is more dominant)

        tot_dc = sum(map(sum, Jzpos))
        tot_rc = sum(map(sum, Jzneg))

        sumJzBlocD = sum(map(sum, JzBloc))  # total signed current helicity

        if sumJzBlocD >= 0:
            ratc = abs(tot_dc) / abs(tot_rc)
            ratcr = abs(tot_rc) / abs(tot_dc)
            print('Positive helicity is more dominant')
        else:
            ratc = abs(tot_dc) / abs(tot_rc)
            ratcr = abs(tot_rc) / abs(tot_dc)
            print('Negative helicity is more dominant')

        print('DC/RC', ratc)
        print('RC/DC', ratcr)

        # savefig_con(img_id('current_helicity_local'),'Current helicity',JzBloc,Bz_loc,'pixel','pixel','$E_{\mathrm{free}}$ [erg/cm]',0,0)

        ###Calculate gradient of magnetic field (forward finite difference 1-order accuracy)
        dbznx = np.zeros((nx, ny))
        dbzny = np.zeros((nx, ny))
        dbox = np.zeros((nx, ny))
        dboy = np.zeros((nx, ny))
        dbtotx = np.zeros((nx, ny))
        dbtoty = np.zeros((nx, ny))

        gradBz = np.zeros((nx, ny))
        gradBh = np.zeros((nx, ny))
        gradBtot = np.zeros((nx, ny))

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

        gradBz = abs(np.sqrt(((dbznx / dx) ** 2) + ((dbzny / dy) ** 2)))  # Gradient of vertical field
        # im_gradBz = plt.imshow(gradBz)

        gradBh = abs(np.sqrt(((dbox / dx) ** 2) + ((dboy / dy) ** 2)))  # gradient of horizontal field
        # im_gradBh = plt.imshow(gradBh)

        gradBtot = abs(np.sqrt(((dbtotx / dx) ** 2) + ((dbtoty / dy) ** 2)))  # gradient of total field
        # im_gradBtot = plt.imshow(gradBtot)

        ##### OUTPUT FILES 2 (Region Of Interest) ########################

        ## Summation of parameters
        sumexc_erg = sum(map(sum, exc_erg))  # Total Free energy
        sumerg_loc = sum(map(sum, erg_loc))  # Total free energy local
        sumJz_tot = sum(map(sum, abs((Jzt * 1.328054057010576e11) / 1000)))  # Total unsigned vertical current (Ampere)
        sumJz_loc = sum(
            map(sum, abs((Jz_loc * 1.328054057010576e11) / 1000)))  # Total unsigned Local vertical current (Ampere)
        sumJzBloc = sum(map(sum, abs(JzBloc)))  # Total unsigned Local current helicity (Gauss^2/m)
        sumbrad = sum(map(sum, abs(brad * 1.328054057010576e15)))  # Total unsigned flux (Mx)

        time_cm_str = t_cm.strftime('%y%m%d%H%M%S')
        time_cm_str_form = t_cm.strftime("%y/%m/%d %H:%M:%S")

        data = {'Waktu': time_cm_str_form, 'Energi_total': sumexc_erg, 'Energi_Lokal': sumerg_loc,
                'Arus_total': sumJz_tot, 'Arus_lokal': sumJz_loc, 'Helisitas_lokal': sumJzBloc,
                'Rasio_netralitas_arus': ratc}
        df = pd.DataFrame([data],columns=['Waktu', 'Energi_total', 'Energi_Lokal', 'Arus_total', 'Arus_lokal', 'Helisitas_lokal','Rasio_netralitas_arus'])
        df.to_csv('{}/output_1.csv'.format(str(ar)), mode='a', index=False,header=False)

        # file1 = open(str(ar)+'/OUTPUT_1.txt',"a+")
        # file1.write(time_cm_str_form+' '+str(sumexc_erg)+' '+ str(sumerg_loc)+' '+ str(sumJz_tot)+' '+ str(sumJz_loc)+' '+ str(sumJzBloc)+' '+ str(ratc)+'\n')
        # file1.close()

        #### Analyze local region (Region of Interest)  ####

        # Define local boundary for x axis (in pixel)
        nxl1 = 300  # 300   #100  # x minimum
        nxl2 = 600  # 500   #600  # x maximum

        # Define local boundary for yaxis (in pixel)
        nyl1 = 250  # 300 #250 # y minimum
        nyl2 = 500  # 450 #500 # y maximum

        # Define new parameters for local boundaries
        Jz_loc = Jzt[nyl1:nyl2, nxl1:nxl2]  # vertical current
        Bz_loc = brad[nyl1:nyl2, nxl1:nxl2]  # B_rad or Bz

        nloc, yloc = Jz_loc.shape
        ## Plotting parameters for specific region in CEA coordinate ###
        xminl = (nxl1 - k_cm.CRPIX1) * k_cm.CDELT1 + k_cm.CRVAL1
        xmaxl = (nxl2 - k_cm.CRPIX1) * k_cm.CDELT1 + k_cm.CRVAL1
        yminl = (nyl1 - k_cm.CRPIX2) * k_cm.CDELT2 + k_cm.CRVAL2
        ymaxl = (nyl2 - k_cm.CRPIX2) * k_cm.CDELT2 + k_cm.CRVAL2

        extent2 = (xminl - abs(k_cm.CDELT1) / 2, xmaxl + abs(k_cm.CDELT1) / 2,
                   yminl - abs(k_cm.CDELT2) / 2, ymaxl + abs(k_cm.CDELT2) / 2)

        # savefigs(img_id('Bz_local_ROI'),'',Bz_loc/1e3,'CEA-deg','CEA-deg','$B_{\mathrm{rad}}$ [kG]',extent2,0)

        erg_loc = np.zeros((nloc, yloc), float)
        erg_loc = exc_erg[nxl1:nxl2, nyl1:nyl2]  ## Free energy local

        # Calculate Direct and Return Currents for local region
        JzBloc = np.zeros((nloc, yloc), float)
        Jzpos = np.zeros((nloc, yloc), float)
        Jzneg = np.zeros((nloc, yloc), float)

        for i in range(nloc):
            for j in range(yloc):
                JzBloc[i, j] = Jz_loc[i, j] * Bz_loc[i, j] * mu * 10  # This is proportional to the current helicity
                if JzBloc[i, j] >= 0:
                    Jzpos[i, j] = JzBloc[i, j]  # direct/return current (depend on which one is more dominant)
                else:
                    Jzneg[i, j] = JzBloc[i, j]  # return/direct current (depend on which one is more dominant)

        # Nullify parameters
        tot_dc = null
        tot_rc = null
        sumJzBlocD = null
        ratc = null
        ratcr = null

        tot_dc = sum(map(sum, Jzpos))
        tot_rc = sum(map(sum, Jzneg))

        sumJzBlocD = sum(map(sum, JzBloc))  # total signed current helicity

        if sumJzBlocD >= 0:
            ratc = abs(tot_dc) / abs(tot_rc)
            ratcr = abs(tot_rc) / abs(tot_dc)
            print('Positive helicity is more dominant')
        else:
            ratc = abs(tot_dc) / abs(tot_rc)
            ratcr = abs(tot_rc) / abs(tot_dc)
            print('Negative helicity is more dominant')

        print('DC/RC', ratc)
        print('RC/DC', ratcr)

        # savefig_con(img_id('current_helicity_local_ROI'), 'Current helicity', JzBloc, Bz_loc, 'CEA-deg', 'CEA-deg',
        #             '$E_{\mathrm{free}}$ [erg/cm]', 0, 0)

        #####OUTPUT FILES########################

        ## Summation of parameters
        sumerg_loc = null
        sumJz_loc = null
        sumJzBloc = null

        sumerg_loc = sum(map(sum, erg_loc))  # Total free energy local
        sumJz_loc = sum(map(sum, abs((Jz_loc * 1.328054057010576e11) / 1000)))  # Total unsigned Local vertical current
        sumJzBloc = sum(map(sum, abs(JzBloc)))  # Total unsigned Local current helicity

        time_cm_str = t_cm.strftime('%y%m%d%H%M%S')
        time_cm_str_form = t_cm.strftime("%y/%m/%d %H:%M:%S")

        data = {'Waktu': time_cm_str_form, 'Energi_total': sumexc_erg, 'Energi_Lokal': sumerg_loc,
                'Arus_total': sumJz_tot, 'Arus_lokal': sumJz_loc, 'Helisitas_lokal': sumJzBloc,'Rasio_netralitas_arus':ratc}
        df = pd.DataFrame([data], columns=['Waktu', 'Energi_total', 'Energi_Lokal','Arus_total', 'Arus_lokal', 'Helisitas_lokal','Rasio_netralitas_arus'])
        df.to_csv('{}/output_2.csv'.format(str(ar)), mode='a', index=False,header=False)
        # file2 = open(str(ar)+'/OUTPUT_2.txt',"a+")
        # file2.write(time_cm_str_form+' '+str(sumexc_erg)+' '+ str(sumerg_loc)+' '+ str(sumJz_tot)+' '+ str(sumJz_loc)+' '+ str(sumJzBloc)+' '+ str(ratc)+'\n')
        # file2.close()
print('Excecution time ',datetime.now() - start_time)