import os
import tqdm
import pandas as pd
from astropy.timeseries import LombScargle
import numpy as np
from astropy.io import fits, ascii
from astropy.table import Table, vstack
from lk_stat_package import lk_stat
import matplotlib.pyplot as plt
from astropy import time, coordinates as coord, units as u
from lmfit import Model, Parameters


params = {
         'mathtext.default': 'regular',
    
          'text.usetex': False}
plt.rcParams.update(params)

plt.rcParams['figure.dpi']=150
plt.rcParams['lines.color']='k'
plt.rcParams['axes.edgecolor']='k'
# plt.rcParams['lines.linewidth']=1
# plt.rcParams['lines.markeredgewidth']=1
plt.rcParams['xtick.minor.visible']=False
plt.rcParams['ytick.minor.visible']=False
plt.rcParams['axes.labelsize']=22
plt.rcParams['xtick.labelsize']=18
plt.rcParams['ytick.labelsize']=18
from scipy.optimize import least_squares

def correct_time(df,mjd_col,ra_col,dec_col,site):
    ip_peg = coord.SkyCoord(df[ra_col].values,df[dec_col].values,
                            unit=(u.deg, u.deg), frame='icrs')
    saao = coord.EarthLocation.of_site(site)
    times = time.Time(list(df[mjd_col]), format='mjd',
                      scale='utc', location=saao) 
    ltt_bary = times.light_travel_time(ip_peg,'barycentric')  
    time_barycentre = times.tdb + ltt_bary
    df.loc[:,'BJD_OBS']=time_barycentre.jd
    
    return df
def remove_outliers(Time,mag,mag_err):

    q3_err, q1_err = np.percentile(mag_err, [75 ,25])
    iqr_err=q3_err-q1_err
    h_err=q3_err+iqr_err*1.5
                 
    Time=Time[mag_err<h_err]
    mag=mag[mag_err<h_err]
    mag_err=mag_err[mag_err<h_err]
        
    return Time,mag,mag_err
def freq_grid(times,oversample_factor=10,f0=None,fn=None):
    times=np.sort(times)
    df = 1.0 / (times.max() - times.min())
    if f0 is None:
        f0 = df
    if fn is None:
        fn = 0.5 / np.median(np.diff(times)) 
    return np.arange(f0, fn, df / oversample_factor)


def fit_amplitude_phase(t, y, f, sigma_y=None):

    omega = 2 * np.pi * f
    X = np.column_stack((np.cos(omega * t), np.sin(omega * t)))

    if sigma_y is not None:
        W = np.diag(1 / sigma_y**2)
        XT_W = X.T @ W
        beta = np.linalg.inv(XT_W @ X) @ XT_W @ y
    else:
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)

    B, C = beta
    A = np.sqrt(B**2 + C**2)
    phi = np.arctan2(B, C)

    
    return A, phi


def jackknife_uncertainty(t, y, yerr, f):
    N = len(t)
    A_vals = []
    phi_vals = []

    for j in range(N):
        t_jack = np.delete(t, j)
        y_jack = np.delete(y, j)
        yerr_jack = np.delete(yerr, j)
        A_j, phi_j = fit_amplitude_phase(t_jack, y_jack, f, None)
        A_vals.append(A_j)
        phi_vals.append(phi_j)

    A_vals = np.array(A_vals)
    phi_vals = np.unwrap(np.array(phi_vals))  # Avoid jumps at p/m π

    A_mean = np.mean(A_vals)
    phi_mean = np.mean(phi_vals)

    sigma_A = np.sqrt((N - 1) / N * np.sum((A_vals - A_mean)**2))
    sigma_phi = np.sqrt((N - 1) / N * np.sum((phi_vals - phi_mean)**2))

    return A_mean, phi_mean, sigma_A, sigma_phi


def optimise_freq(psi,freq,Time,mag,mag_err,oversampling_factor):

    f_step=np.diff(freq)[0]

    idx_peak=np.argmax(psi)
            
    peak_freq=freq[idx_peak]
            
    #Here we take 10 steps (or oversampling_factor steps) before and after the frequency peak as a new frequency search range. 
    
    lower_range=max(freq.min(),peak_freq - (oversampling_factor*f_step))
    upper_range=min(freq.max(),peak_freq + (oversampling_factor*f_step))
    
    fine_grid_freq=freq_grid(Time,oversample_factor=200,f0=lower_range,fn=upper_range)
     
    lsp_fg = LombScargle(t=Time, y=mag, dy=mag_err,nterms=1).power(frequency=fine_grid_freq, method="cython", normalization="psd")
    theta_fg= lk_stat(1/fine_grid_freq, mag, mag_err, Time)
        
    psi_fine_grid=2*lsp_fg/theta_fg
        
    best_freq=fine_grid_freq[np.argmax(psi_fine_grid)]
            
    best_freq=fine_grid_freq[np.argmax(lsp_fg)]
    
    best_period=1/best_freq

    return best_freq

def sine_model(t, freq, amp, phase):
    return amp * np.sin(2 * np.pi * freq * t + phase)

def build_composite_model(n_components):
    def model_func(t, **kwargs):
        result = np.zeros_like(t)
        for i in range(n_components):
            A = kwargs[f'amp{i}']
            f = kwargs[f'freq{i}']
            phi = kwargs[f'phase{i}']
            result += sine_model(t, f, A, phi)
        return result
    return Model(model_func)


# Read BlackGEM light curve of all objects (here 4) from a single file

lc=fits.open("BG_lightcurves/lightcurves_4targets.fits")

# Select one object (source id). A "for" loop can be used instead if one wants to pre-whiten all objects at once. This can take a long time, though.

source_ids=[3600841623951744640]

# Convert the fits table to a pandas dataframe

data=Table(lc[1].data).to_pandas()

# These are light curve quality cuts for BlackGEM. These can be different for other data sources.
data=data[(data['QC-FLAG']!='red')&(data['FLAGS_MASK']==0)&(data['FNU_OPT']>0)]

# Extract the light curve of the specified object or source id
data=data[data['SOURCE_ID']==source_ids[0]]

# Specify the pass band to be used. A "for" loop can be used to run multiple bands in a single run (e.g. passbands=['q','u','i'] for BlackGEM).

passbands=['i']

passband='i'

target_name=source_ids[0]

# Convert MJD time to BJD time coordinates to account for the Rømer delay. This line can be commented if time is already in BJD.

data=correct_time(data,"MJD-OBS","RA","DEC","lasilla")

# Choose a unique reference time t0 (or midpoint) for the phase calculations. Here the minimum of the timestamps in the q-band light curve is used as t0 in the u- and i-band light curves.

midpoint=np.min(data['BJD_OBS'][data['FILTER']=='q'].values)

# Select the light curve corresponding to the selected passband

df=data[data['FILTER']==passband]

# Select flux, flux error, and time columns
fnu_opt = df['FNU_OPT'].dropna().values
fnu_err_opt=df['FNUERRTOT_OPT'].dropna().values
Time = df['BJD_OBS'].dropna().values

# Convert flux to magnitude
mag = -2.5*np.log10(fnu_opt)+ 23.9
mag_err=(fnu_err_opt/fnu_opt)*(2.5/np.log(10))


Time = Time - midpoint

# Convert data types to double (required by the LK statistic function).
Time=Time.astype(np.float64)
mag=mag.astype(np.float64)
mag_err=mag_err.astype(np.float64)


# Path to save the plots. If it does not exist, it will be created below
fig_path="prewhite_%s_%s_processed"%(str(target_name),passband)

# Path to save the periodogram data. If it does not exist, it will be created below
save_ls_to=f"/idia/users/princy/BG_periodograms_{target_name}/"


if os.path.exists(fig_path)==False:
    os.makedirs(fig_path)
    
if os.path.exists(save_ls_to)==False:
    os.makedirs(save_ls_to+"lsp")
    os.makedirs(save_ls_to+"theta")


prewhitening_params=[] # A list to save parameters, such as frequency, amplitude, and phase

offset=np.median(mag)  # Required in the amplitude computation

y_original=np.copy(mag) - offset # Original light curve
residual=np.copy(mag) - offset # (initial) residual light curve

# These are for plotting reasons. To see area where freq extracted in iteration i-1 lies in iteration i.
previous_freq_window=0
power_freq_widow=0


# Specifiy oversampling factor, fmin (f0), and fmax (fn) to create the frequency grid.
oversampling_factor=10
f0=20
fn=200
frequencies = freq_grid(Time, oversample_factor=oversampling_factor, f0=f0, fn=fn)

# To build a composite model with n components. The composite model is used to fit and optimise all extracted frequencies, amplitudes, and phases at the end of the iteration. These parameters are added to the composite model at each iteration using "Parameters()"

n_components = 0
params = Parameters()

# Here, we use 20 iterations. A stopping criterion can also be defined instead.
n_iterations = 20

for i in range(n_iterations):

    periods = 1 / frequencies
    
    lsp_path = save_ls_to + 'lsp/' + f'{target_name}_{passband}_{i}.npy'
    theta_path = save_ls_to + 'theta/' + f'{target_name}_{passband}_{i}.npy'

    # Compute the Lomb-Scargle periodogram. Parameters can be different for other types of data (e.g., for TESS,
    # method="fast" can be used for faster computing)

    lsp = LombScargle(Time, residual, mag_err, nterms=1).power(
                                    frequencies, method="cython", normalization="psd"
                                        )
    # The LK stat is particularly designed for sparsely sampled data. For almost continuous data, such as TESS, the Lomb-Scargle periodogram alone is enough. In this case, comment the theta line and replace psi and psi_norm by lsp and lsp_norm throughout the code.
    theta = lk_stat(periods, residual, mag_err, Time)

    psi=2*lsp/theta
    
    psi_norm=psi/psi.max() # or lsp_norm=lsp/lsp.max() if theta is ignored.
    
    # Compute the window function
    lspw = LombScargle(Time, np.ones(len(Time)), None, nterms=1,fit_mean=False,center_data=False).power(
        frequencies, method="cython", normalization="psd"
            )
    
    freq_window=frequencies[np.argmax(lspw)]
    
    period_window=(1/freq_window)

    # Exclude aliasing frequencies around a window width delta_f. Can be adjusted as needed.
    delta_f=0.1

    exclude_mask = (np.abs(frequencies - freq_window) < freq_window*delta_f)
    
    psi[exclude_mask]=0

    
    # Optimise the peak frequency around a small window.
    
    best_freq=optimise_freq(psi,frequencies,Time,residual,mag_err,oversampling_factor)

    # Compute uncertainties using the Jackknife technique
    
    A_jk, phi_jk, sigma_A_jk, sigma_phi_jk = jackknife_uncertainty(Time, residual, mag_err,best_freq)

    
    print(f"Amplitude: {A_jk}")
    print(f"freq: {best_freq}")

    
    params.add(f'freq{n_components}', value=best_freq, vary=True) 
    params.add(f'amp{n_components}', value=A_jk, vary=True)
    params.add(f'phase{n_components}', value=phi_jk, vary=True)

    
    # Build and fit the composite model using all parameters (including those in the previous iterations)
    model = build_composite_model(n_components+1)
    result = model.fit(y_original, t=Time, weights=1.0 / mag_err, params=params)
    params = result.params 


    A_jk = params[f'amp{i}'].value
    best_freq = params[f'freq{i}'].value
    phi_jk = params[f'phase{i}'].value

    print(f"Optimised amplitude: {A_jk}")
    print(f"Optimised freq: {best_freq}")
    

    period_est=1/best_freq

    period_second=period_est*86400

    std=np.std(residual)

    prewhitening_params.append([best_freq,period_second, A_jk, sigma_A_jk, phi_jk , sigma_phi_jk,std])

    phase_fit=np.linspace(0,1,1000)

    # remove outliers in the light curve. These cleaned parameters are only used for clean plotting reasons.

    Time_clean,residual_clean,mag_err_clean=remove_outliers(Time,residual,mag_err)
    
    phase=(Time_clean/period_est)%1

    # y_model_best_freq is a sinusoidal model corresponding to the frequency, amplitude, and phase extracted in the current iteration
    
    y_model_best_freq=sine_model(Time_clean, best_freq, A_jk, phi_jk)
                      
    # y_model_all_freq includes all parameters from previous iterations
    y_model_all_freq=result.best_fit
    
    
    # The following plotting lines can be commented if not needed
    color='#1f77b4'
                
    plt.figure(figsize=[8,8],facecolor='white')
                
    plt.subplot(311)
    plt.title('Gaia ID %s'%(str(source_ids[0])),fontsize=15)
    # plt.title(r'Period: %.5f $\pm$ %.3e d.'%(period_est,period_err),fontsize=15)
    plt.plot(frequencies,psi_norm)
    plt.plot(frequencies[np.argmax(psi)],psi_norm[np.argmax(psi)],marker='*')
    plt.plot(previous_freq_window,power_freq_widow,alpha=0.5)
    
    plt.ylabel(r'$\Psi$')
    plt.xlabel(r'Frequency ($d^{-1}$)')
    # plt.xlim([130,165])
    # plt.xlim([0,50])
    plt.minorticks_on()
    plt.tick_params('both', length=5, width=1., which='major', direction='in')	
    plt.tick_params('both', length=2, width=1, which='minor', direction='in')

    plt.subplot(312)
            
    plt.title(r'Period window: %.2f hr.'%(period_window*24),fontsize=15)
    plt.plot(frequencies,lsp/lsp.max())
    plt.plot(frequencies,lspw/lspw.max(),alpha=0.5)
    plt.plot(frequencies[np.argmax(psi)],1,marker='*')


    plt.ylabel(r'$LSP$')
    plt.xlabel(r'Frequency ($d^{-1}$)')
      
    plt.minorticks_on()
    plt.tick_params('both', length=5, width=1., which='major', direction='in')	
    plt.tick_params('both', length=2, width=1, which='minor', direction='in')
    
    plt.subplot(313)
    plt.title(r'Period : %.3f hr, Freq: %.3f $d^{-1}$'%(period_est*24,best_freq),fontsize=15)
    # plt.title(r'Period: %.5f $\pm$ %.3e d.'%(period_est,period_err),fontsize=15)
    plt.errorbar(phase,residual_clean,yerr=mag_err_clean,fmt='o',color=color,zorder=0)
    plt.errorbar(phase+1,residual_clean,yerr=mag_err_clean,fmt='o',color=color,zorder=0)
        
    plt.plot(phase, y_model_best_freq,'ok',zorder=1)
    plt.plot(phase+1, y_model_best_freq,'ok',zorder=1)
   
    
    
    plt.xlabel('phase')
    plt.ylabel('BG %s (mag)'%(passband))
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    
    plt.gca().invert_yaxis()
    
    plt.minorticks_on()
    plt.tick_params('both', length=12, width=1., which='major', direction='in')	
    plt.tick_params('both', length=2, width=1, which='minor', direction='in')

    plt.tight_layout()

    
    plt.savefig(fig_path+"/freq_%d_%s.png"%(i,passband),format='png')
    
    plt.close()


    # Save the residuals at each iteration for future reference
    np.save(fig_path+"/residual_%d_%s.npy"%(i,passband),np.vstack([residual_clean,phase,y_model_best_freq]))
    np.save(fig_path+"/residual_not_cleaned_%d_%s.npy"%(i,passband),residual)

    # Compute the residual for the next iteration

    residual=y_original-y_model_all_freq
    
    Amp_original=A_jk
    
    n_components += 1


    previous_freq=params[f'freq{i}'].value
    idx_previous_freq=(frequencies>previous_freq-2)&(frequencies<previous_freq+2)
    previous_freq_window=frequencies[idx_previous_freq]
    power_freq_widow=psi_norm[idx_previous_freq]



# Extract parameters and save them to a csv file
Amp_all=np.abs([params[f"amp{i}"].value for i in range(n_components)])
phi_all=np.abs([params[f"phase{i}"].value for i in range(n_components)])
freq_all=np.abs([params[f"freq{i}"].value for i in range(n_components)])

prewhitening_results=pd.DataFrame(prewhitening_params, columns=["frequency","period_second","A_jk","sigma_A_jk", "phi_jk","sigma_phi_jk","std"])

prewhitening_results[f'amp_final_{passband}']=Amp_all
prewhitening_results[f'phi_final_{passband}']=phi_all
prewhitening_results[f'freq_final_{passband}']=freq_all


prewhitening_results.to_csv(f"prewhitening_{target_name}_{passband}_processed_t0.csv",index=None)
