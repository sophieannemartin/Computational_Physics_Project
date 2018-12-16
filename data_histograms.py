"""
INITIAL REPRESENTATION OF THE DATA FROM CSV FILE & PLOTTING HISTOGRAMS
21/11/18
SOPHIE MARTIN
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import define_functions as funcs

func = funcs.DecayFunction()
ts, sigmas = func.import_data()

# Plot histogram of the times, error is calculated using root(N)
time_freq, time_edges = np.histogram(ts, bins=300, normed=True)
time_centers = 0.5*(time_edges[1:]+time_edges[:-1])
widths_time= time_edges[1:]-time_edges[:-1]
error_time = np.sqrt(time_freq)/max(time_freq)

# Plot histogram for the uncertainty spread
sig_freq, sig_edges = np.histogram(sigmas, bins=300, normed=True)
sig_centers = 0.5*(sig_edges[1:]+sig_edges[:-1])
widths_sig = sig_edges[1:]-sig_edges[:-1]
error_sig = np.sqrt(sig_freq)

# p0 is the initial guess for the scipy fitting coefficients (t) for fm
p0_f = [1, 1]
coeff_fm, var_matrix_fm = curve_fit(func.fm_function, time_centers, time_freq, p0=p0_f)
hist_fit_fm = func.fm_function(time_centers, *coeff_fm)


# Plot distributions and gaussian/fm fit to the time spread
fig, (ax1, ax2) = plt.subplots(2,1, figsize=(8,15))
fig.subplots_adjust(hspace=0.5)
ax1.bar(time_centers, time_freq, width=widths_time, color='orange')
ax1.set_ylabel('Number of entries',  fontsize=15)
ax1.set_xlabel('Time ($p$s)',  fontsize=15)
ax1.set_title('Histogram of times measured')
ax1.grid()

ax2.bar(sig_centers, sig_freq, width=widths_sig, color='blue')
ax2.set_ylabel('Number of entries',  fontsize=15)
ax2.set_xlabel('$\sigma$ ($p$s)',  fontsize=15)
ax2.set_title('Histogram of errors on times')
ax2.grid()

plt.figure()
plt.plot(ts, sigmas, '.')
plt.title('No correlation between t and sigma')
plt.grid()

plt.figure()
plt.bar(time_centers, time_freq, width=widths_time, color='orange',label='data')
plt.plot(time_centers, hist_fit_fm, label='F$^m$(t) Fit', lw=1, color='purple')
plt.ylabel('Number of entries', fontsize=15)
plt.xlabel('Time ($p$s)', fontsize=15)
plt.grid()
plt.title('Histogram of times with F$^m$(t) scipy fit')
plt.legend(prop={'size': 14})
plt.show()

print('Fm tau scipy fit: ', coeff_fm[1], 'Fixed resolution: ', coeff_fm[0])