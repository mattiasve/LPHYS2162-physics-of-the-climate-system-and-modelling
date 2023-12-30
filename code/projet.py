# %%
import numpy as np
import matplotlib.pyplot as plt
from numpy.random import normal, randn
from scipy.signal import argrelextrema
import os

if not os.path.exists('plots'):
    os.makedirs('plots')
    print(f"The folder '{'plots'}' has been created.")
else:
    print(f"The folder '{'plots'}' already exists.")

# -- Define all constants 
H  = 4500           # [m]
S0 = 35             # [psu]
tho_d = 219         # [years]
L = 8250e3          # [m]
sigmaw = 300e3      # [m]
V = L*H*sigmaw      # [m^3]
q = 3.27e19         # [m^3 years^-1]
alpha_s = 0.75e-3   # [psu^-1]
alpha_t = 0.17e-3   # [°C^-1]
theta = 20          # [°C]

# %%
###############################################
#### PART 1 : Consider F(t)= F = constant ####
##############################################

# -- 1.1 : Take F=3.24 m year-1. Show graphically that the system is bistable
F = 3.24  # [m year^-1]

# array of values of Delta S to to compute d Delta S / dt
DeltaS_val = np.arange(0, 6, 0.01)

# compute using the right-hand side of the equation
DS = (F*S0/H) - (DeltaS_val/tho_d) - (q*DeltaS_val*(alpha_s*DeltaS_val - alpha_t*theta)**2)/V

plt.figure()
plt.plot(DeltaS_val,DS)
plt.xlabel('$\Delta S$  [psu]')
plt.ylabel('$d \Delta S/dt$  [psu/years]')
plt.axhline(0, c='k', ls='--')
plt.grid(linestyle = ':')
plt.savefig('./plots/bistability_F324.png', dpi=500, bbox_inches='tight')
plt.savefig('./plots/bistability_F324.pdf' , bbox_inches='tight')
plt.show()

# Compute and pot the potential
pot = - np.cumsum(DS)*(DeltaS_val[1]-DeltaS_val[0])

# Find extremas
min_idx = argrelextrema(pot, np.less)
max_idx = argrelextrema(pot, np.greater)
ext_idx = np.concatenate((min_idx, max_idx), axis=1)
extremas_y = pot[ext_idx]
extremas_x = DeltaS_val[ext_idx]

plt.figure()
plt.ylabel('V  $[psu^2/years]$')
plt.xlabel('$ \Delta S$  [psu]')
plt.plot(DeltaS_val, pot)
plt.scatter(extremas_x, extremas_y, c='k')
plt.grid(linestyle = ':')
plt.savefig('./plots/potential_F324.png', dpi=500, bbox_inches='tight')
plt.savefig('./plots/potential_F324.pdf' , bbox_inches='tight')
plt.show()

print('Extremas of potential : ' + str(extremas_x))

# %%
# -- 1.2 : Cessi proposes F=2.3 m year-1. Is the system bistable in this case?
# Estimate numerically the critical value of F at which the system changes the number of stable solutions
Fvals = np.arange(2.3, 3.3, 0.1)
all_DS = np.zeros((len(DeltaS_val), len(Fvals)))

for i in range(len(Fvals)):
    all_DS[:,i] = (Fvals[i]*S0/H) - (DeltaS_val/tho_d) - (q*DeltaS_val*(alpha_s*DeltaS_val - alpha_t*theta)**2)/V
    
colors1 = plt.cm.plasma(np.linspace(0,0.9,len(Fvals)))

f = plt.figure()
for i in range(len(Fvals)):
    plt.plot(DeltaS_val, all_DS[:,i],  c = colors1[i])
    
plt.axhline(0, c='k', ls='--')
plt.xlabel('$\Delta S$  [psu]')
plt.ylabel('$d \Delta S/dt$  [psu/years]')
f.legend([str(round(F,2)) for F in Fvals], bbox_to_anchor=(1.05, 0.8))
plt.grid(linestyle=':')
plt.savefig('./plots/Fvals.png', dpi=500, bbox_inches='tight')
plt.savefig('./plots/Fvals.pdf', bbox_inches='tight')
plt.show()

flag = True
F = 2.3

while (flag):
    DeltaS_values = np.arange(0, 6, 0.001)
    DS = (F/H)*S0 - (DeltaS_values/tho_d) - (q*DeltaS_values*(alpha_s*DeltaS_values - alpha_t*theta)**2)/V
    Number = (np.diff(np.sign(DS)) != 0).sum()
    
    if(Number > 1):
        flag = False
    F += 0.0001
print("for F = {}, we cross 0 : {} times".format(round(F,10), Number))

# Plot for threshold value
threshold_DS = (F*S0/H) - (DeltaS_val/tho_d) - (q*DeltaS_val*(alpha_s*DeltaS_val - alpha_t*theta)**2)/V

plt.figure()
plt.plot(DeltaS_val, threshold_DS)
plt.axhline(0, c='k', ls='--')
plt.xlabel('$\Delta S$  [psu]')
plt.ylabel('$d \Delta S/dt$  [psu/years]')
plt.legend([str(2.5649)])
plt.grid(linestyle=':')
plt.savefig('./plots/Fthreshold.png', dpi=500, bbox_inches='tight')
plt.savefig('./plots/Fthreshold.pdf', bbox_inches='tight')
plt.show()

# Compute and plot the potential for threshold value
pot_thr = - np.cumsum(threshold_DS)*(DeltaS_val[1]-DeltaS_val[0])

# Find extremas
min_idx = argrelextrema(pot_thr, np.less)
max_idx = argrelextrema(pot_thr, np.greater)
ext_idx = np.concatenate((min_idx, max_idx), axis=1)
extremas_y_thrpot = pot_thr[ext_idx]
extremas_x_thrpot = DeltaS_val[ext_idx]

plt.figure()
plt.ylabel('V  $[psu^2/years]$')
plt.xlabel('$ \Delta S$  [psu]')
plt.plot(DeltaS_val, pot_thr)
plt.scatter(extremas_x_thrpot, extremas_y_thrpot, c='k')
plt.grid(linestyle = ':')
plt.savefig('./plots/potential_Fthreshold.png', dpi=500, bbox_inches='tight')
plt.savefig('./plots/potential_Fthreshold.pdf' , bbox_inches='tight')
plt.show()

print('Extremas of threshold potential : ' + str(extremas_x_thrpot))


# %%
# -- 1.3 Write a code to integrate in time the equation and show plots of 
# time series of the solutions to support what you found in points 1 and 2. You can take a timestep of 1 year.

# Forward-Euler for d(DeltaS)/dt
T   = 1000
dt  = 1
init_vals = np.arange(0, 6, 0.5)

# Define values of F
Flow = 2.3
Fup  = 3.4
Fthr = 2.5649

# Initialize figure
colors2 = plt.cm.plasma(np.linspace(0,0.9,len(init_vals)))
fig, axs = plt.subplots(1,3, figsize=(12, 5), sharey=True)

# -- Try with one stable solution : Flow
int_DS = np.zeros((T, len(init_vals)))
int_DS[0,:] = init_vals

for n in range(len(init_vals)):
    for i in range(1,T):
        rhs = (Flow*S0/H) - (int_DS[i-1,n]/tho_d) - (q*int_DS[i-1,n]*(alpha_s*int_DS[i-1,n] - alpha_t*theta)**2)/V
        int_DS[i,n] = int_DS[i-1,n] + rhs*dt
    axs[0].plot(np.arange(0,T,1) ,int_DS[:,n], c=colors2[n])

# -- Try with various stable solution : Fup
int_DS = np.zeros((T, len(init_vals)))
int_DS[0,:] = init_vals

for n in range(len(init_vals)):
    for i in range(1,T):
        rhs = (Fup*S0/H) - (int_DS[i-1,n]/tho_d) - (q*int_DS[i-1,n]*(alpha_s*int_DS[i-1,n] - alpha_t*theta)**2)/V
        int_DS[i,n] = int_DS[i-1,n] + rhs*dt
    axs[1].plot(np.arange(0,T,1), int_DS[:,n], c=colors2[n])

# -- Try with critical case : Fthreshold
int_DS = np.zeros((T, len(init_vals)))
int_DS[0,:] = init_vals

for n in range(len(init_vals)):
    for i in range(1,T):
        rhs = (Fthr*S0/H) - (int_DS[i-1,n]/tho_d) - (q*int_DS[i-1,n]*(alpha_s*int_DS[i-1,n] - alpha_t*theta)**2)/V
        int_DS[i,n] = int_DS[i-1,n] + rhs*dt
    axs[2].plot(np.arange(0,T,1), int_DS[:,n], c=colors2[n])

# Figure aesthetics
axs[0].grid(linestyle=':')
axs[1].grid(linestyle=':')
axs[2].grid(linestyle=':')
axs[0].set_ylabel('$\Delta S$  [psu]')
axs[0].set_xlabel('Time  [years]')
axs[1].set_xlabel('Time  [years]')
axs[2].set_xlabel('Time  [years]')
axs[0].text(0.97, 0.93, '(a)', transform=axs[0].transAxes, fontweight='bold', fontsize=12, ha='right')
axs[1].text(0.97, 0.93, '(b)', transform=axs[1].transAxes, fontweight='bold', fontsize=12, ha='right')
axs[2].text(0.97, 0.93, '(c)', transform=axs[2].transAxes, fontweight='bold', fontsize=12, ha='right')

fig.legend([str(round(init,3)) for init in init_vals], bbox_to_anchor=(1.06, 0.8))
plt.tight_layout()
plt.savefig('./plots/timeseries.png', dpi=500, bbox_inches='tight')
plt.savefig('./plots/timeseries.pdf', bbox_inches='tight')
plt.show()
# %%
########################################################
#### PART 2 : Consider F(t) as a stochastic forcing ####
########################################################

# -- 2.1 : Take F=3.24 m year-1 and plot the potential of the model
# already done : cfr 1.1

# %%
# -- 2.2 : Consider a stochastic forcing ; write a code and ̄perform simulations, discussing the results
# Forward-Euler for stochastic d(DeltaS)/dt

T   = 100000 # obligé d'être super long pour voir les jumps entre équilibres
dt  = 1
DS0 = 3      # à tester aussi
F =  3.25 # 2.3 # à tester aussi

DS_sto    = np.zeros(T)
DS_sto[0] = DS0

for i in range(1,T):
    sigmaF = 3*F*np.sqrt(dt) # tester différentes val aussi 
    rhs = (S0*F/H) - (DS_sto[i-1]/tho_d) - (q*DS_sto[i-1]*(alpha_s*DS_sto[i-1] - alpha_t*theta)**2)/V 
    DS_sto[i] = DS_sto[i-1] + rhs*dt + (sigmaF*randn()*S0/H)

f = plt.figure()
plt.xlabel('Time [years]')
plt.ylabel('$\Delta S$ [psu]')
plt.plot(DS_sto)
plt.savefig('./plots/stochastic_timeseries.png', dpi=500, bbox_inches='tight')
plt.savefig('./plots/stochastic_timeseries.pdf', bbox_inches='tight')
plt.show()
# %%
# --- Compute ensemble statistics for F=3.25 and sigma = 3F
T    = 100000 # obligé d'être super long pour voir les jumps entre équilibres
Nens = 100
dt   = 1
DS0  = 3      # à tester aussi
F    =  3.25 # 2.3 # à tester aussi

DS_sto_ens = np.zeros((T, Nens))
DS_sto_ens[0,:] = DS0

for n in range(Nens):
    for i in range(1,T):
        sigmaF = 3*F*np.sqrt(dt) # tester différentes val aussi 
        rhs = (S0*F/H) - (DS_sto_ens[i-1,n]/tho_d) - (q*DS_sto_ens[i-1,n]*(alpha_s*DS_sto_ens[i-1,n] - alpha_t*theta)**2)/V 
        DS_sto_ens[i,n] = DS_sto_ens[i-1,n] + rhs*dt + (sigmaF*randn()*S0/H)
# %%
plt.figure()
plt.xlabel('$\Delta S$ [psu]')
plt.ylabel('Counts')
plt.hist(DS_sto_ens.ravel(), bins=150)
plt.savefig('./plots/sto_base_ensemble_hist.png', dpi=500, bbox_inches='tight')
plt.savefig('./plots/sto_base_ensemble_hist.pdf', bbox_inches='tight')
plt.show()

# %%
# --- For F=3.25, test various stochastic forcings
T    = 100000 # obligé d'être super long pour voir les jumps entre équilibres
Nens = 100
dt   = 1
DS0  = 3      
F    =  3.25 

DS_sto_ens15 = np.zeros((T, Nens))
DS_sto_ens15[0,:] = DS0
for n in range(Nens):
    for i in range(1,T):
        sigmaF = 1.5*F*np.sqrt(dt) 
        rhs = (S0*F/H) - (DS_sto_ens15[i-1,n]/tho_d) - (q*DS_sto_ens15[i-1,n]*(alpha_s*DS_sto_ens15[i-1,n] - alpha_t*theta)**2)/V 
        DS_sto_ens15[i,n] = DS_sto_ens15[i-1,n] + rhs*dt + (sigmaF*randn()*S0/H)

#----------------------#
DS_sto_ens2 = np.zeros((T, Nens))
DS_sto_ens2[0,:] = DS0
for n in range(Nens):
    for i in range(1,T):
        sigmaF = 2*F*np.sqrt(dt) 
        rhs = (S0*F/H) - (DS_sto_ens2[i-1,n]/tho_d) - (q*DS_sto_ens2[i-1,n]*(alpha_s*DS_sto_ens2[i-1,n] - alpha_t*theta)**2)/V 
        DS_sto_ens2[i,n] = DS_sto_ens2[i-1,n] + rhs*dt + (sigmaF*randn()*S0/H)

#----------------------#
DS_sto_ens25 = np.zeros((T, Nens))
DS_sto_ens25[0,:] = DS0
for n in range(Nens):
    for i in range(1,T):
        sigmaF = 2.5*F*np.sqrt(dt) 
        rhs = (S0*F/H) - (DS_sto_ens25[i-1,n]/tho_d) - (q*DS_sto_ens25[i-1,n]*(alpha_s*DS_sto_ens25[i-1,n] - alpha_t*theta)**2)/V 
        DS_sto_ens25[i,n] = DS_sto_ens25[i-1,n] + rhs*dt + (sigmaF*randn()*S0/H)
plt.show()


# %%
# --- Subplot for report
from matplotlib.ticker import ScalarFormatter

fig, axs = plt.subplots(3,2, sharex='col', figsize=(8,7))
axs[0,0].plot(DS_sto_ens15[:,0]) ; axs[0,0].set_ylim([-0.2,6.4]) ; axs[0,0].set_ylabel('$\Delta S$ [psu]')
axs[1,0].plot(DS_sto_ens2 [:,2], color='orange') ; axs[1,0].set_ylim([-0.2,6.4]) ; axs[1,0].set_ylabel('$\Delta S$ [psu]')
axs[2,0].plot(DS_sto_ens25[:,0], color='green') ; axs[2,0].set_ylim([-0.2,6.4]) ; axs[2,0].set_xlabel('Time [years]') ; axs[2,0].set_ylabel('$\Delta S$ [psu]')

axs[0,1].hist(DS_sto_ens15.ravel(), bins=150)
axs[1,1].hist(DS_sto_ens2 .ravel(), bins=150, color='orange')
axs[2,1].hist(DS_sto_ens25.ravel(), bins=150, color='green') ; axs[2,1].set_xlabel('$\Delta S$ [psu]')

for ax in axs[:, 1]:
    ax.yaxis.get_major_formatter().set_powerlimits((0, 1))

plt.tight_layout()
# plt.savefig('./plots/sto_forcing_sensitivity.png', dpi=500, bbox_inches='tight')
# plt.savefig('./plots/sto_forcing_sensitivity.pdf', bbox_inches='tight')
plt.show()

# %%
# --- For the same stochastic forcing, change the value of F
T    = 100000 # obligé d'être super long pour voir les jumps entre équilibres
Nens = 100
dt   = 1
DS0  = 3  
    
F = 2.3
# stochasic solutions
DS_ens_F23 = np.zeros((T, Nens))
DS_ens_F23[0,:] = DS0
for n in range(Nens):
    for i in range(1,T):
        sigmaF = 3*F*np.sqrt(dt) 
        rhs = (S0*F/H) - (DS_ens_F23[i-1,n]/tho_d) - (q*DS_ens_F23[i-1,n]*(alpha_s*DS_ens_F23[i-1,n] - alpha_t*theta)**2)/V 
        DS_ens_F23[i,n] = DS_ens_F23[i-1,n] + rhs*dt + (sigmaF*randn()*S0/H)
# deterministic solution
# DS_det_F23 = np.zeros(T)
# DS_det_F23[0] = DS0
# for i in range(1,T): 
#     rhs = (S0*F/H) - (DS_det_F23[i-1]/tho_d) - (q*DS_det_F23[i-1]*(alpha_s*DS_det_F23[i-1] - alpha_t*theta)**2)/V 
#     DS_det_F23[i] = DS_det_F23[i-1] + rhs*dt
#--------------------------------------------#
F = 2.65
# stochasic solutions
DS_ens_Fc = np.zeros((T, Nens))
DS_ens_Fc[0,:] = DS0
for n in range(Nens):
    for i in range(1,T):
        sigmaF = 3*F*np.sqrt(dt) 
        rhs = (S0*F/H) - (DS_ens_Fc[i-1,n]/tho_d) - (q*DS_ens_Fc[i-1,n]*(alpha_s*DS_ens_Fc[i-1,n] - alpha_t*theta)**2)/V 
        DS_ens_Fc[i,n] = DS_ens_Fc[i-1,n] + rhs*dt + (sigmaF*randn()*S0/H)
# deterministic solution
# DS_det_Fc = np.zeros(T)
# DS_det_Fc[0] = DS0
# for i in range(1,T): 
#     rhs = (S0*F/H) - (DS_det_Fc[i-1]/tho_d) - (q*DS_det_Fc[i-1]*(alpha_s*DS_det_Fc[i-1] - alpha_t*theta)**2)/V 
#     DS_det_Fc[i] = DS_det_Fc[i-1] + rhs*dt
#--------------------------------------------#
F = 2.9
# stochasic solutions
DS_ens_F29 = np.zeros((T, Nens))
DS_ens_F29[0,:] = DS0
for n in range(Nens):
    for i in range(1,T):
        sigmaF = 3*F*np.sqrt(dt) 
        rhs = (S0*F/H) - (DS_ens_F29[i-1,n]/tho_d) - (q*DS_ens_F29[i-1,n]*(alpha_s*DS_ens_F29[i-1,n] - alpha_t*theta)**2)/V 
        DS_ens_F29[i,n] = DS_ens_F29[i-1,n] + rhs*dt + (sigmaF*randn()*S0/H)
# deterministic solution
# DS_det_F29 = np.zeros(T)
# DS_det_F29[0] = DS0
# for i in range(1,T): 
#     rhs = (S0*F/H) - (DS_det_F29[i-1]/tho_d) - (q*DS_det_F29[i-1]*(alpha_s*DS_det_F29[i-1] - alpha_t*theta)**2)/V 
#     DS_det_F29[i] = DS_det_F29[i-1] + rhs*dt

# %%
meanF23 = np.average(DS_ens_F23, axis=1)
meanFc  = np.average(DS_ens_Fc, axis=1)
meanF29 = np.average(DS_ens_F29, axis=1)

fig, axs = plt.subplots(3,2, figsize=(12,7.5), sharex='col')
axs[0,0].plot(DS_ens_F23, 'tab:blue', alpha=.33, lw=.5) ; axs[0,0].set_ylabel('$\Delta S$ [psu]')
# axs[0,0].plot(DS_det_F23, 'k')
axs[0,0].plot(meanF23, 'r')
axs[0,1].hist(DS_ens_F23.ravel(), bins=100, color='tab:blue')

axs[1,0].plot(DS_ens_Fc, color='orange', alpha=.33, lw=.5) ;  axs[1,0].set_ylabel('$\Delta S$ [psu]')
# axs[1,0].plot(DS_det_Fc, 'k')
axs[1,0].plot(meanFc, 'r')
axs[1,1].hist(DS_ens_Fc.ravel(), bins=100, color='orange')

axs[2,0].plot(DS_ens_F29, 'g', alpha=.33, lw=.5) ; axs[2,0].set_ylabel('$\Delta S$ [psu]')
# axs[2,0].plot(DS_det_F29, 'k')
axs[2,0].plot(meanF29, 'r')
axs[2,1].hist(DS_ens_F29.ravel(), bins=100, color='green')

axs[2,0].set_xlabel('Time [years]')
axs[2,1].set_xlabel('$\Delta S$ [psu]')

axs[0,0].set_ylim(-0.5, 6.5)
axs[1,0].set_ylim(-0.5, 6.5)
axs[2,0].set_ylim(-0.5, 6.5)

for ax in axs[:, 1]:
    ax.yaxis.get_major_formatter().set_powerlimits((0, 1))
    
plt.tight_layout()
# plt.savefig('./plots/sto_Fvals.png', dpi=500, bbox_inches='tight')
plt.savefig('./plots/sto_Fvals.svg', bbox_inches='tight')
plt.show()
# %%

