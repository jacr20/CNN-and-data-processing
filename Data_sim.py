import numpy as np
import random
import matplotlib.pyplot as plt
%matplotlib inline

import wimprates
import numericalunits as nu
import blueice as bi
from laidbax import base_model
import pandas as pd

energies = np.linspace(0.01, 14, 100)
dr = wimprates.rate_wimp_std(energies, mw=500, sigma_nucleon=1e-45)/(1000*365)

plt.plot(energies, dr)
plt.xlabel("Recoil energy [keV]")
plt.ylabel("Rate [events per (keV kg day)]")
plt.title("$m_\chi = 500$ GeV/c${}^2$, $\sigma_\chi = 10^{-45}$ cm${}^2$")
plt.xlim(0, energies.max())
plt.ylim(0, None)

#m = bi.Model(base_model.config)
source_config = {
    'energy_distribution': (energies, dr),
    'name': 'WIMP',
    'label': 'WIMP particles',
    'recoil_type': 'nr',
    'color': 'black',
    # The amount of events to simulate to create the PDF
    # Depending on how important deep tails are for your model, you may want
    # to increase this from the default of 1e6.
    'n_events_for_pdf':1e7,
    # This tells blueice which settings are irrelevant for the source.
    # Since it's an NR source, this includes all the ER parameters.
    # It helps blueice's caching to know which settings cannot change the model.
    # However, if you just want the nominal models and/or don't care about a small
    # slowdown, this can be safely ignored.
    'extra_dont_hash_settings': base_model.nr_ignore_settings}

# Copy the old config
from copy import deepcopy
new_config = deepcopy(base_model.config)

# Simulate only the new WIMP particle
#new_config['sources'] = [source_config]

# If you want to keep the background models, and just replace the WIMPs by your model, instead do:
new_config['sources'][-1] = source_config
#This will take a moment the first time (or when you change the energy spectrum)
m2 = bi.Model(new_config)
print (type(m2))

print (m2.expected_events())
ER = m2.sources[0]
CNNS = m2.sources[1]
RN = m2.sources[2]
AC = m2.sources[3]
Wall = m2.sources[4]
Anon = m2.sources[5]
WIMP = m2.sources[6]

ER_fake = ER.simulate(500)
CNNS_fake = CNNS.simulate(0)
RN_fake = RN.simulate(0)
AC_fake = AC.simulate(0)
Wall_fake = Wall.simulate(0)
Anon_fake = Anon.simulate(0)
WIMP_fake = WIMP.simulate(500)

df_ER = pd.DataFrame.from_records(ER_fake)
df_CNNS = pd.DataFrame.from_records(CNNS_fake)
df_RN = pd.DataFrame.from_records(RN_fake)
df_AC = pd.DataFrame.from_records(AC_fake)
df_Wall = pd.DataFrame.from_records(Wall_fake)
df_Anon = pd.DataFrame.from_records(Anon_fake)
df_WIMP = pd.DataFrame.from_records(WIMP_fake)

df_ER['source'] = 0
df_CNNS['source'] = 1
df_RN['source'] = 2
df_AC['source'] = 3
df_Wall['source'] = 4
df_Anon['source'] = 5
df_WIMP['source'] = 6

df =  pd.concat([df_ER, df_CNNS, df_RN, df_AC, df_Wall, df_Anon, df_WIMP])

for i in range (0,7):
    print (df.loc[df.source == i, 'source'].count())

m2.show(df)

def plot_model(m2):
    for s in m2.sources:
        pdf = s._pdf_histogram
        events_per_bin = pdf * pdf.bin_volumes()
        events_per_bin.plot(cblabel='Expected events per bin')

        plt.title(s.name)
        #plt.yscale('log')
        plt.show()
        
plot_model(m2)

df_ER['x'] =np.sqrt(df_ER.r2)*np.cos(df_ER.theta)
df_ER['y'] =np.sqrt(df_ER.r2)*np.sin(df_ER.theta)
df_ER['t'] = [random.randint(0,400001) for k in df_ER.index ]
df_ER['instruction'] = [df_ER.index[0] +i for i in range(0,len(df_ER.index))] 
df_ER['recoil_type'] = 'ER'
df_ER['depth'] =abs(df_ER['z'])


fake_ER = df_ER.drop(['z','source','energy','r2','theta','p_photon_detected','p_electron_detected','electrons_detected','s1','s2','cs1','cs2','csratio','electron_lifetime','s1_photons_detected','s1_photoelectrons_produced'], axis=1)

fakedER = fake_ER[['instruction', 'recoil_type', 'x','y', 'depth', 'photons_produced', 'electrons_produced' ,'t']]
fakedER.columns=(['instruction', 'recoil_type', 'x','y', 'depth', 's1_photons', 's2_electrons' ,'t'])


print(fakedER.shape)
#fakedER.head(10)

df_CNNS['x'] =np.sqrt(df_CNNS.r2)*np.cos(df_CNNS.theta)
df_CNNS['y'] =np.sqrt(df_CNNS.r2)*np.sin(df_CNNS.theta)
df_CNNS['t'] = [random.randint(0,400001) for k in df_CNNS.index ]
df_CNNS['instruction'] = [df_CNNS.index[0] +i for i in range(0,len(df_CNNS.index))] 
df_CNNS['recoil_type'] = 'NR'
df_CNNS['depth'] =abs(df_CNNS['z'])


fake_CNNS = df_CNNS.drop(['z','source','energy','r2','theta','p_photon_detected','p_electron_detected','electrons_detected','s1','s2','cs1','cs2','csratio','electron_lifetime','s1_photons_detected','s1_photoelectrons_produced'], axis=1)

fakedCNNS = fake_CNNS[['instruction', 'recoil_type', 'x','y', 'depth', 'photons_produced', 'electrons_produced' ,'t']]
fakedCNNS.columns=(['instruction', 'recoil_type', 'x','y', 'depth', 's1_photons', 's2_electrons' ,'t'])

#fakedCNNS.head(10)
print(fakedCNNS.shape)


df_RN['x'] =np.sqrt(df_RN.r2)*np.cos(df_RN.theta)
df_RN['y'] =np.sqrt(df_RN.r2)*np.sin(df_RN.theta)
df_RN['t'] = [random.randint(0,400001) for k in df_RN.index ]
df_RN['instruction'] = [df_RN.index[0] +i for i in range(0,len(df_RN.index))] 
df_RN['recoil_type'] = 'NR'
df_RN['depth'] =abs(df_RN['z'])


fake_RN = df_RN.drop(['z','source','energy','r2','theta','p_photon_detected','p_electron_detected','electrons_detected','s1','s2','cs1','cs2','csratio','electron_lifetime','s1_photons_detected','s1_photoelectrons_produced'], axis=1)

fakedRN = fake_RN[['instruction', 'recoil_type', 'x','y', 'depth', 'photons_produced', 'electrons_produced' ,'t']]
fakedRN.columns=(['instruction', 'recoil_type', 'x','y', 'depth', 's1_photons', 's2_electrons' ,'t'])

#fakedRN.head(10)
print(fakedRN.shape)

df_AC['x'] =np.sqrt(df_AC.r2)*np.cos(df_AC.theta)
df_AC['y'] =np.sqrt(df_AC.r2)*np.sin(df_AC.theta)
df_AC['t'] = [random.randint(0,400001) for k in df_AC.index ]
df_AC['instruction'] = [df_AC.index[0] +i for i in range(0,len(df_AC.index))] 
df_AC['recoil_type'] = 'NR'
df_AC['depth'] =abs(df_AC['z'])


fake_AC = df_AC.drop(['z','source','energy','r2','theta','p_photon_detected','p_electron_detected','electrons_detected','s1','s2','cs1','cs2','csratio','electron_lifetime','s1_photons_detected','s1_photoelectrons_produced'], axis=1)

fakedAC = fake_AC[['instruction', 'recoil_type', 'x','y', 'depth', 'photons_produced', 'electrons_produced' ,'t']]
fakedAC.columns=(['instruction', 'recoil_type', 'x','y', 'depth', 's1_photons', 's2_electrons' ,'t'])

print(fakedAC.shape)
#fakedAC.head(10)

df_Wall['x'] =np.sqrt(df_Wall.r2)*np.cos(df_Wall.theta)
df_Wall['y'] =np.sqrt(df_Wall.r2)*np.sin(df_Wall.theta)
df_Wall['t'] = [random.randint(0,400001) for k in df_Wall.index ]
df_Wall['instruction'] = [df_Wall.index[0] +i for i in range(0,len(df_Wall.index))] 
df_Wall['recoil_type'] = 'NR'
df_Wall['depth'] =abs(df_Wall['z'])


fake_Wall = df_Wall.drop(['z','source','energy','r2','theta','p_photon_detected','p_electron_detected','electrons_detected','s1','s2','cs1','cs2','csratio','electron_lifetime','s1_photons_detected','s1_photoelectrons_produced'], axis=1)

fakedWall = fake_Wall[['instruction', 'recoil_type', 'x','y', 'depth', 'photons_produced', 'electrons_produced' ,'t']]
fakedWall.columns=(['instruction', 'recoil_type', 'x','y', 'depth', 's1_photons', 's2_electrons' ,'t'])

print(fake_Wall.shape)
#fakedWall.head(10)


df_Anon['x'] =np.sqrt(df_Anon.r2)*np.cos(df_Anon.theta)
df_Anon['y'] =np.sqrt(df_Anon.r2)*np.sin(df_Anon.theta)
df_Anon['t'] = [random.randint(0,400001) for k in df_Anon.index ]
df_Anon['instruction'] = [df_Anon.index[0] +i for i in range(0,len(df_Anon.index))] 
df_Anon['recoil_type'] = 'NR'
df_Anon['depth'] =abs(df_Anon['z'])


fake_Anon = df_Anon.drop(['z','source','energy','r2','theta','p_photon_detected','p_electron_detected','electrons_detected','s1','s2','cs1','cs2','csratio','electron_lifetime','s1_photons_detected','s1_photoelectrons_produced'], axis=1)

fakedAnon = fake_Anon[['instruction', 'recoil_type', 'x','y', 'depth', 'photons_produced', 'electrons_produced' ,'t']]
fakedAnon.columns=(['instruction', 'recoil_type', 'x','y', 'depth', 's1_photons', 's2_electrons' ,'t'])


print(fake_Anon.shape)
#fakedAnon.head(10)

df_WIMP['x'] =np.sqrt(df_WIMP.r2)*np.cos(df_WIMP.theta)
df_WIMP['y'] =np.sqrt(df_WIMP.r2)*np.sin(df_WIMP.theta)
df_WIMP['t'] = [random.randint(0,400001) for k in df_WIMP.index ]
df_WIMP['instruction'] = [df_WIMP.index[0] +i for i in range(0,len(df_WIMP.index))] 
df_WIMP['recoil_type'] = 'NR'
df_WIMP['depth'] =abs(df_WIMP['z'])


fake_WIMP = df_WIMP.drop(['z','source','energy','r2','theta','p_photon_detected','p_electron_detected','electrons_detected','s1','s2','cs1','cs2','csratio','electron_lifetime','s1_photons_detected','s1_photoelectrons_produced'], axis=1)

fakeWIMP = fake_WIMP[['instruction', 'recoil_type', 'x','y', 'depth', 'photons_produced', 'electrons_produced' ,'t']]
fakeWIMP.columns=(['instruction', 'recoil_type', 'x','y', 'depth', 's1_photons', 's2_electrons' ,'t'])
#print(fake_df.shape)
#fake_df.head(10)

print(fakeWIMP.shape)
#fakedWIMP.head(10)

fakedWIMP = fakeWIMP[:35000]
print (len(fakedWIMP))
fakedWIMP.head()

fakedWIMP.to_csv('/home/joel/Documents/XENON-ML/WIMPSIM.csv',index=False)



