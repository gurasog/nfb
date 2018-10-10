from pynfb.io.xml_ import xml_file_to_params
from pynfb.io.hdf5 import load_xml_str_from_hdf5_dataset
from pynfb.signals import DerivedSignal
from utils.load_results import load_data
import pylab as plt
import numpy as np
from time import sleep
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import h5py
from pynfb.signal_processing.decompositions import CSPDecomposition, ICADecomposition
from pynfb.signal_processing.filters import ButterFilter, ButterBandEnvelopeDetector, ExponentialSmoother, FilterStack
from mne.viz import plot_topomap
from pynfb.inlets.montage import Montage
from sklearn.linear_model import LogisticRegressionCV
from sklearn.ensemble import RandomForestClassifier
from utils.lsl_transformer import LSLTransformer
from scipy.signal import welch



h5_dataset = r'C:\Projects\nfblab\nfb\pynfb\results\ar-ntln_10-08_16-34-45\experiment_data.h5'
df, fs, channels, p_names = load_data(h5_dataset)
print(channels)
#channels = [ch for ch in channels if ch not in ['O1', 'O2', 'Oz', 'P8', 'P7']]
#indxs = [j for j, ch in enumerate(channels) if ch not in ['A1', 'A2', 'P4', 'CP4', 'FZ', 'Fc2']]
fs = int(fs)
#channels = channels[:24]
n_channels = len(channels)

preprocess = ButterFilter((5, 35), fs, n_channels)

df[channels] = preprocess.apply(df[channels])

df = df.iloc[fs*2:]


states_names = ['Legs', 'Right']

X = df.loc[df.block_name.isin(states_names), channels].values
y = (df.loc[df.block_name.isin(states_names)].block_name == states_names[1]).astype(int).values
print(y)

plt.plot(y)
plt.show()

band = (16, 23)
#band = (8, 15)
csp = CSPDecomposition(channels, fs, band, reg_coef=0.0001)
#csp = ICADecomposition(channels, fs, band)
np.random.seed()
csp.fit(X, y)

print(csp.scores)

#components = [0,6,19]
components = np.arange(len(channels))[[0,1,-2,-1]]
#components = [0, -1]
#components = [8]

def estimate_component(comp):
    fig, axes = plt.subplots(5)
    axes[0].plot(X.dot(csp.filters[:, comp]))
    axes[0].fill_between(np.arange(len(y)), -X.dot(csp.filters[:, comp]).max() * y, X.dot(csp.filters[:, comp]).max() * y, color='r', alpha=0.5)
    axes[1].plot(*welch(X.dot(csp.filters[:, comp]), fs, nfft=fs*2))
    axes[1].set_xlim(0,60)
    bands = [band]#[(k,k+6) for k in range(13, 22, 3)]#[(5,13), (13, 30)]
    main_filter = FilterStack([ButterFilter(b, fs, 1) for b in bands])
    smoother = ButterFilter((None, 0.2), fs, len(bands))

    X_comps = X.dot(csp.filters[:, [comp]])
    envs = smoother.apply(np.abs(main_filter.apply(X_comps)))

    classifier = LogisticRegressionCV()
    #classifier = RandomForestClassifier(max_depth=2)
    classifier.fit(envs, y)
    print(list(zip(bands,classifier.coef_[0])))

    print(comp, np.mean((classifier.predict(envs)==y)))
    axes[3].plot(classifier.predict_proba(envs)[:,1], label=states_names[1])
    axes[3].legend()
    axes[3].fill_between(np.arange(len(y)), classifier.predict(envs), alpha=0.3)
    axes[3].fill_between(np.arange(len(y)), y, alpha=0.3)
    axes[3].set_title('{}: {:.2f}'.format(comp, np.mean((classifier.predict(envs)==y))))

    plot_topomap(csp.topographies[:,  comp], Montage(channels).get_pos('EEG'), axes=axes[2], show=False)
    plot_topomap(csp.filters[:, comp], Montage(channels).get_pos('EEG'), axes=axes[4], show=False)
    plt.tight_layout()
    plt.show()

    return main_filter, smoother, classifier


#for comp in components: main_filter, smoother, classifier = estimate_component(comp)
x=-1

#x = int(input('Select component:\n'))
components = [x]
main_filter, smoother, classifier = estimate_component(x)


class TwoSatesClassifier(LSLTransformer):
    def __init__(self, *args, **kwargs):
        super(TwoSatesClassifier, self).__init__(*args, **kwargs)
        inlet_channels = [ch.upper() for ch in self.inlet.get_channels_labels()]
        recorded_channels = [ch.upper() for ch in channels]
        self.ch_indexes = [j for j, ch in enumerate(inlet_channels) if ch in recorded_channels]


    def transform(self, X):
        X = X[:, self.ch_indexes]
        X = preprocess.apply(X)
        X_comps = X.dot(csp.filters[:, components])
        envs = smoother.apply(np.abs(main_filter.apply(X_comps)))
        p = classifier.predict_proba(envs)[:, [1]]
        return p


transformer = TwoSatesClassifier('SmartBCI_Data', 'BCIState', ['BCI'])
while True:
    transformer.update()
    sleep(0.05)