import matplotlib.pyplot as plt
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

from sklearn.model_selection import ShuffleSplit, cross_val_score, cross_val_predict
from sklearn.metrics import confusion_matrix
from sklearn.pipeline import Pipeline

from mne import Epochs, events_from_annotations, pick_types
from mne.channels import make_standard_montage
from mne.datasets import eegbci
from mne.decoding import CSP
from mne.io import concatenate_raws, read_raw_edf, read_raw_fif

# #############################################################################
# # Set parameters and read data

# avoid classification of evoked responses by using epochs that start 1s after
# cue onset.
tmin, tmax = -1.0, 4.0 
event_id = dict(hands=2, feet=3)
#subject = 1
#runs = [6,10,14]  # motor imagery: hands vs feet # COMO Q ELE SABE QUE ESSES DADOS SÃO DISSO? 
# ILUMINA SENHOR

#raw_fnames = eegbci.load_data(subject, runs) # Baixa os dados em formato "cru"
#raw = concatenate_raws([read_raw_edf(f, preload=True) for f in raw_fnames]) #junta os dados em um só "DF"

matrizes = list()

for individual in range(1,110):

    for time in range(5):
        raw = read_raw_fif(f"individuals\S{individual}R6-10-14_eeg.fif", preload=True)

        eegbci.standardize(raw)
        montage = make_standard_montage("standard_1005") 
        raw.set_montage(montage)

        raw.filter(7.0, 30.0, fir_design="firwin", skip_by_annotation="edge")

        events, _ = events_from_annotations(raw, event_id=dict(T1=2, T2=3))

        picks = pick_types(raw.info, meg=False, eeg=True, stim=False, eog=False, exclude="bads")

        # Read epochs (train will be done only between 1 and 2s)
        # Testing will be done with a running classifier
        epochs = Epochs(
            raw,
            events,
            event_id,
            tmin,
            tmax,
            proj=True,
            picks=picks,
            baseline=None,
            preload=True,
        )
        epochs_train = epochs.copy().crop(tmin=1.0, tmax=2.0)
        labels = epochs.events[:, -1] - 2

        #### PARTE 2 

        # Define a monte-carlo cross-validation generator (reduce variance):
        scores = []
        epochs_data = epochs.get_data()
        epochs_data_train = epochs_train.get_data()
        cv = ShuffleSplit(10, test_size=0.2)
        cv_split = cv.split(epochs_data_train)

        # Assemble a classifier
        lda = KNeighborsClassifier()
        csp = CSP(n_components=4, reg=None, log=True, norm_trace=False)

        # Use scikit-learn Pipeline with cross_val_score function
        clf = Pipeline([("CSP", csp), ("KNN", lda)])
        scores = cross_val_score(clf, epochs_data_train, labels, cv=cv, n_jobs=None)
        predicts = cross_val_predict(clf, epochs_data_train, labels, n_jobs=None)

        # Printing the results
        class_balance = np.mean(labels == labels[0])
        class_balance = max(class_balance, 1.0 - class_balance)

        matrizes.append(confusion_matrix(labels, predicts))
