import numpy as np


def create_datasets(signals, dataset_signals, dataset_labels, perm, no_train, no_val):

    no_val = int(0.85 * signals.shape[0])

    np.savez("eeg-seizure_train.npz",
             train_signals=dataset_signals[perm[:no_train]], train_labels=dataset_labels[perm[:no_train]])

    np.savez("eeg-seizure_val.npz",
             val_signals=dataset_signals[perm[no_train:no_val]], val_labels=dataset_labels[perm[no_train:no_val]])

    np.savez("eeg-seizure_test.npz",
             test_signals=dataset_signals[perm[no_val:]])
    np.savez("eeg-seizure_test_labels.npz",
             test_labels=dataset_labels[perm[no_val:]])

    print('Seizures in train:')

    print(np.sum(np.array(dataset_labels[perm[:no_train]]), axis=0))
    print('of')
    print(no_train)

    print('Seizures in test and val:')
    print(np.sum(np.array(dataset_labels[perm[no_train:]]), axis=0))
    print('of')
    print(dataset_labels.shape[0] - no_train)
