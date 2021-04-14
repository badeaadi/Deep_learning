import numpy as np
import librosa


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


def make_stft_dataset():

    # training_file = np.load('eeg-seizure_train.npz', allow_pickle=True)
    # train_signals = training_file['train_signals']
    # train_labels = training_file['train_labels']

    # val_file = np.load('eeg-seizure_val.npz', allow_pickle=True)
    # val_signals = val_file['val_signals']
    # val_labels = val_file['val_labels']

    test_file = np.load('eeg-seizure_test.npz', allow_pickle=True)
    test_signals = test_file['test_signals']

    samples = []
    for i in range(0, test_signals.shape[0]):

        sample = []
        for j in range(0, test_signals.shape[1]):
            y = librosa.stft(test_signals[i][j],
                             n_fft=32, hop_length=16).flatten()
            sample.append(y)

        if i % 100 == 0:
            print("{}th done with shape : {}".format(i, np.array(sample).shape))

        samples.append(sample)

    samples = np.array(samples)

    np.savez("eeg-seizure_test_stft.npz",
             test_signals=samples  # , val_labels=val_labels
             )


def make_mfcc_dataset():

    training_file = np.load('eeg-seizure_train.npz', allow_pickle=True)
    train_signals = training_file['train_signals']
    train_labels = training_file['train_labels']

    val_file = np.load('eeg-seizure_val.npz', allow_pickle=True)
    val_signals = val_file['val_signals']
    val_labels = val_file['val_labels']

    test_file = np.load('eeg-seizure_test.npz', allow_pickle=True)
    test_signals = test_file['test_signals']

    print("Processing train")

    samples = []
    for i in range(0, train_signals.shape[0]):

        sample = []
        for j in range(0, train_signals.shape[1]):
            y = librosa.feature.mfcc(train_signals[i][j],
                                     sr=256).flatten()
            sample.append(y)

        if i % 100 == 0:
            print("{}th done with shape : {}".format(i, np.array(sample).shape))

        samples.append(sample)

    samples = np.array(samples)

    np.savez("eeg-seizure_train_mfcc.npz",
             train_signals=samples, train_labels=train_labels
             )

    print("Processing validation")

    samples = []
    for i in range(0, val_signals.shape[0]):

        sample = []
        for j in range(0, val_signals.shape[1]):
            y = librosa.feature.mfcc(val_signals[i][j],
                                     sr=256).flatten()
            sample.append(y)

        if i % 100 == 0:
            print("{}th done with shape : {}".format(i, np.array(sample).shape))

        samples.append(sample)

    samples = np.array(samples)

    np.savez("eeg-seizure_val_mfcc.npz",
             val_signals=samples, val_labels=val_labels
             )

    print("Processing test")

    samples = []
    for i in range(0, test_signals.shape[0]):

        sample = []
        for j in range(0, test_signals.shape[1]):
            y = librosa.feature.mfcc(test_signals[i][j],
                                     sr=256).flatten()
            sample.append(y)

        if i % 100 == 0:
            print("{}th done with shape : {}".format(i, np.array(sample).shape))

        samples.append(sample)

    samples = np.array(samples)

    np.savez("eeg-seizure_test_mfcc.npz",
             test_signals=samples
             )


def make_mel_dataset():

    training_file = np.load('eeg-seizure_train.npz', allow_pickle=True)
    train_signals = training_file['train_signals']
    train_labels = training_file['train_labels']

    val_file = np.load('eeg-seizure_val.npz', allow_pickle=True)
    val_signals = val_file['val_signals']
    val_labels = val_file['val_labels']

    test_file = np.load('eeg-seizure_test.npz', allow_pickle=True)
    test_signals = test_file['test_signals']

    print("Processing train")

    samples = []
    for i in range(0, train_signals.shape[0]):

        sample = []
        for j in range(0, train_signals.shape[1]):
            y = librosa.feature.melspectrogram(train_signals[i][j],
                                               sr=256).flatten()
            sample.append(y)

        if i % 100 == 0:
            print("{}th done with shape : {}".format(i, np.array(sample).shape))

        samples.append(sample)

    samples = np.array(samples)

    np.savez("eeg-seizure_train_mel.npz",
             train_signals=samples, train_labels=train_labels
             )

    print("Processing validation")

    samples = []
    for i in range(0, val_signals.shape[0]):

        sample = []
        for j in range(0, val_signals.shape[1]):
            y = librosa.feature.melspectrogram(val_signals[i][j],
                                               sr=256).flatten()
            sample.append(y)

        if i % 100 == 0:
            print("{}th done with shape : {}".format(i, np.array(sample).shape))

        samples.append(sample)

    samples = np.array(samples)

    np.savez("eeg-seizure_val_mel.npz",
             val_signals=samples, val_labels=val_labels
             )

    print("Processing test")

    samples = []
    for i in range(0, test_signals.shape[0]):

        sample = []
        for j in range(0, test_signals.shape[1]):
            y = librosa.feature.melspectrogram(test_signals[i][j],
                                               sr=256).flatten()
            sample.append(y)

        if i % 100 == 0:
            print("{}th done with shape : {}".format(i, np.array(sample).shape))

        samples.append(sample)

    samples = np.array(samples)

    np.savez("eeg-seizure_test_mel.npz",
             test_signals=samples
             )


def make_cwt_dataset():

    training_file = np.load('eeg-seizure_train.npz', allow_pickle=True)
    train_signals = training_file['train_signals']
    train_labels = training_file['train_labels']

    val_file = np.load('eeg-seizure_val.npz', allow_pickle=True)
    val_signals = val_file['val_signals']
    val_labels = val_file['val_labels']

    test_file = np.load('eeg-seizure_test.npz', allow_pickle=True)
    test_signals = test_file['test_signals']

    print("Processing train")

    samples = []
    for i in range(0, train_signals.shape[0]):

        sample = []
        for j in range(0, train_signals.shape[1]):
            y = librosa.feature.melspectrogram(train_signals[i][j],
                                               sr=256).flatten()
            sample.append(y)

        if i % 100 == 0:
            print("{}th done with shape : {}".format(i, np.array(sample).shape))

        samples.append(sample)

    samples = np.array(samples)

    np.savez("eeg-seizure_train_mel.npz",
             train_signals=samples, train_labels=train_labels
             )

    print("Processing validation")

    samples = []
    for i in range(0, val_signals.shape[0]):

        sample = []
        for j in range(0, val_signals.shape[1]):
            y = librosa.feature.melspectrogram(val_signals[i][j],
                                               sr=256).flatten()
            sample.append(y)

        if i % 100 == 0:
            print("{}th done with shape : {}".format(i, np.array(sample).shape))

        samples.append(sample)

    samples = np.array(samples)

    np.savez("eeg-seizure_val_mel.npz",
             val_signals=samples, val_labels=val_labels
             )

    print("Processing test")

    samples = []
    for i in range(0, test_signals.shape[0]):

        sample = []
        for j in range(0, test_signals.shape[1]):
            y = librosa.feature.melspectrogram(test_signals[i][j],
                                               sr=256).flatten()
            sample.append(y)

        if i % 100 == 0:
            print("{}th done with shape : {}".format(i, np.array(sample).shape))

        samples.append(sample)

    samples = np.array(samples)

    np.savez("eeg-seizure_test_mel.npz",
             test_signals=samples
             )


make_mel_dataset()
