import os
import numpy as np
import random
from scipy.io import wavfile
from python_speech_features import fbank, delta


def normalize_frames(m,epsilon=1e-12):
    return [(v - np.mean(v)) / max(np.std(v),epsilon) for v in m]


def get_fbank(signal, target_sample_rate):    
    filter_banks, energies = fbank(signal, samplerate=target_sample_rate, nfilt=40,nfft=int(target_sample_rate*0.025))
    filter_banks = normalize_frames(filter_banks)
    return np.array(filter_banks)


def read_wav(fname):
    fs, signal = wavfile.read(fname)
    return fs, signal


def get_train_batch(train_file, batch_size):
    batch_feats = []
    batch_labs = []

    for i in range(batch_size):
        s = random.choice(train_file)
        u = os.path.join(s, random.choice(os.listdir(s)))
        fs, signal = read_wav(u)
        # take two seconds
        if len(signal) < 2*fs:
            signal = np.hstack((signal,[0]*(2*fs-len(signal))))
        start_sample = random.randint(0,len(signal)-2*fs)
        signal = signal[start_sample:start_sample+2*fs]

        batch_feats.append(get_fbank(signal,fs))
        batch_labs.append(train_file.index(s))

    return batch_feats, batch_labs


def get_train_batch_somethingwrong(train_file):
    batch_feats = []
    batch_labs = []
    for s in random.sample(train_file,20):
        for u in [os.path.join(s, f) for f in random.sample(os.listdir(s),10)]:
            fs, signal = read_wav(u)
            # take two seconds
            if len(signal) < 2*fs:
                signal = np.hstack((signal,[0]*(2*fs-len(signal))))
            start_sample = random.randint(0,len(signal)-2*fs)
            signal = signal[start_sample:start_sample+2*fs]
        
            batch_feats.append(get_fbank(signal,fs))
            batch_labs.append(train_file.index(s))
    return np.array(batch_feats), np.array(batch_labs)


def get_test_batch(test_file, batch_size):
    enroll_feats = []
    enroll_labels = []
    test_feats = []
    test_labels = []
    for s in test_file:
        u_i = 0
        for u in [os.path.join(s, f) for f in random.sample(os.listdir(s),15)]:
            fs, signal = read_wav(u)
            # take two seconds
            if len(signal) < 2*fs:
                signal = np.hstack((signal,[0]*(2*fs-len(signal))))
            start_sample = random.randint(0,len(signal)-2*fs)
            signal = signal[start_sample:start_sample+2*fs]
        
            if u_i < 5:
                enroll_feats.append(get_fbank(signal,fs))
                enroll_labels.append(test_file.index(s))
            else:
                test_feats.append(get_fbank(signal,fs))
                test_labels.append(test_file.index(s))
            u_i += 1    

    for i in range(2*batch_size-15*len(test_file)):
        s = random.choice(test_file)
        u = os.path.join(s, random.choice(os.listdir(s)))
        fs, signal = read_wav(u)
        # take two seconds
        if len(signal) < 2*fs:
            signal = np.hstack((signal,[0]*(2*fs-len(signal))))
        start_sample = random.randint(0,len(signal)-2*fs)
        signal = signal[start_sample:start_sample+2*fs]

        test_feats.append(get_fbank(signal,fs))
        test_labels.append(test_file.index(s))

    return enroll_feats, enroll_labels, test_feats, test_labels


def enroll(embeddings, labels):
    enroll_list = []
    for i in range(np.max(labels)+1):
        enroll_list.append(np.mean(embeddings[np.where(np.array(labels)==i)[0],:], axis=0))
    
    return enroll_list


def speaker_identification(embeddings, enroll_list):
    predict_labels = []
    for emb in embeddings:
        dist = [np.sqrt(np.sum(np.square(emb-enroll))) for enroll in enroll_list]
        predict_labels.append(np.argmin(dist))
    return predict_labels


def format_time(time):
    """ It formats a datetime to print it
        Args:
            time: datetime
        Returns:
            a formatted string representing time
    """
    m, s = divmod(time, 60)
    h, m = divmod(m, 60)
    d, h = divmod(h, 24)
    return ('{:02d}d {:02d}h {:02d}m {:02d}s').format(int(d), int(h), int(m), int(s))


