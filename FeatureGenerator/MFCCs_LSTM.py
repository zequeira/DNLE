"""
This code generates a *.cvs datasets with containing different spectral and chroma features.
The audio files are read from "audio_files" and the *.csv dataset is stored in "mfcc_data".

As the number of audio files is large (1668), as well as the resulting *.csv file, then a
High-Performance-Computing-Cluster (HPC-Cluster) was used, employing parallelization.
The file "hpc_job.sh" contains the code for executing this file in an HPC-Cluster.
"""
import numpy as np
import pandas as pd
from itertools import repeat
import sklearn.preprocessing as pp
from multiprocessing import Pool
import multiprocessing as mp
from glob import glob
import librosa


def calculate_mfcc(file, i):
    print(str(mp.current_process())+' and '+file)

    y, sr = librosa.load(file, sr=None)

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=i, dct_type=2, norm='ortho')
    # mfcc_delta = librosa.feature.delta(mfcc, order=1)
    mfcc_delta2 = librosa.feature.delta(mfcc, order=2)

    spectral_center = librosa.feature.spectral_centroid(y=y, sr=sr)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    chroma_cens = librosa.feature.chroma_cens(y=y, sr=sr)

    feature_set = np.vstack((mfcc, mfcc_delta2, spectral_center, chroma, chroma_cens))

    feature_set = np.array(feature_set)
    feature_set = feature_set.transpose()

    # Normalization
    scaler = pp.StandardScaler()
    feature_set_Norm = scaler.fit_transform(feature_set)

    str_file = file.split('/')[-1]
    # if executed on Windows
    # str_file = file.split('\\')[-1]
    label = np.full((np.shape(feature_set_Norm)[0], 1), str_file.split(' _ ')[0])
    label_level = np.full((np.shape(feature_set_Norm)[0], 1), str_file.split(' _ ')[3])

    if label[0] == 'quiet':
        label_g = 'Quiet'
    elif label[0] == 'TV' or label[0] == 'Music' or label[0] == 'Radio':
        label_g = 'Melodic'
    else:
        label_g = 'Mechanic'
    label_group = np.full((np.shape(feature_set_Norm)[0], 1), label_g)

    student = np.full((np.shape(feature_set_Norm)[0], 1), str_file.split(' _ ')[1])
    file_name = np.full((np.shape(feature_set_Norm)[0], 1), str_file.split('.wav')[0])
    feature_set_Norm = np.hstack((feature_set_Norm, label, label_group, label_level, file_name, student))
    return feature_set_Norm


if __name__ == '__main__':
    audio_path = '../audio_files/*/'
    mfcc_data = '../mfcc_data/'
    print("Number of processors: ", mp.cpu_count())

    # i controls the number of computed MFCC coefficients
    for i in range(20, 21, 1):
        P = Pool(mp.cpu_count())

        coefficients = P.starmap(calculate_mfcc, zip(glob(audio_path+'/*.wav'), repeat(i)))

        features = np.vstack(coefficients)

        P.close()

        mfcc_headers = ['MFCC{:01d}'.format(j) for j in range(1, i+1, 1)]
        delta_headers = ['Delta{:01d}'.format(j) for j in range(1, i+1, 1)]
        chroma_headers = ['Chroma{:01d}'.format(j) for j in range(1, 12+1, 1)]
        chromaCens_headers = ['ChromaCENS{:01d}'.format(j) for j in range(1, 12+1, 1)]

        headers = mfcc_headers + delta_headers + ['SpectralCent'] + chroma_headers + chromaCens_headers + ['LABEL', 'LABEL_GROUP', 'LABEL_LEVEL', 'FILE', 'STUDENT']

        featuresDF = pd.DataFrame(data=features, columns=headers)

        featuresDF.to_csv(mfcc_data+'featuresNormalized_MFCC_Extended{:01d}.csv'.format(i), sep=';', float_format='%.3f')