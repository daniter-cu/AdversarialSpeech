"""
Compute the inverse MFCC using Librosa.

Code adapted from code example provided from
Librosa: https://github.com/librosa/librosa/issues/424

"""

import librosa
import numpy as np
from IPython.lib.display import Audio
import matplotlib.pyplot as plt
import librosa.display

class InverseMfcc(object):

    def __init__(self, orig, sr):
        '''Pass in digitized waveform and native sampling rate'''
        self.orig = np.array(orig)
        self.recon = None
        self.mfcc = None
        self.sr = sr

        #### Default parameters ####
        self.hop_length=512
        self.num_mfcc=20
        self.include_deltas=False
        self.noise_scale = 0.0
        self.mfcc_noise = 0.0

        #### Don't change these for now ####
        self.n_mel_recon = 128 
        self.n_fft_recon = 2048

    def modify_parameters(self,
                          sr=None,
                          hop_length=None,
                          num_mfcc=None,
                          include_deltas=None,
                          n_mel_recon=None,
                          n_fft_recon=None,
                          noise_scale=None,
                          mfcc_noise=None):
        '''Modify any parameters passed in'''
    
        if (sr != None):
            self.sr = sr
        if (hop_length != None):
            self.hop_length = hop_length
        if (num_mfcc != None):
            self.num_mfcc = num_mfcc
        if (include_deltas != None):
            self.include_deltas = include_deltas
        if (n_mel_recon != None):
            self.n_mel_recon = n_mel_recon
        if (n_fft_recon != None):
            self.n_fft_recon = n_fft_recon
        if (noise_scale != None):
            self.noise_scale = noise_scale
        if (mfcc_noise != None):
            self.mfcc_noise = mfcc_noise

    def transform(self, mfcc=None):
        #### Can overwrite given MFCC ####
        if (mfcc is None):
            #### Get first order MFCCs ####
            mfccs = librosa.feature.mfcc(y=self.orig,
                                         sr=self.sr,
                                         n_mfcc=self.num_mfcc,
                                         hop_length=self.hop_length)

            #### If necessary, add higher-order mfccs ####
            if (self.include_deltas):
                delta_mfcc  = librosa.feature.delta(mfccs)
                delta2_mfcc = librosa.feature.delta(mfccs, order=2)
                mfccs = np.vstack([mfccs, delta_mfcc, delta2_mfcc])

        else:
            mfccs = mfcc

        self.mfcc = mfccs
        #### Now fully in ceptstral domain, start inverse#### 
        n_mfcc = mfccs.shape[0] #will vary if deltas included
        dctm = librosa.filters.dct(n_mfcc, self.n_mel_recon)
        mel_basis = librosa.filters.mel(self.sr, self.n_fft_recon)

        # Optional - Add additional noise
        noise = self.mfcc_noise*np.random.randn(self.mfcc.shape[0], self.mfcc.shape[1])
        self.mfcc += noise

        #TODO: Figure out if should modify this
        bin_scaling = 1.0 / np.maximum(0.01, np.sum(np.dot(mel_basis.T, mel_basis), axis=0))

        #Approx squared magnitude
        recon_stft = bin_scaling[:, np.newaxis] * np.dot(mel_basis.T,
                                                         self._invlogamplitude(np.dot(dctm.T, mfccs)))
        # Optional - Add noise to reconstruction
        noise = self.noise_scale*np.ones(recon_stft.shape)
        recon_stft += noise
        
        # Impose reconstructed magnitude on white noise STFT
        #excitation = np.random.randn(self.orig.shape[0])
        excitation = np.random.randn(self.hop_length*(mfccs.shape[1]-1))
        E = librosa.stft(excitation, hop_length=self.hop_length)
        self.recon = librosa.istft(E / np.abs(E) * np.sqrt(recon_stft))

    def play_original(self):
        return Audio(self.orig, rate=self.sr)

    def play_transformed(self):
        if (self.recon is None):
            print "No reconstructed waveform"
            return
        return Audio(self.recon, rate=self.sr)

    def get_mfcc(self):
        if (self.mfcc is None):
            print "No MFCC constructed"
            return
        return self.mfcc
         
    def _invlogamplitude(self, S):
        '''librosa.logamplitude is actually 10*log10, so invert that.'''
        return 10.0 ** (S / 10.0)

