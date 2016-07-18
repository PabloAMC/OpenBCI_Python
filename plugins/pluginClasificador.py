# -*- coding: utf-8 -*-
import plugin_interface as plugintypes
import collections
import cPickle
import numpy as np
from  scipy.signal import butter, lfilter
from scipy.optimize import curve_fit
import uinput

""" WARNING: EXECUTE WITH ROOT PRIVILEGES OR DIE!!"""

CHANNEL_QUEUE_SIZE = 100
USE_CHANNEL = 2

class PluginClasificador(plugintypes.IPluginExtended):
    def __init__(self):
        super(PluginClasificador, self).__init__()
        self.dev = uinput.Device([uinput.KEY_SPACE])
        self.channel_data = collections.deque(maxlen=CHANNEL_QUEUE_SIZE)
        self.classifier = None
        # assume sampling freq = 250 Hz and electric noise of 50 Hz
        self.b, self.a = butter(2, np.array([49, 51.0]) / (250.0 / 2.0), 'bandstop')
        filter_size = max(len(self.a),len(self.b))-1
        self.zi = np.repeat(0.0, filter_size)
        self.buffer=np.zeros(100)


    def activate(self):
        if self.args:
            # Load classifier from classifier_file
            with open(self.args[0], 'rb') as fid:
               self.classifier = cPickle.load(fid)
               print "Successfully loaded classifier from " + str(self.args[0])
        else:
            print "Could not register classifier. Valid filename must be specified."

    def deactivate(self):
        print "Deactivating movement detector"

    def __gaussian(self,list, A, mu, sigma):
        from math import e, sqrt, pi
        # print list
        gauss = []
        for x in list:
            gauss += [np.e ** (-0.5 * (float(x - mu) / sigma) ** 2)]
            # print gauss
            # raw_input()

        return A * np.array(gauss)

    def __gaussianFit(self, list):
        signal = np.array(list)
        signal = signal - np.percentile(signal, 95)
        #  get initial values for offset, amplitude, center, sigma2
        # offset0 = np.median(signal)
        amplitude0 = np.min(signal) - np.median(signal)
        mu0 = np.argmin(signal)
        # find crossing of the function with min/e for estimating the sd of the gaussian
        if (amplitude0 > 0):
            crossings = np.where(signal[mu0:] < amplitude0 / np.e)
        else:
            crossings = np.where(signal[mu0:] > amplitude0 / np.e)
        if len(signal[mu0:][crossings]) > 0:
            var0 = np.min(crossings) ** 2
        else:
            # No crossings... use some default small value
            var0 = 1.0
        p0 = [amplitude0, mu0, var0]
        #print p0
        xrange = np.array(range(len(signal)))
        try:
            popt, pcov = curve_fit(self.__gaussian, xrange, signal, p0=p0,
                                   bounds=([-np.inf, 0.0, 1e-12],
                                           [np.inf, len(signal), np.inf]))
        except:
            popt = np.array([0.0, 0.0, 1e-12])
        #print popt
        #print "===="
        gaussian_fit = self.__gaussian(xrange, popt[0], popt[1], popt[2])
        c = [abs(i - j) for i, j in zip(gaussian_fit, signal)]
        return np.array([popt[0], popt[2]])
        #return np.concatenate([popt, [sum(c)]])

    def __polinomialFit(self,list):
        shapearray = 3
        signal = np.array(list)
        popt, residuals, rank, singular_values, rcond = np.polyfit(np.array(range(len(signal))), signal, shapearray,
                                                                   full=True)
        return popt + residuals

    def __getFeatures(self, x):
        return self.__polinomialFit(x)

    def __call__(self, sample): #al ejecutarlo
        # Add samples to the FIFO queues
        self.channel_data.append(sample.channel_data[USE_CHANNEL - 1])
        # If buffers are full, then send to classifier
        if len(self.channel_data) is self.channel_data.maxlen:
            # Get features from buffers
            channel_data_array = np.array(self.channel_data)
            channel_data_array, self.zi = lfilter(self.b, self.a,
                                                    channel_data_array,
                                                    zi = self.zi)
            features = self.__getFeatures(channel_data_array)
            
            # Obtain class from SVM
            prediction  = self.classifier.predict(features.reshape(1, -1))[0]            
            if prediction == 1:
                # do things
                print "Salto Detectado!!!   "
                self.dev.emit_click(uinput.KEY_SPACE)


