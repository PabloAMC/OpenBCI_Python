
import csv
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal
from notch import apply_notch50
import numpy as np
#from scipy.optimize import leastsq
from scipy.optimize import curve_fit
#training_2016-7-14_19-19-18training_sample
#row is the row of the aim Xdata
import pandas as pd
import math
import statsmodels.api as sm
import csv
from sklearn import svm, tree
from sklearn.feature_selection import RFE
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.externals.six import StringIO
import pydot


import cPickle
my_data = np.genfromtxt('/home/vaio/PycharmProjects/OpenBCI_Python/example3.csv', delimiter=',')

my_data[:,1] = apply_notch50(my_data[:,1],)
def copytoXdata(my_data):
    #use odd number for w. Otherwise trouble with int(w/2)
    w = 300
    size = 50
    datalen = len(my_data[:,0])
    i = 0
    k=0
    row = 0
    separation = 500
    for e in my_data[:,2]:
        if e == 1:
            k += 1
    print 'k= ' +str(k)
    classifier1 = np.zeros(4*k)
    #please consult reference http://docs.scipy.org/doc/numpy/reference/generated/numpy.Xdata.item.html
    #http://stackoverflow.com/questions/3582601/how-to-call-an-element-in-an-numpy-array
    #http://stackoverflow.com/questions/2220968/python-setting-an-element-of-a-numpy-Xdata
    #print 'Len(my_data)= '+str(len(my_data))
    finalXdata = np.zeros(( 4 * k,2 * size))
    while i < len(my_data):
        #print my_data[i,2]
        if my_data[i,2]==1:
            mini = np.argmin(my_data[i:(i + w - 1), 1]) + i
            finalXdata[row,:]=my_data[np.max([0, (mini-size)]):np.min([datalen,(mini+size)]),1]
            classifier1[row]= 1
            row += 1
            if mini+2*size < datalen:
                finalXdata[row,:]=my_data[mini:(mini+2*size),1]
                classifier1[row]= -1
                row += 1
            if mini-2*size > 0:
                finalXdata[row,:]=my_data[(mini-2*size):mini,1]
                classifier1[row]= -1
                row += 1
            if mini + separation < datalen:
                finalXdata[row,:]=my_data[(mini+separation-2*size):(mini+separation),1]
                classifier1[row]=-1
                row += 1
        i += 1
    l = 0
    for e in classifier1:
        if e == 0:
            l += 1
    classifier = np.zeros(4*k-l)
    Xdata = np.zeros(( 4 * k -l,2 * size))
    for row in range(0,4*k-l):
        classifier [row] = classifier1[row]
        Xdata[row,:]=(finalXdata[row,:]-np.percentile(finalXdata[row,:], 95))
        row += 1
    return Xdata, classifier
#Xdata, classifier = copytoXdata(my_data)
#print Xdata
#print classifier
#a=plt.plot(Xdata[123,:])
#plt.plot(classifier)
#plt.show(a)

'''
def gaussian(list, mu, sigma):
    from math import e, sqrt, pi
    #print list
    gauss = []
    for x in list:
        gauss += [ 1. / (sqrt(2. * pi) * sigma) * e ** (-0.5 * (float(x - mu) / sigma) ** 2)]
        #print gauss
        #raw_input()
'''
def gaussian(list, A, mu, sigma):
    from math import e, sqrt, pi
    # print list
    gauss = []
    for x in list:
        gauss += [np.e ** (-0.5 * (float(x - mu) / sigma) ** 2)]
        # print gauss
        # raw_input()

    return  A * np.array(gauss)

def polinomial(list, a0, a1, a2, a3):
    from math import e, sqrt, pi
    # print list
    gauss = []
    for x in list:
        gauss += [a0+a1*x+a2*x*x+a3*x*x*x]
        # print gauss
        # raw_input()

    return np.array(gauss)

def gaussianFit(list):
    signal = np.array(list)
    #  get initial values for offset, amplitude, center, sigma2
    #offset0 = np.median(signal)
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
    popt, pcov = curve_fit(gaussian, xrange, signal, p0 = p0,
                    bounds=([-np.inf, 0.0, 1e-12],
                           [np.inf, len(signal), np.inf]))
    gaussian_fit = gaussian(xrange, popt[0], popt[1], popt[2])
    c=[abs(i - j) for i, j in zip(gaussian_fit, signal)]
    return popt
    #return np.concatenate([popt, [sum(c)]])


def polinomialFit(list):
    shapearray=3
    signal = np.array(list)
    popt, residuals, rank, singular_values, rcond=np.polyfit(np.array(range(len(signal))), signal, shapearray, full=True)
    print popt + residuals
    return popt + residuals


def obtainparameters(x):
    return polinomialFit(x)

if __name__ == "__main__":
    # load data
    #training_2016-7-14_19-19-18training_sample
    my_data = np.genfromtxt('/home/vaio/PycharmProjects/OpenBCI_Python/example3.csv', delimiter=',')
    my_data[:,1] = apply_notch50(my_data[:,1])
    Xdata, classifier = copytoXdata(my_data)
    # print some example
    #plt.plot(Xdata[4,:])
    #plt.show()

    # before fitting, check the gaussian function
    #plt.plot(gaussian(np.linspace(-3, 3, 100), 1, 0, 1))

    # Check some fits
    shapearray=np.shape(obtainparameters(Xdata[0,:]))[0]
    #print shapearray, 'waka'
    XparameterArray = np.zeros((len(Xdata[:, 0]), shapearray))
    for i in range(0, len(Xdata[:, 0])):
        try:
            XparameterArray[i, :] = obtainparameters(Xdata[i, :])
        except:
            print "Caspita! Ha fallado"
            XparameterArray[i, :] = 0.1

    p0 = XparameterArray[0, :]
    xrange = range(0,100)
    # plt.plot(t,a,'r') # plotting t,a separately
    # plt.plot(t,b,'b') # plotting t,b separately
    # print [gaussian(xrange,mu, sigma, K)]
    #plt.plot(Xdata[0, :], 'r')
    #plt.plot([p0[0]+p0[1]*x+p0[2]*x*x+p0[3]*x*x*x for x in xrange], 'b')
    #plt.plot(gaussian(xrange,p0[0],p0[1],p0[2]))
    #plt.show()

    classifier = classifier.astype(int)
    print XparameterArray, classifier

    # Definir SVM
    #classification = svm.SVC(decision_function_shape='ovo')
    classification = tree.DecisionTreeClassifier(max_depth=2)

    # Entrenar SVM
    classification.fit(XparameterArray, classifier)
    success = np.sum(classification.predict(XparameterArray) == classifier)
    failure = XparameterArray.shape[0] - success
    print 'success: ' + str(success) + '. failure: ' + str(failure)

    # execute in linux console:
    # cd /tmp/ && dot -Tpdf tree.dot -o tree.pdf
    with open("/tmp/tree.dot", 'w') as f:
        f = tree.export_graphviz(classification, out_file=f)

    # Guardar clasificador
    with open('/tmp/clf.pkl', 'wb') as fid:
        cPickle.dump(classification, fid)


