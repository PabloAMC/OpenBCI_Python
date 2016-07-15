import csv
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal
from notch import apply_notch50
import numpy as np
#from scipy.optimize import leastsq
from scipy.optimize import curve_fit
#training_2016-7-14_19-19-18training_sample
my_data = np.genfromtxt('/home/vaio/PycharmProjects/OpenBCI_Python/example3.csv', delimiter=',')
#row is the row of the aim matrix
import pandas as pd
import math
import statsmodels.api as sm
import csv
from sklearn import svm
from sklearn.feature_selection import RFE
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
import cPickle

my_data[:,1] = apply_notch50(my_data[:,1],)
def copytomatrix(my_data):
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
    #please consult reference http://docs.scipy.org/doc/numpy/reference/generated/numpy.matrix.item.html
    #http://stackoverflow.com/questions/3582601/how-to-call-an-element-in-an-numpy-array
    #http://stackoverflow.com/questions/2220968/python-setting-an-element-of-a-numpy-matrix
    #print 'Len(my_data)= '+str(len(my_data))
    finalmatrix = np.zeros(( 4 * k,2 * size))
    while i < len(my_data):
        #print my_data[i,2]
        if my_data[i,2]==1:
            mini = np.argmin(my_data[i:(i + w - 1), 1]) + i
            finalmatrix[row,:]=my_data[np.max([0, (mini-size)]):np.min([datalen,(mini+size)]),1]
            classifier1[row]= 1
            row += 1
            if mini+2*size < datalen:
                finalmatrix[row,:]=my_data[mini:(mini+2*size),1]
                classifier1[row]= -1
                row += 1
            if mini-2*size > 0:
                finalmatrix[row,:]=my_data[(mini-2*size):mini,1]
                classifier1[row]= -1
                row += 1
            if mini + separation < datalen:
                finalmatrix[row,:]=my_data[(mini+separation-2*size):(mini+separation),1]
                classifier1[row]=-1
                row += 1
        i += 1
    l = 0
    for e in classifier1:
        if e == 0:
            l += 1
    classifier = np.zeros(4*k-l)
    matrix = np.zeros(( 4 * k -l,2 * size))
    for row in range(0,4*k-l):
        classifier [row] = classifier1[row]
        matrix[row,:]=-(finalmatrix[row,:]-max(finalmatrix[row,:]))
        row += 1
    return matrix, classifier
matrix, classifier = copytomatrix(my_data)
print matrix
print classifier
#a=plt.plot(matrix[123,:])
#plt.plot(classifier)
#plt.show(a)


def gaussian(list, mu, sigma):
    from math import e, sqrt, pi
    #print list
    gauss = []
    for x in list:
        gauss += [ 1. / (sqrt(2. * pi) * sigma) * e ** (-0.5 * (float(x - mu) / sigma) ** 2)]
        #print gauss
        #raw_input()

    return gauss


def obtainparameters(list):
    p0 = [50, 1e-3]
    length=len(list)
    xarray=np.zeros(len(list))
    l=0
    for e in range(0,len(list)):
        xarray[e]=e
    popt, pcov = curve_fit(gaussian, xarray, list, p0)
    #print 'done'
    #perr = np.sqrt(np.diag(pcov))

    #print 'Parameters:', popt
    #print 'Errors:', perr
    return popt

Xarray=np.zeros((len(matrix[:,0]),2))
for i in range(0,len(matrix[:,0])):
    try:
        Xarray[i,:]=obtainparameters(matrix[i,:])
    except:
        Xarray[i,:]=(50, 1000)

mu, sigma = Xarray[0,:]
xarray=np.zeros(100)
for e in range(0, 100):
    xarray[e] = e
classifier=classifier.astype(int)
print Xarray, classifier
#plt.plot(t,a,'r') # plotting t,a separately
#plt.plot(t,b,'b') # plotting t,b separately
#print [gaussian(xarray,mu, sigma, K)]
a=plt.plot(matrix[0,:],'r')
b=plt.plot([gaussian(xarray,mu, sigma)],'b')
plt.show()

# Definir SVM
classification = svm.SVC(decision_function_shape='ovo')

# Entrenar SVM
classification.fit(Xarray, classifier)
print len(Xarray[:,0])
succes=0
failure=0
for i in range(0,len(Xarray[:,0])):
    print classification.predict([Xarray[i,:]]), classifier[i]
    if classification.predict([Xarray[i,:]]) == classifier[i]:
         succes += 1
    else:
        failure += 1
print 'succes: '+str(succes)+'. failure: '+str(failure)

# Guardar clasificador
with open('/tmp/clf.pkl', 'wb') as fid:
    cPickle.dump(classification, fid)

