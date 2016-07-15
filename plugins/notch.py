import csv
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal



# 250 Hz is the sample rate of OpenBCI
def apply_notch50(x, fs_hz=250.0):
    # create the 50 Hz filter
    bp_stop_hz = np.array([49, 51.0])
    b, a = scipy.signal.butter(2, bp_stop_hz / (fs_hz / 2.0), 'bandstop')
    return scipy.signal.filtfilt(b, a, x)

if __name__ == "__main__":
    my_data = np.genfromtxt('../training_2016-7-12_11-35-26training sample.csv', delimiter=',')
    a=apply_notch50(my_data[:,1],)

    np.savetxt("foo.csv", a, delimiter=",")


    plt.plot(my_data[:,1])
    plt.plot(a)