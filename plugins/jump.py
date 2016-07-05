import plugin_interface as plugintypes
import numpy as np
from sklearn import svm
import uinput
import timeit
import datetime


class PluginCSVCollect(plugintypes.IPluginExtended):

	def __init__(self):
		rbf_svc = svm.SVC(kernel='rbf')
		clf = svm.SVC(C=0.5,gamma=1)
		clf.fit(arrayX, arrayY)
	
	
	
	'''
	def __init__(self, verbose=False)
	source: http://scikit-learn.org/stable/auto_examples/svm/plot_rbf_parameters.html#example-svm-plot-rbf-parameters-py
	source: https://github.com/OpenBCI/OpenBCI_Python/blob/master/plugins/csv_collect.py
	source: https://github.com/tuomasjjrasanen/python-uinput/blob/master/README.rst
	'''


	def activate(self):
	  
		print "Activated"

		events = (uinput.KEY_SPACE)
		device = uinput.Device(events)
		time.sleep(1)
		# arrayX and arrayY are to be recorded

	def deactivate(self):
		print "Deactivated"

	def show_help(self):
		print "Think of jumping in order to jump"

	def __call__(self, sample):
		clf.decision_function_shape = "ovr"
		dec = clf.decision_function(sample)

		'''here there is probably a mistake'''
		if dec == True:
			device = uinput.Device(events)

	    		
