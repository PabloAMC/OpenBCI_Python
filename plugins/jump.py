import plugin_interface as plugintypes
import numpy as np
from sklearn import svm
import uinput
import timeit
import datetime


class PluginCSVCollect(plugintypes.IPluginExtended):
		def __init__(self):
'''			now = datetime.datetime.now()
			self.time_stamp = '%d-%d-%d_%d-%d-%d'%(now.year,now.month,now.day,now.hour,now.minute,now.second)
			self.file_name = self.time_stamp
			self.start_time = timeit.default_timer()
'''
'''
	def __init__(self, verbose=False):
    source: http://scikit-learn.org/stable/auto_examples/svm/plot_rbf_parameters.html#example-svm-plot-rbf-parameters-py
    source: https://github.com/OpenBCI/OpenBCI_Python/blob/master/plugins/csv_collect.py
    source: https://github.com/tuomasjjrasanen/python-uinput/blob/master/README.rst

'''


	def activate(self):
	  
		print "Activated"
'''		rbf_svc = svm.SVC(kernel='rbf')
		clf = svm.SVC(C=0.5,gamma=1)
		clf.fit(arrayX, arrayY)
		# arrayX and arrayY are to be recorded
'''
		
	def deactivate(self):
		print "Deactivated"

	def show_help(self):
		print "Think of jumping in order to jump"

	def __call__(self, sample):
'''		clf.decision_function_shape = "ovr"
		dec = clf.decision_function(sample)'''

'''here there is probably a mistake'''
		for i in range (0,900000000):
			if i%10000000:
				with uinput.Device([uinput.KEY_SPACE]) as device:
					device.emit_click(uinput.KEY_SPACE)
			
	    		
