import os
import csv
import timeit
import datetime
import plugin_interface as plugintypes
import random


class PluginPrint(plugintypes.IPluginExtended):
    def __init__(self, file_name="collect.csv", delim=",", verbose=False):
        now = datetime.datetime.now()
        self.time_stamp = '%d-%d-%d_%d-%d-%d' % (now.year, now.month, now.day, now.hour, now.minute, now.second)
        self.file_name = self.time_stamp
        self.start_time = timeit.default_timer()
        self.delim = delim
        self.verbose = verbose

    def activate(self):
        if len(self.args) > 0:
            if 'no_time' in self.args:
                self.file_name = self.args[0]
            else:
                self.file_name = self.args[0] + '_' + self.file_name;
            if 'verbose' in self.args:
                self.verbose = True

        self.file_name = self.file_name + 'training sample'+'.csv'
        print "Will export CSV to:", self.file_name
        # Open in append mode
        with open(self.file_name, 'a') as f:
            #f.write('%' + self.time_stamp + '\n')

    def deactivate(self):
        print "Closing, CSV saved to:", self.file_name
        return

    def show_help(self):
        print "Optional argument: [filename] (default: collect.csv)"

    # called with each new sample
    def __call__(self, sample):
        t = timeit.default_timer() - self.start_time
        row = ''
        row += str(t)
        row += self.delim
        row += str(sample.channel_data[1])
        row += self.delim
        ran=random.random()
        if sample.id == 0 and ran < 1. / 3.:
            os.system('play -n synth 0.1 tri  440.0')
            b=1
        elif sample.id == 50:
            b=0
        row += str(b)
        row += '\n'
        with open(self.file_name, 'a') as f:
            f.write(row)
