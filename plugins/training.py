import os
import csv
import timeit
import datetime
import plugin_interface as plugintypes


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
            f.write('%' + self.time_stamp + '\n')

    # called with each new sample
    def __call__(self, sample):
        ran=random.random()
        if sample.id == 0 and ran < 1./3.:
            os.system('play --no-show-progress --null --channels 1 synth %s sine %f' % (0.5, 440))
        if sample.id > 3 and sample.id < 53 and ran < 1./3.:
            row = ''
            row += str(t)
            row += self.delim
            row += str(sample.channel_data[1])
            with open(self.file_name, 'a') as f:
                f.write(row)
