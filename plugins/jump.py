import plugin_interface as plugintypes
import uinput
import random


class PluginJump(plugintypes.IPluginExtended):
    def __init__(self):
        self.dev = uinput.Device([uinput.KEY_SPACE])

    def activate(self):
        print "Activated"

    def deactivate(self):
        print "Deactivated"

    def show_help(self):
        print "Think of jumping in order to jump"

    def __call__(self, sample):
        if random.random() > (1 - 1 /  250.0):
            self.dev.emit_click(uinput.KEY_SPACE)
