# -*- coding: utf-8 -*-
"""
Created on Tue Jun  6 15:38:17 2017

@author: pj
"""

import matplotlib.pyplot as plt
import numpy as np
import wave
import sys

from scipy.io.wavefile import write

spf = wave.open('helloworld.wav', 'r')

signal = spf.readframes(-1)

