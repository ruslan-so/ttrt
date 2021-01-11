from __future__ import print_function
import sys
import time
import json
import logging
#import path
import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable as V
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

scale_fn = {'linear':lambda x: x,
            'squared': lambda x: x**2,
            'cubic': lambda x: x**3}
