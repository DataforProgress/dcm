"""
dcmpy

a discrete choice modeling package for python
"""

__version__ = "0.1.0"
__author__ = 'Nic Fishman'
__credits__ = 'University of Oxford'

from dcmpy.cvx.maxdiff import maxchoice
from dcmpy.models import *
from dcmpy.losses import * 
from dcmpy.fit import *
from dcmpy.uncertainty import *
