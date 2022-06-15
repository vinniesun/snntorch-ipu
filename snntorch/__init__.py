from ._version import __version__
from ._neurons import *

import os
import subprocess

CWD = os.path.dirname(__file__)
if not os.path.isfile(os.path.join(CWD, "so_file/heaviside_custom_ops.so")) or \
        not os.path.isfile(os.path.join(CWD, "so_file/straight_through_estimator_custom_ops.so")) or \
        not os.path.isfile(os.path.join(CWD, "fast_sigmoid_custom_ops.so")):
            print("Missing so files, will compile them now!")
            
            custom_ops_path = os.path.join(CWD, "custom_ops")
            subprocess.call(["make", "-C", custom_ops_path])
            print("Successfully created!")
