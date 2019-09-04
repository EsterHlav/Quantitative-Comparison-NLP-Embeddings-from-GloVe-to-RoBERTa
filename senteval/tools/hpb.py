"""
Support nicer user syntax:
    from hyperopt import hp
    hp.uniform('x', 0, 1)
"""
from __future__ import absolute_import

from hyperopt_changes import hp_logbaseuniform as logbaseuniform
from hyperopt_changes import hp_qlogbaseuniform as qlogbaseuniform

from hyperopt_changes import hp_logbasenormal as logbasenormal
from hyperopt_changes import hp_qlogbasenormal as qlogbasenormal
