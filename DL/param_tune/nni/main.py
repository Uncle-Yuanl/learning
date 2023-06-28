#! /usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File   :   main.py
@Time   :   2023/05/10 16:52:02
@Author :   yhao 
@Email  :   data4.cmi@unilever.com
@Desc   :   experiment设置，使model.py在NNI生效
            refer: https://github.com/microsoft/nni/tree/c31d2574cb418acfb80c17bb2bd03531556325bd/examples/tutorials/hpo_quickstart_pytorch
                   https://nni.readthedocs.io/zh/latest/tutorials/hpo_quickstart_pytorch/main.html
'''

import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(f'【{__file__}】')
from pathlib import Path

# Step 1: Prepare the model
# -------------------------
# In first step, we need to prepare the model to be tuned.
#
# The model should be put in a separate script.
# It will be evaluated many times concurrently,
# and possibly will be trained on distributed platforms.
#
# In this tutorial, the model is defined in :doc:`model.py <model>`.
#
# In short, it is a PyTorch model with 3 additional API calls:
#
# 1. Use :func:`nni.get_next_parameter` to fetch the hyperparameters to be evalutated.
# 2. Use :func:`nni.report_intermediate_result` to report per-epoch accuracy metrics.
# 3. Use :func:`nni.report_final_result` to report final accuracy.
#
# Please understand the model code before continue to next step.

# Step 2: Define search space
# ---------------------------
# In model code, we have prepared 3 hyperparameters to be tuned:
# *features*, *lr*, and *momentum*.
#
# Here we need to define their *search space* so the tuning algorithm can sample them in desired range.
#
# Assuming we have following prior knowledge for these hyperparameters:
#
# 1. *features* should be one of 128, 256, 512, 1024.
# 2. *lr* should be a float between 0.0001 and 0.1, and it follows exponential distribution.
# 3. *momentum* should be a float between 0 and 1.
#
# In NNI, the space of *features* is called ``choice``;
# the space of *lr* is called ``loguniform``;
# and the space of *momentum* is called ``uniform``.
# You may have noticed, these names are derived from ``numpy.random``.
#
# For full specification of search space, check :doc:`the reference </hpo/search_space>`.
#
# Now we can define the search space as follow:

search_space = {
    'features': {'_type': 'choice', '_value': [128, 256, 512, 1024]},
    'lr': {'_type': 'loguniform', '_value': [0.0001, 0.1]},
    'momentum': {'_type': 'uniform', '_value': [0, 1]},
}

# Step 3: Configure the experiment
from nni.experiment import Experiment
experiment = Experiment('local')
experiment.config.trial_command = 'python model.py'
# When ``trial_code_directory`` is a relative path, it relates to current working directory.
# To run ``main.py`` in a different path, you can set trial code directory to ``Path(__file__).parent``.
experiment.config.trial_code_directory = Path(__file__).parent

experiment.config.search_space = search_space
# Configure tuning algorithm
experiment.config.tuner.name = 'TPE'
experiment.config.tuner.class_args['optimize_mode'] = 'maximize'

experiment.config.max_trial_number = 10
experiment.config.trial_concurrency = 2

# Step 4: Run the experiment
# You can use the web portal to view experiment status: http://localhost:8080.
experiment.run(8090)

input('Press enter to quit')
experiment.stop()