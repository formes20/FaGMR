"""
  Copyright 2020 ETH Zurich, Secure, Reliable, and Intelligent Systems Lab

  Licensed under the Apache License, Version 2.0 (the "License");
  you may not use this file except in compliance with the License.
  You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

  Unless required by applicable law or agreed to in writing, software
  distributed under the License is distributed on an "AS IS" BASIS,
  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
  See the License for the specific language governing permissions and
  limitations under the License.
"""


import multiprocessing

from enum import Enum



class Device(Enum):
    CPU = 0
    CUDA = 1


class config:

    # General options
    # netname = '../data/acasxu/nets/ACASXU_run2a_1_1_batch_2000.onnx'  # the network name, the extension can be only .pyt, .tf and .meta
    # netname = '../data/mnist_crown_large_0.2.onnx'
    # netname = '../data/mnist_ffnn_10x80.onnx'
    # netname = '../data/mnist_8_100_flattened.pyt'
    # netname = '../data/mnist_relu_6_200.onnx'
    # netname = '../data/ffnnRELU__Point_6_500.onnx'
    netname = '../data/mnist_relu_5_100.onnx'
    # netname = '../data/convSmallRELU__Point.onnx'
    # netname = '../data/convBigRELU__DiffAI_mnist.onnx'

    epsilon = 0.05 # the epsilon for L_infinity perturbation
    zonotope = None # file to specify the zonotope matrix
    domain = 'refinepoly' # the domain name can be either deepzono, refinezono, deeppoly or refinepoly
    # dataset = 'mnist'  # the dataset, can be either mnist, cifar10, or acasxu
    dataset = 'mnist'
    complete = False # flag specifying where to use complete verification or not
    timeout_lp = 1 # timeout for the LP solver
    timeout_milp = 1 # timeout for the MILP solver
    timeout_final_lp = 360
    timeout_final_milp = 100
    partial_milp = 0 # number of activation layers to encode with milp: 0 none, -1
    max_milp_neurons = 30 # maximum number of neurons per layer to encode using MILP for partial MILP attempt
    timeout_complete = True # cumulative timeout for the complete verifier
    use_default_heuristic = True # whether to use the area heuristic for the DeepPoly ReLU approximation or to always create new noise symbols per relu for the DeepZono ReLU approximation
    mean = None # the mean used to normalize the data with
    std = None # the standard deviation used to normalize the data with
    num_tests = 1000 # Number of images to test
    from_test = 0 # From which number to start testing
    debug = True # Whether to display debug info
    subset = None
    target = None # 
    epsfile = None
    vnn_lib_spec = None # Use inputs and constraints defined in a file respecting the vnn_lib standard

    # refine options
    use_milp = False# Whether to use MILP, faster but less precise
    refine_neurons = True # whether to refine intermediate neurons
    n_milp_refine = 1 # # Number of milp refined layers
    sparse_n = 70
    # sparse_n = 70
    numproc = multiprocessing.cpu_count() # number of processes for milp/lp/krelu
    normalized_region = True
    k = 3 # group size for k-activation relaxations
    K = 3
    s = -1 # sparsity parameter for k-activation relaxatin. Maximum overlap. Negative numbers compute s<-K+s. -2 is the default
    # Geometric options
    geometric = False # Whether to do geometric analysis
    attack = False # Whether to attack in geometric analysis
    data_dir = None # data location for geometric analysis
    geometric_config = None # geometric config location
    num_params = 0 # Number of transformation parameters for geometric analysis
    approx_k = True



    # Acas Xu
    specnumber = 1 # Acas Xu spec number
    epsilon_list = []
    # arbitrary input / output
    input_box = None # input box file to use
    output_constraints = None # output constraints file to check

    # GPU options
    device = Device.CPU # Which device Deeppoly should run on

    # spatial options
    spatial = True
    t_norm = 'inf'
    delta = 0.026
    gamma = float('inf')
    quant_step = None

    precise_flag = 0
    # 9 x 100 : precise_num = 14  0.25 2<- pre_per=0.5 conv = 1
    precise_num = 13
    # from 0.25, conv = 2 -> 0.5**conv control where to start e.g. start from the conv th : 0.125
    conv = 3
    # 4
    pre_per = 0.125


    # Addition
    relu_count = 0
    effect_count = 0

    record_prebound_l = []
    record_prebound_u = []
    sum_args_time = 0
    sum_cons_time = 0
    sum_refine_time = 0
    sum_time = 0
    model = 'FA' # model can be ['para', 'all', 'FA']
    test = ''
    actnum = 0 # how many activate layer after FC in NN
    # the num of correctly classified image used
    cor_num = 1000
    conv_flag = 1
