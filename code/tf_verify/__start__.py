from config import *
from My_fun import *


for model in ['para']:
    config.model = model
    for per in [1/255]:
        config.delta = per
        for from_test in [0]:

            config.from_test = from_test
            config.num_tests = 200
            # config.netname = '../data/convMedGSIGMOID__Point.onnx'
            config.netname = '../data/mnist_relu_5_100.onnx'
            config.dataset = 'mnist'

            config.relu_count = 0
            config.effect_count = 0
            config.record_prebound_u = []
            config.record_prebound_l = []
            config.precise_flag = 1
            config.sum_args_time = 0
            config.sum_cons_time = 0
            config.sum_refine_time = 0
            config.sum_time = 0
            config.actnum = 0
            used.rest_points = []
            used.record = []
            used.conv_count = 0
            used.bias_new = []
            used.weight_new = []



            exec(open("__main__.py", encoding = 'utf-8').read())