import sys
import torch 
import torch.nn as nn
import os

import argparse
import GPUtil

from Colorization_Train.Colorization_Train import Colorization_Train_runner

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--conf', type=str, default='./confs/color.conf')
    opt = parser.parse_args()
    deviceIDs = GPUtil.getAvailable(order='memory', limit=1, maxLoad=0.5, maxMemory=0.5, includeNan=False,
                                    excludeID=[], excludeUUID=[])

    gpu = deviceIDs[0]
    print("GPU: ", gpu)  
    os.environ["CUDA_VISIBLE_DEVICES"] = '{0}'.format(gpu)
    print(opt.conf)
    trainrunner = Colorization_Train_runner(
        opt.conf
    )
    
    trainrunner.train()