# -*- coding: utf8 -*-
"""
     OpenSeqSLAM
     Copyright 2013, Niko S��nderhauf Chemnitz University of Technology niko@etit.tu-chemnitz.de

     pySeqSLAM is an open source Python implementation of the original SeqSLAM algorithm published by Milford and Wyeth at ICRA12 [1]. SeqSLAM performs place recognition by matching sequences of images.

     [1] Michael Milford and Gordon F. Wyeth (2012). SeqSLAM: Visual Route-Based Navigation for Sunny Summer Days and Stormy Winter Nights. In Proc. of IEEE Intl. Conf. on Robotics and Automation (ICRA)

     gy_Rick:
     I change the demo.py, support loop closure and some visualizations

"""

from parameters import defaultParameters
from utils import AttributeDict
import matplotlib
import matplotlib.pyplot as plt
from copy import deepcopy
from scipy.io import loadmat, savemat
import time
import os
import numpy as np
from seqslam import *
import sys 
import pickle

def main():

    # set the default parameters
    groundtruthPath = '../0_datasets/city_centre/CityCentreGroundTruth.mat'
    params = defaultParameters()   # 给定一些初始的、公用的参数

    # set the custom parameters
    ds = AttributeDict()
    ds.name = 'citycentre_first'

    ds.preprocessing = AttributeDict()
    ds.preprocessing.save = 1
    ds.preprocessing.load = 1
    # ds.crop=[1 1 60 32]  # x0 y0 x1 y1  cropping will be done AFTER resizing!
    ds.crop = []
    params.dataset = [ds, deepcopy(ds)]  # 两个一模一样的 ds，是因为做回环检测时，对比的数据集就是它本身

    # where to save / load the results
    params.savePath = './results_citycentre_first'
    if not os.path.exists(params.savePath):
        os.mkdir(params.savePath)

    # now process the dataset
    # 每个数据集里面包含两份完全相同的数据集 
    seqslam = SeqSLAM(params)
    t1 = time.time()
    results = seqslam.findLoopClosure()  
    t2 = time.time()
    print("time taken: " + str(t2 - t1))

    # process and visual result
    # 对 gt 预处理，否则不能用，左右眼是掺杂在一起的，左右眼的图片可能会被匹配起来，这显然不对！
    groundtruthMat = loadmat(groundtruthPath)
    groundtruthMat = groundtruthMat['truth'][::2,::2]
    # 上述 gt 中还是存在一行中连续多列为 1，即一幅图片还可能与多幅邻近的图片匹配，在计算 positive 时不能直接累加，否则 recall 会很低
    gt_loop = np.count_nonzero(np.sum(groundtruthMat, 1))
    pr = []
    row = results.matches.shape[0]
    if len(results.matches) > 0:
        for mu in np.arange(0, 1, 0.01):
            idx = np.copy(results.matches[:, 0])  # The LARGER the score, the WEAKER the match.
            idx[results.matches[:, 1] > mu] = np.nan  # remove the weakest matches

            loopMat = np.zeros((row, row))
            for i in range(row):
                if not np.isnan(idx[i]):
                    loopMat[i, int(idx[i])] = 1
            
            p_loop = np.sum(loopMat)
            TP = np.sum(loopMat * groundtruthMat)
            print(TP)
            print(p_loop)
            pre = TP / p_loop
            rec = TP / gt_loop
            pr.append([pre, rec])

    pr = np.array(pr)

    # 尽管只画一幅图，但是 matplotlib 也推荐用 plt.subplots() 这种形式，不加参数就是一个 figure，一个 axis
    fig, ax = plt.subplots()
    ax.plot(pr[:, 1], pr[:, 0], '-o')
    print(len(pr))
    ax.set_title('PR Curve')
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.grid()
    plt.axis([0, 1.05, 0, 1.05])
    plt.show()

    with open('citycentre_first.pkl', 'wb') as f:  
        pickle.dump(pr, f)





if __name__ == "__main__":
    main()
