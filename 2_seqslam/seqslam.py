from utils import AttributeDict
import os
import numpy as np
from scipy.io import loadmat, savemat
import matplotlib.image as mpimg
from PIL import Image 
from copy import deepcopy

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm

class SeqSLAM():
    params = None
    
    def __init__(self, params):
        self.params = params

    def findLoopClosure(self):
        # begin with preprocessing of the images
        results = AttributeDict()

        # image difference matrix
        if self.params.DO_DIFF_MATRIX:
            results = self.doDifferenceMatrix(results)

        results.DD = results.D

        # find the matches
        if self.params.DO_FIND_MATCHES:
            results = self.doFindLoop(results)
        return results

    def getDifferenceMatrix(self, data0preproc, data1preproc):
        # TODO parallelize
        n = data0preproc.shape[1]
        m = data1preproc.shape[1]
        D = np.zeros((n, m))   
    
        #parfor?
        for i in range(n):
            d = data1preproc - np.tile(data0preproc[:,i],(m, 1)).T
            D[i,:] = np.sum(np.abs(d), 0)/n
            
        return D  # 这里面对角元素肯定是 0，因为对比的两个数据集是相同的。
    
    def doDifferenceMatrix(self, results):
        filename = '%s/difference-%s.mat' % (self.params.savePath, self.params.dataset[0].name)  
    
        if self.params.differenceMatrix.load and os.path.exists(filename):
            print('Loading image difference matrix from file %s ...' % filename)
    
            d = loadmat(filename)
            results.D = d['D']
        else:
            if len(results.dataset)<2:
                print('Error: Cannot calculate difference matrix with less than 2 datasets.')
                return None
    
            print('Calculating image difference matrix ...')
    
            results.D=self.getDifferenceMatrix(results.dataset[0].preprocessing, results.dataset[1].preprocessing)
            
            # save it
            if self.params.differenceMatrix.save:                   
                savemat(filename, {'D':results.D})
            
        return results


    def doFindLoop(self, results):
        filename = '%s/LoopClosure-%s.mat' % (self.params.savePath, self.params.dataset[0].name)
        if self.params.matching.load and os.path.exists(filename):
            print('Loading LoopClosure from file %s ...' % filename)
            m = loadmat(filename)
            results.matches = m['Loop']
        else:
            print('Searching for matching images ...')
            # make sure ds is dividable by two
            self.params.matching.ds = self.params.matching.ds + np.mod(self.params.matching.ds,2)

            matches = self.getLoopClosure(results.DD)
            # save it
            if self.params.matching.save:
                savemat(filename, {'Loop':matches})
            results.matches = matches

        return results

    def getLoopClosure(self, DD):
        # query 的 image 会考虑之前总长为 ds 的图片序列
        # 被 query 的 image 长度固定，也是 ds，但是根据速度的不同，可能会有相同的或者间隔的图片     
        move_min = int(self.params.matching.vmin * self.params.matching.ds)    
        move_max = int(self.params.matching.vmax * self.params.matching.ds)       
        # 设定若干可能的位移量
        move = np.arange(move_min, move_max+1)
        # 设定若干可能的速度
        v = move.astype(float) / self.params.matching.ds
        
        # 最终给出的 idx_add 矩阵每行对应一个 v，其中的元素从 0 到 (ds-1)*v
        idx_add = np.tile(np.arange(0, self.params.matching.ds), (len(v),1))
        idx_add = np.floor(idx_add * np.tile(v, (idx_add.shape[1], 1)).T) 
              
        # 每个 query 的 image 对应一行， query 的 image 来自 DD 的列
        # 并非所有 DD 列对应的图片都作为 query，
        # 例如前边的若干图片都会被忽略掉，因为一般不会在一开始就出现 loop
        # 而且还要考虑 seq 的长度，要留 ds 张图片的余量
        matches = np.nan*np.ones((DD.shape[1],2))  

        # 开头的若干幅图片不作为 query 图片
        ignore_head = 50     

        # 邻近的若干幅图片不被 query
        ignore = 10

        for N in range(ignore_head + self.params.matching.ds, DD.shape[1]):            
            # 矩阵 x 的行都相同，每一行都是从第 N 列开始，共追溯 ds 列，每一列的开头位置 
            x= np.tile(np.arange(N - self.params.matching.ds + 1, N+1), (len(v), 1)) * DD.shape[0]            
            # score 包含每次多 v 路径搜索得到的 DD 矩阵元素累加的最小值
            score = np.infty * np.ones(N)   
            local_best_index = np.zeros(N)
            flatDD = DD.flatten('F')
            # 搜索从第一幅图片开始，直到靠近 query 图片的索引为止，要留出空隙，避免邻近图片误判成 loop
            for s in range(N - move_max - ignore + 1):   
                # 给定了 s 就是给定了固定点，然后以 s 为起点以不同速度 v 搜索多条轨迹               
                y = np.copy(idx_add+s) 
                # x 确定了与 N 相关的 ds 列
                # y 确定了从上向下以此搜索
                idx = (x + y).astype(int)
                # idx 中的每一行就是一条轨迹
                ds = np.sum(flatDD[idx],1)
                # 找到本次搜索得到的最小值，存入 score，其中第 s 个元素存放的是以第 s 行为固定点时以不同速度搜索路径的最小值
                v_index = np.argmin(ds)
                score[s] = ds[v_index]
                local_best_index[s] = int(y[v_index, -1])            

            # find min score and 2nd smallest score outside of a window around the minimum             
            min_idx = np.argmin(score)
            global_best_index = local_best_index[min_idx] 
            min_value = score[min_idx]
            # 定义以 min_idx 为中心的窗口
            window = np.arange(np.max((0, min_idx-self.params.matching.Rwindow/2)), np.min((len(score), min_idx+self.params.matching.Rwindow/2)))
            # 
            not_window = list(set(range(len(score))).symmetric_difference(set(window))) #xor
            min_value_2nd = np.min(score[not_window])
            
            # 每个 query 图片，只返回一个最佳匹配。
            match = [global_best_index, min_value / min_value_2nd]
            matches[N,:] = match

        return matches
