#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from .base import SkoBase  # 导入基类
from sko.tools import func_transformer  # 导入函数变换工具
from abc import ABCMeta, abstractmethod  # 导入抽象基类
from .operators import crossover, mutation, ranking, selection  # 导入遗传算法操作函数


class GeneticAlgorithmBase(SkoBase, metaclass=ABCMeta):
    # 遗传算法基类
    def __init__(self, func, n_dim,
                 size_pop=50, max_iter=200, prob_mut=0.001,
                 constraint_eq=tuple(), constraint_ueq=tuple(), early_stop=None, n_processes=0):
        self.func = func_transformer(func, n_processes)  # 函数变换
        assert size_pop % 2 == 0, 'size_pop must be even integer'
        self.size_pop = size_pop  # 种群大小
        self.max_iter = max_iter  # 最大迭代次数
        self.prob_mut = prob_mut  # 变异概率
        self.n_dim = n_dim  # 变量维度
        self.early_stop = early_stop  # 提前停止标志

        # 约束条件
        self.has_constraint = len(constraint_eq) > 0 or len(constraint_ueq) > 0
        self.constraint_eq = list(constraint_eq)  # 等式约束
        self.constraint_ueq = list(constraint_ueq)  # 不等式约束

        self.Chrom = None  # 染色体
        self.X = None  # 决策变量（解空间）
        self.Y_raw = None  # 原始适应度
        self.Y = None  # 适应度（带约束惩罚项）
        self.FitV = None  # 适应度值

        self.generation_best_X = []  # 每代最佳决策变量
        self.generation_best_Y = []  # 每代最佳适应度

        self.all_history_Y = []  # 所有历史适应度
        self.all_history_FitV = []  # 所有历史适应度值

        self.best_x, self.best_y = None, None  # 最优解和最优值

    @abstractmethod
    def chrom2x(self, Chrom):
        pass

    def x2y(self):
        self.Y_raw = self.func(self.X)  # 计算原始适应度
        if not self.has_constraint:  # 无约束情况
            self.Y = self.Y_raw
        else:
            # 计算约束惩罚项
            penalty_eq = np.array([np.sum(np.abs([c_i(x) for c_i in self.constraint_eq])) for x in self.X])
            penalty_ueq = np.array([np.sum(np.abs([max(0, c_i(x)) for c_i in self.constraint_ueq])) for x in self.X])
            self.Y = self.Y_raw + 1e5 * penalty_eq + 1e5 * penalty_ueq
        return self.Y

    @abstractmethod
    def ranking(self):
        pass

    @abstractmethod
    def selection(self):
        pass

    @abstractmethod
    def crossover(self):
        pass

    @abstractmethod
    def mutation(self):
        pass

    def run(self, max_iter=None):
        self.max_iter = max_iter or self.max_iter  # 更新最大迭代次数
        best = []  # 记录最优解
        for i in range(self.max_iter):
            self.X = self.chrom2x(self.Chrom)  # 染色体转换为解空间
            self.Y = self.x2y()  # 计算适应度
            self.ranking()  # 排序
            self.selection()  # 选择
            self.crossover()  # 交叉
            self.mutation()  # 变异

            # 记录每代最优解
            generation_best_index = self.FitV.argmax()
            self.generation_best_X.append(self.X[generation_best_index, :])
            self.generation_best_Y.append(self.Y[generation_best_index])
            self.all_history_Y.append(self.Y)
            self.all_history_FitV.append(self.FitV)

            if self.early_stop:
                best.append(min(self.generation_best_Y))
                if len(best) >= self.early_stop:
                    if best.count(min(best)) == len(best):
                        break
                    else:
                        best.pop(0)

        global_best_index = np.array(self.generation_best_Y).argmin()
        self.best_x = self.generation_best_X[global_best_index]
        self.best_y = self.func(np.array([self.best_x]))
        return self.best_x, self.best_y

    fit = run


class GA(GeneticAlgorithmBase):
    """遗传算法

    参数
    ----------------
    func : function
        要优化的函数
    n_dim : int
        函数的变量数量
    size_pop : int
        种群大小
    max_iter : int
        最大迭代次数
    prob_mut : float
        变异概率
    constraint_eq : tuple
        等式约束
    constraint_ueq : tuple
        不等式约束
    precision : array_like
        每个变量的精度
    early_stop : int
        提前停止的代数
    n_processes : int
        进程数，0表示使用所有CPU
    """

    def __init__(self, func, n_dim,
                 size_pop=50, max_iter=200,
                 prob_mut=0.001,
                 lb=-1, ub=1,
                 constraint_eq=tuple(), constraint_ueq=tuple(),
                 precision=1e-7, early_stop=None, n_processes=0):
        super().__init__(func, n_dim, size_pop, max_iter, prob_mut, constraint_eq, constraint_ueq, early_stop, n_processes=n_processes)

        self.lb, self.ub = np.array(lb) * np.ones(self.n_dim), np.array(ub) * np.ones(self.n_dim)
        self.precision = np.array(precision) * np.ones(self.n_dim)  # 精度

        # Lind是每个变量的基因数
        Lind_raw = np.log2((self.ub - self.lb) / self.precision + 1)
        self.Lind = np.ceil(Lind_raw).astype(int)

        # 如果精度是整数
        self.int_mode_ = (self.precision % 1 == 0) & (Lind_raw % 1 != 0)
        self.int_mode = np.any(self.int_mode_)
        if self.int_mode:
            self.ub_extend = np.where(self.int_mode_,
                                      self.lb + (np.exp2(self.Lind) - 1) * self.precision,
                                      self.ub)

        self.len_chrom = sum(self.Lind)

        self.crtbp()

    def crtbp(self):
        # 创建种群
        self.Chrom = np.random.randint(low=0, high=2, size=(self.size_pop, self.len_chrom))
        return self.Chrom

    def gray2rv(self, gray_code):
        # 格雷码转换为实数值
        _, len_gray_code = gray_code.shape
        b = gray_code.cumsum(axis=1) % 2
        mask = np.logspace(start=1, stop=len_gray_code, base=0.5, num=len_gray_code)
        return (b * mask).sum(axis=1) / mask.sum()

    def chrom2x(self, Chrom):
        cumsum_len_segment = self.Lind.cumsum()
        X = np.zeros(shape=(self.size_pop, self.n_dim))
        for i, j in enumerate(cumsum_len_segment):
            if i == 0:
                Chrom_temp = Chrom[:, :cumsum_len_segment[0]]
            else:
                Chrom_temp = Chrom[:, cumsum_len_segment[i - 1]:cumsum_len_segment[i]]
            X[:, i] = self.gray2rv(Chrom_temp)

        if self.int_mode:
            X = self.lb + (self.ub_extend - self.lb) * X
            X = np.where(X > self.ub, self.ub, X)
        else:
            X = self.lb + (self.ub - self.lb) * X
        return X

    ranking = ranking.ranking
    selection = selection.selection_tournament_faster
    crossover = crossover.crossover_2point_bit
    mutation = mutation.mutation

    def to(self, device):
        '''
        使用PyTorch获得并行性能
        '''
        try:
            import torch
            from .operators_gpu import crossover_gpu, mutation_gpu, selection_gpu, ranking_gpu
        except:
            print('需要PyTorch')
            return self

        self.device = device
        self.Chrom = torch.tensor(self.Chrom, device=device, dtype=torch.int8)

        def chrom2x(self, Chrom):
            '''
            我们不打算将所有操作符都作为张量，
            因为目标函数可能不适用于PyTorch
            '''
            Chrom = Chrom.cpu().numpy()
            cumsum_len_segment = self.Lind.cumsum()
            X = np.zeros(shape=(self.size_pop, self.n_dim))
            for i, j in enumerate(cumsum_len_segment):
                if i == 0:
                    Chrom_temp = Chrom[:, :cumsum_len_segment[0]]
                else:
                    Chrom_temp = Chrom[:, cumsum_len_segment[i - 1]:cumsum_len_segment[i]]
                X[:, i] = self.gray2rv(Chrom_temp)

            if self.int_mode:
                X = self.lb + (self.ub_extend - self.lb) * X
                X = np.where(X > self.ub, self.ub, X)
            else:
                X = self.lb + (self.ub - self.lb) * X
            return X

        self.register('mutation', mutation_gpu.mutation). \
            register('crossover', crossover_gpu.crossover_2point_bit). \
            register('chrom2x', chrom2x)

        return self


class EGA(GA):
    """精英遗传算法"""

    def __init__(self, func, n_dim,
                 size_pop=50, max_iter=200,
                 prob_mut=0.001, n_elitist=0,
                 lb=-1, ub=1,
                 constraint_eq=tuple(), constraint_ueq=tuple(),
                 precision=1e-7, early_stop=None):
        super().__init__(func, n_dim, size_pop, max_iter, prob_mut, lb, ub, constraint_eq, constraint_ueq, precision,
                         early_stop)
        self._n_elitist = n_elitist

    def run(self, max_iter=None):
        self.max_iter = max_iter or self.max_iter
        best = []
        for i in range(self.max_iter):
            self.X = self.chrom2x(self.Chrom)
            self.Y = self.x2y()
            self.ranking()

            # 选择精英个体不执行选择(), 交叉() 和 变异()，并且从种群中移除它们。
            # 临时地。
            idx_elitist = np.sort(self.Y.argsort()[0:self._n_elitist])
            self.size_pop -= self._n_elitist
            elitist_FitV = np.take(self.FitV, idx_elitist, axis=0)
            self.FitV = np.delete(self.FitV, idx_elitist, axis=0)
            elitist_Chrom = np.take(self.Chrom, idx_elitist, axis=0)
            self.Chrom = np.delete(self.Chrom, idx_elitist, axis=0)

            self.selection()
            self.crossover()
            self.mutation()

            # 将精英个体添加回下一代种群中。
            idx_insert = np.array([idx_v - i for i, idx_v in enumerate(idx_elitist)])
            self.size_pop += self._n_elitist
            self.FitV = np.insert(self.FitV, idx_insert, elitist_FitV, axis=0)
            self.Chrom = np.insert(self.Chrom, idx_insert, elitist_Chrom, axis=0)

            # 记录每代最优解
            generation_best_index = self.FitV.argmax()
            self.generation_best_X.append(self.X[generation_best_index, :])
            self.generation_best_Y.append(self.Y[generation_best_index])
            self.all_history_Y.append(self.Y)
            self.all_history_FitV.append(self.FitV)

            if self.early_stop:
                best.append(min(self.generation_best_Y))
                if len(best) >= self.early_stop:
                    if best.count(min(best)) == len(best):
                        break
                    else:
                        best.pop(0)

        global_best_index = np.array(self.generation_best_Y).argmin()
        self.best_x = self.generation_best_X[global_best_index]
        self.best_y = self.func(np.array([self.best_x]))
        return self.best_x, self.best_y


class MGG(GeneticAlgorithmBase):
    """多目标遗传算法"""

    def __init__(self, func, n_dim,
                 size_pop=50, max_iter=200,
                 prob_mut=0.001,
                 lb=-1, ub=1,
                 constraint_eq=tuple(), constraint_ueq=tuple(),
                 precision=1e-7, early_stop=None, n_processes=0):
        super().__init__(func, n_dim, size_pop, max_iter, prob_mut, constraint_eq, constraint_ueq, early_stop,
                         n_processes=n_processes)

        self.lb, self.ub = np.array(lb), np.array(ub)
        self.precision = np.array(precision) * np.ones(self.n_dim)

        self.int_mode_ = (self.precision % 1 == 0)
        self.int_mode = np.any(self.int_mode_)
        if self.int_mode:
            self.ub_extend = np.where(self.int_mode_,
                                      self.lb + (np.exp2(self.Lind) - 1) * self.precision,
                                      self.ub)

        self.len_chrom = sum(self.Lind)

        self.crtbp()

    def crtbp(self):
        # 创建种群
        self.Chrom = np.random.randint(low=0, high=2, size=(self.size_pop, self.len_chrom))
        return self.Chrom

    def chrom2x(self, Chrom):
        cumsum_len_segment = self.Lind.cumsum()
        X = np.zeros(shape=(self.size_pop, self.n_dim))
        for i, j in enumerate(cumsum_len_segment):
            if i == 0:
                Chrom_temp = Chrom[:, :cumsum_len_segment[0]]
            else:
                Chrom_temp = Chrom[:, cumsum_len_segment[i - 1]:cumsum_len_segment[i]]
            X[:, i] = self.gray2rv(Chrom_temp)

        if self.int_mode:
            X = self.lb + (self.ub_extend - self.lb) * X
            X = np.where(X > self.ub, self.ub, X)
        else:
            X = self.lb + (self.ub - self.lb) * X
        return X

    ranking = ranking.ranking
    selection = selection.selection_tournament
    crossover = crossover.crossover_2point_bit
    mutation = mutation.mutation

    def run(self, max_iter=None):
        self.max_iter = max_iter or self.max_iter
        best = []
        for i in range(self.max_iter):
            self.X = self.chrom2x(self.Chrom)
            self.Y = self.x2y()
            self.ranking()
            self.selection()
            self.crossover()
            self.mutation()

            # 记录每代最优解
            generation_best_index = self.FitV.argmax()
            self.generation_best_X.append(self.X[generation_best_index, :])
            self.generation_best_Y.append(self.Y[generation_best_index])
            self.all_history_Y.append(self.Y)
            self.all_history_FitV.append(self.FitV)

            if self.early_stop:
                best.append(min(self.generation_best_Y))
                if len(best) >= self.early_stop:
                    if best.count(min(best)) == len(best):
                        break
                    else:
                        best.pop(0)

        global_best_index = np.array(self.generation_best_Y).argmin()
        self.best_x = self.generation_best_X[global_best_index]
        self.best_y = self.func(np.array([self.best_x]))
        return self.best_x, self.best_y


if __name__ == '__main__':
    '''
    示例：
    求解函数 f = 10*sin(5*x) + 7*cos(4*x), x ∈ [-10, 10]
    '''
    import matplotlib.pyplot as plt

    ga = GA(func=lambda x: 10 * np.sin(5 * x) + 7 * np.cos(4 * x), n_dim=1, size_pop=100, max_iter=500, lb=-10, ub=10)
    best_x, best_y = ga.run()
    print('best_x:', best_x, 'best_y:', best_y)
    Y_history = np.array(ga.all_history_Y)
    plt.plot(Y_history)
    plt.show()
