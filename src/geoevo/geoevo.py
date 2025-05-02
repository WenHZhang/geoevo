import gc
import numpy as np
from multiprocessing import Pool
import random
from scipy.stats import rankdata

class GeoEvoOptimizer:
    def __init__(self, popsize, iteration, n, obj_function, core, zonelist, adjmat, obj_type):
        """
        Initialize the GeoEvoOptimizer.
        
        Parameters
        ----------
        popsize (int):
        Population size of the optimization algorithm (number of candidate solutions in each generation).

        iteration (int):
        Maximum number of iterations the algorithm will execute.

        n (int):
        Number of hyperparameters to be tuned.

        obj_function (callable):
        Objective function to be optimized. Must accept a parameter array (train_data, val_data, param) and return a value.

        core (int):
        Number of parallel threads/workers for computation. Set to 1 for sequential processing.

        zonelist (list of str):
        Partition describing SSH. List of region identifier (e.g., ['zone1', 'zone2'])

        adjmat (2D array-like):
        Adjacency matrix defining neighborhood relationships between zones. Should be a matrix where 1 indicates adjacent zones and 0 otherwise.

        obj_type (bool):
        Optimization direction flag:
        - True: Minimization problem (seeks lowest objective value)
        - False: Maximization problem (seeks highest objective value)

        Returns
        ----------
        GeoEvoOptimizer object.
        """
        self.popsize = popsize
        self.iteration = iteration
        self.n = n
        self.obj_function = obj_function
        self.pop = None
        self.obj = None
        self.rank = None
        self.best = None
        self.core = core
        self.zonelist = zonelist
        self.obj_type = obj_type
        self.adjmat = adjmat
        self.mut = 0.8
        self.bound = None
        self.train_data = None
        self.val_data = None

    def _train(self, i, param, train_data, val_data):
        fitness_list = []
        for zone in self.zonelist:
            zone_data = train_data[train_data['zone'] == zone]
            zone_val = val_data[val_data['zone'] == zone]
            fitness = self.obj_function(zone_data, zone_val, param[i,:])
            if self.obj_type == False:
                fitness = fitness * -1
            fitness_list.append(fitness)
        gc.collect()
        return fitness_list
    
    def _train_single(self, param):
        fitness_list = []
        for zone in self.zonelist:
            zone_data = self.train_data[self.train_data['zone'] == zone]
            zone_val = self.val_data[self.val_data['zone'] == zone]
            fitness = self.obj_function(zone_data, zone_val, param)
            if self.obj_type == False:
                fitness = fitness * -1
            fitness_list.append(fitness)
        gc.collect()
        return fitness_list

    def _partrain(self, param, train_data, val_data):
        with Pool(processes=self.core) as p:
            results = p.starmap(self._train, [(i, param, train_data, val_data)
                                        for i in range(self.popsize)])
            p.close()
            p.join()
        fit = np.array(results)
        return fit

    def _evolve(self, j):
        min_b, max_b = np.array(self.bound).T
        dim = self.n
        a = self.pop[j,:]
        j_best_task = self.best[j]
        task_best_idx = np.argmin(self.rank[:,j_best_task])
        j_best_task_name = self.zonelist[j_best_task]
        adj_task_name = [str(name) for name in self.adjmat.index[self.adjmat[j_best_task_name]==1]]
        adj_task_idx = [np.where(self.zonelist == name)[0][0] for name in adj_task_name if name in self.zonelist]
        idxs = [idx for idx in range(self.popsize) if idx != j]
        b = j
        bidx = -1
        c = j
        cidx = -1
        if random.random() < 0.5:
            b, c = self.pop[np.random.choice(idxs, 2, replace=False), :]
        else:
            #Geo-DE
            task1, task2 = np.random.choice(adj_task_idx, 2, replace=False)
            temp = self.rank
            temp[j, task1] = 1e18
            temp[j, task2] = 1e18
            bidx = np.argmin(temp[:, task1])
            cidx = np.argmin(temp[:, task2])
            b = self.pop[bidx,:]
            c = self.pop[cidx,:]
        d = self.pop[task_best_idx,:]
        mutant = a + self.mut * (d - a) + self.mut * (b - c)
        trial = np.clip(mutant, min_b, max_b)
        mu_index = np.random.rand(dim) < 1/dim
        j_rand = int(np.floor(np.random.rand()*dim)) + 1
        mu_index[j_rand-1] = 1>0
        rand_k1 = np.random.rand(dim) < 0.5
        rand_k2 = ~rand_k1
        rand_k = np.random.rand(dim)
        delta_k = np.where(rand_k1, 2*rand_k ** (1/21) - 1, 1-(2-2*rand_k)**(1/21))
        mutation = trial + delta_k * (max_b-min_b)
        trial = np.where(mu_index, mutation, trial)
        trial = np.clip(trial, min_b, max_b)
        trial[0:1] = (trial[0:1]).astype(int)
        fobj = self._train_single(trial)
        return j, fobj, trial

    def _initilization(self, bound, train_data, val_data):
        self.bound = bound
        self.train_data = train_data
        self.val_data = val_data
        pop = np.random.rand(self.popsize, self.n)
        min_b, max_b = np.array(bound).T
        diff = np.fabs(min_b - max_b)
        pop = min_b + pop * diff
        self.pop = pop
        self.obj = self._partrain(pop, train_data, val_data)
        self.rank = rankdata(self.obj,axis=0)
        self.best = np.argmin(self.obj, axis=1)
        
    def optimize(self, bound, train_data, val_data):
        """
        Execute the algorithm.
        
        Parameters
        ----------
        bound (array-like of shape (n, 2)):
        Search space boundaries for hyperparameters. A 2D array where each row corresponds to a hyperparameter, with the first column defining its lower bound and the second column defining its upper bound.
        Example: [[0.1, 1.0], [10, 100]] indicates two hyperparameters - the first ranges [0.1, 1.0], the second ranges [10, 100].

        train_data (pd.DataFrame):
        Training dataset containing both input features and target values. Rows represent samples and columns represent features (with the last column typically being the target variable).

        val_data (pd.DataFrame):
        Validation dataset used for hyperparameter evaluation. Should follow the same format/structure as train_data. Used to assess generalization performance during optimization.
        
        Returns
        ----------
        output_solution (array-like of shape (m, 2)):
        Optimized hyperparameter matrix for all tasks. Each row corresponds to a task's optimal hyperparameters, where:
        - Column 0: First hyperparameter value
        - Column 1: Second hyperparameter value
        Example: For 3 tasks → [[0.5, 10], [0.8, 15], [0.3, 12]] indicates each task's optimized parameter pair.

        output_best_obj (array-like of shape (m, 1)):
        Best objective values achieved for each task. Values correspond to:
        - Minimum values when obj_type=True (minimization)
        - Maximum values when obj_type=False (maximization)
        Example: [[0.92], [0.85], [0.94]] represents each task's optimal performance metric.
        """
        self._initilization(bound, train_data, val_data)
        zoneslist = self.zonelist
        adjmat = self.adjmat
        obj = self.obj
        for i in range(self.iteration):
            with Pool(processes=self.core) as p:
                results = p.map(self._evolve, [m for m in range(self.popsize)])
                p.close()
                p.join()
            for j, err_j, pop_j in results:
                j_best_task = np.argmin(err_j)
                j_best_task_name = zoneslist[j_best_task]
                adj_task_name = [str(name) for name in adjmat.index[adjmat[j_best_task_name]==1]]
                adj_task_idx = [np.where(zoneslist == name)[0][0] for name in adj_task_name if name in zoneslist]
                np.append(adj_task_idx, j_best_task)
                if random.random() < 0.5:
                    update_range = range(len(zoneslist))
                else:
                    #Geo-SL
                    update_range = adj_task_idx
                for task_idx in update_range:
                    k = np.argmin(obj[:,task_idx])
                    k_fit = obj[k, :]
                    if err_j[task_idx] < k_fit[task_idx]:
                        self.obj[k, :] = err_j
                        self.pop[k, :] = pop_j
                        self.rank = rankdata(self.obj, axis=0)
                        self.best = np.argmin(self.obj, axis=1)   
            print(f'The {i}th iteration is done.')
        pop = self.pop
        obj = self.obj
        rank = self.rank
        best_ind_for_task_idx = np.argmin(rank, axis=0)
        output_solution = np.ones((np.size(rank, axis=1), np.size(pop, axis=1)))
        output_best_obj = np.ones((np.size(rank, axis=1), 1))
        for k in range(np.size(rank, axis=1)):
            output_solution[k, :] = pop[best_ind_for_task_idx[k], :]
            output_best_obj[k, :] = obj[best_ind_for_task_idx[k], k]
        return output_solution, output_best_obj