import random as r
import numpy as np
import sys
import math
import time
import os
from matplotlib import pyplot as plt

class genetic_alg:

    # algorithm
    # test fitness
    # check exit criteria
    # save % of the elite for the next generation
    # mutate random -% from full pop including elite
    # perform crossover on -% of full pop including elite
    # add elite to next gen
    # fill the rest of the next gen from mutated/crossover old pop

    def __init__(self):
        self.random_cap = 6
        self.KFCV_index = 1
        self.pop_size = 500
        self.mutation_rate = int(self.pop_size *.60)
        self.crossover_rate = int(self.pop_size *.80)
        self.elite_rate = int(self.pop_size * .1)
        self.elite_hold = []
        self.gen_limit = 25
        self.layer = [4,3,3,3,3]
        self.beta = self.initialize_pop()
        self.mse = []
        self.min_err = sys.maxsize
        self.fitness = Fitness(self.pop_size, len(self.layer)-1)
        self.fitness.set_betas(self.beta)
        self.current_milli_time = lambda: int(round(time.time() * 1000))
        self.best_betas = []
        self.best_training_mse = []
        self.best_testing_mse = []
        self.plot_x =[]
        self.plot_y =[]
        
    def initialize_pop(self):
        beta = []
        for i in range(self.pop_size):
            beta.append(self.initialize_beta())
        return beta

    def initialize_beta(self):
        beta = []
        for i in range(len(self.layer)-1):
            beta.append([])
            for j in range(self.layer[i]+1):
                beta[i].append(np.array([self.rand(), self.rand(), self.rand()]))
            beta[i] = np.matrix(beta[i])
        return beta

    def write_to_file(self):
        if os.path.exists("Results"):
            os.remove("Results")
        fh = open('Results',"a")
        fh.write("Genetic Algorithm Stats\n")
        fh.write("Population Size: "+str(self.pop_size)+"\n" )
        fh.write("Generation Limit: "+str(self.gen_limit)+"\n")
        fh.write("Mutation rate: "+str(self.mutation_rate)+"\n")
        fh.write("Crossover rate: "+str(self.crossover_rate)+"\n")
        fh.write("Elite Rollover rate: "+str(self.elite_rate)+"\n")
        fh.write("Layers: "+str(self.layer)+"\n\n")
        fh.write("Results of 10 Fold Cross Validation Using Genetic Algorithm\n")
        min_avg = sys.maxsize
        for i in range(len(self.best_betas)):
            avg =  (self.best_training_mse[i]+self.best_testing_mse[i])/2
            if avg < min_avg or i == 0:
                min_avg = avg
                temp_beta = self.best_betas[i]
            string = 'Fold: '+ str(i+1)+'\n'
            string += '\tTraining MSE: '+str(self.best_training_mse[i])+'\n'
            string += '\tTestning MSE: '+str(self.best_testing_mse[i])+'\n'
            string += '\tAverage MSE:  '+ str(avg)+'\n'
            fh.write(string)
        fh.write("\nBest Scoring Beta (mse avg: "+str(min_avg)+")\n\n")
        for i in range(len(temp_beta)):
            string = 'Beta[{}] = [\n'
            fh.write(string.format(str(i)))
            for j in range(len(temp_beta[i])):
                fh.write("\t"+str(temp_beta[i][j])+"\n")
            fh.write("]\n")
    
    def plot_fold(self):
        name = "Fold "+str(self.KFCV_index)+" MSE: "+str(self.min_err)
        plt.plot(self.plot_x, self.plot_y,label=name)
        plt.xlabel("Generations")
        plt.ylabel('Mean Squared Error')
        self.plot_x = []
        self.plot_y = []
            
    def run_alg(self):
        while self.KFCV_index < 11:
            # tm = self.current_milli_time()
            self.run_fold()
            self.best_training_mse.append(self.min_err)
            self.best_betas.append(self.beta[0])
            # print('Run fold time: '+str(self.current_milli_time() - tm) )
            print("Training Min Err: "+str(self.min_err)+" Fold: "+str(self.KFCV_index))
            self.best_testing_mse.append(self.run_test_fold(self.beta[0]))
            print("Testing Error: "+str(self.best_testing_mse[self.KFCV_index-1]))
            self.plot_fold()
            self.KFCV_index +=1
            # generate the next fold data
            self.fitness.data.KFCV(self.KFCV_index)
            self.min_err = sys.maxsize
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.05),
          ncol=3, fancybox=True, shadow=True)
        plt.show()
        self.write_to_file()
        

    def run_fold(self):
        count = 0
        while count < self.gen_limit:
            tm = self.current_milli_time()
            self.beta, self.mse = self.fitness.pop_fitness("train")# test fitness
            self.set_min_err()
            count +=1# check exit criteria
            if count == self.gen_limit:
                break
            self.stow_elite()# save 5% of the elite for the next generation
            self.mutate_population()# mutate random 30% from full pop including elite
            self.crossover_population()# perform crossover on 80% of full pop including elite
            # add elite to next gen
            # fill the rest of the next gen from mutated/crossover old pop
            self.record_gen(count+1)
            self.update_population()
            self.set_min_err()
            print('Gen: '+str(count)+' of '+str(self.gen_limit)+' Time: '+str((self.current_milli_time() - tm) /1000)+' Min Err: '+str(self.min_err))
        self.fitness.reset()

    def record_gen(self,count):
        self.plot_y.append(self.min_err)
        self.plot_x.append(count)

    def run_test_fold(self,beta):
        mse = self.fitness.pop_fitness('test',beta)
        return mse
        
    def set_min_err(self):
        if self.mse[0][1]< self.min_err:
            self.min_err = self.mse[0][1]

    def update_population(self):
        next_gen = []
        r.shuffle(self.beta)
        for i in range(self.pop_size):
            if i < len(self.elite_hold):
                next_gen.append(self.elite_hold[i])
            else:
                next_gen.append(self.beta[i])
        self.beta = next_gen
        self.elite_hold = []
        # update the fitness class with latest beta
        self.fitness.set_betas(self.beta)

    def stow_elite(self):
        for i in range(self.elite_rate):
            self.elite_hold.append(self.beta[i])

    def mutate_population(self):
        chromosome_index = [i for i in range(self.pop_size)]
        r.shuffle(chromosome_index)
        # mutate rows of betas
        for i in range(self.mutation_rate):
            self.beta[chromosome_index[i]] = self.mutate_chromosome(self.beta[chromosome_index[i]])

    # chromosome is a beta value array of matrices 
    def mutate_chromosome(self, chromosome):
        rows_to_change = int(r.random()*len(chromosome) )+1
        index = [i for i in range(len(chromosome))]
        r.shuffle(index)
        for i in range(rows_to_change):
            row_num = int(r.random()*len(chromosome[index[i]]))
            chromosome[index[i]][row_num] = self.mutate_row()
        return chromosome
        
    def mutate_row(self):
        row = np.array([self.rand(), self.rand(), self.rand()])
        return row

    def mutate_column(self):
        pass

    def mutate_element(self):
        pass

    def crossover_population(self):
        index = [i for i in range(self.pop_size)]
        r.shuffle(index)
        i = 0
        while i <= self.crossover_rate:
            self.beta[index[i]], self.beta[index[i+1]] = self.crossover_chromosome(self.beta[index[i]], self.beta[index[i+1]])
            i+=2
        
    def crossover_chromosome(self, chromo1, chromo2):
        matrices_to_crossover = int(r.random()*len(chromo1) )+1
        index = [i for i in range(len(chromo1))]
        r.shuffle(index)
        for i in range(matrices_to_crossover):
            chromo1[index[i]], chromo2[index[i]] = self.crossover(chromo1[index[i]],chromo2[index[i]])
        return chromo1, chromo2

    # trade rows between matracies of betas
    def crossover(self, mat1, mat2):
        rows_to_crossover = int(r.random()*len(mat1) )+1
        index = [i for i in range(len(mat1))]
        r.shuffle(index)
        for i in range(rows_to_crossover):
            temp = mat1[index[i]]
            mat1[index[i]] = mat2[index[i]]
            mat2[index[i]] = temp
        return np.matrix(mat1), np.matrix(mat2)

    def rand(self):
        #return  r.random() * r.randint(-sys.maxsize-1,sys.maxsize)
        return float("%.4f" %(r.random()*r.randint(-self.random_cap,self.random_cap)))


class Fitness:

    def __init__(self, pop, beta_len):
        self.pop_size = pop
        self.layer = [4,3,3,3,3]
        self.mse = self.init_mse()
        self.min_err = sys.maxsize
        self.data = Data(pop)
        self.test_err = 0
        self.beta_len = beta_len
        self.current_milli_time = lambda: int(round(time.time() * 1000))

    def init_mse(self):
        mse = []
        for i in range(self.pop_size):
            mse.append((i, sys.maxsize))
        return mse

    def set_betas(self, Beta):
        self.Beta = Beta
        
    def reset(self):
        self.mse = self.init_mse()
        self.min_err = sys.maxsize
    
    def pop_fitness(self, run_type, beta=None):
        if run_type == "train":
            x = self.data.train_x
            y = self.data.train_y
            for i in range(self.pop_size):
                self.find_fitness(x,y,i)
            self.sort_mse()
            self.sort_betas()
            self.min_err = self.mse[0][1]
            return self.Beta, self.mse
        else:
            x = self.data.test_x
            y = self.data.test_y
            mse = self.find_test_fitness(x,y,beta)
            return mse
        

    def find_fitness(self,data_x, data_y,beta_index):
        for i in range(len(data_x)):
            Z = np.array(data_x[i])
            Z = np.asarray(Z, dtype='float64')
            Z = np.transpose(np.matrix(np.append(Z,[1])))
            Yk = np.asarray(data_y[i], dtype='float64')
            for j in range(self.beta_len):
                B = np.matrix(np.asarray(self.Beta[beta_index][j], dtype='float64'))
                T = np.transpose(B) @ Z
                # T = self.sigmoid(T)
                if j+1 < self.beta_len:
                    T = self.sigmoid(T)
                    Z = np.append(T,[[1.0]], axis=0)
                else:
                    Z = self.sigmoid(T)
            
            self.test_err = self.test_err + self.get_error(Yk, Z)
        self.test_err = self.test_err/self.layer[-1]
        self.test_err = float("%.6f"% (self.test_err / len(data_x)))
        self.mse[beta_index] = (beta_index, self.test_err)

    def find_test_fitness(self,data_x, data_y,beta):
        test_err = 0
        for i in range(len(data_x)):
            Z = np.array(data_x[i])
            Z = np.asarray(Z, dtype='float64')
            Z = np.transpose(np.matrix(np.append(Z,[1])))
            Yk = np.asarray(data_y[i], dtype='float64')
            for j in range(self.beta_len):
                B = np.matrix(np.asarray(beta[j], dtype='float64'))
                T = np.transpose(B) @ Z
                if j+1 < self.beta_len:
                    T = self.sigmoid(T)
                    Z = np.append(T,[[1.0]], axis=0)
                else:
                    Z = self.sigmoid(T)
            test_err = test_err + self.get_error(Yk, Z)
        test_err = test_err/self.layer[-1]
        test_err = float("%.6f"% (test_err / len(data_x)))
        return test_err

    def sigmoid(self, T):
        for i in range(len(T)):
            T[i] =(1+math.exp(-float("%.6f"%T[i])))
        T = np.divide(1,T)
        return T
    
    def get_error(self, y, z):
        z = np.array(z)
        a = []
        for i in range(len(y)):
            a.append(y[i] - z[i])
        a = np.power(a,2)
        mse = np.sum(np.sum(a,0))
        return mse
            
    def sort_mse(self):
        count = 1
        while count > 0:
            count = 0
            for i in range(len(self.mse)-1):
                if self.mse[i][1] > self.mse[i+1][1]:
                    temp = self.mse[i]
                    self.mse[i] = self.mse[i+1]
                    self.mse[i+1] = temp
                    count +=1

    def sort_betas(self):
        temp = []
        for i in range(len(self.mse)):
            temp.append(self.Beta[self.mse[i][0]])
        self.Beta = temp



class Data:

    def __init__(self, pop):
        self.pop_size = pop
        self.KFCV_index = 0
        self.X, self.Y = self.load_data()
        self.KFCV(1)

    def load_data(self):
        X = self.load_file("X.txt")
        Y = self.load_file("Y.txt")
        return X,Y
    
    def load_file(self, file):
        hold = []
        fh = open(file,'r')
        line = fh.readline()
        while line:
            line = line.rstrip()
            arr = line.split(" ")
            hold.append(arr)
            line = fh.readline()
        fh.close()
        return hold
        
    # set training/test data per fold of k fold cross validation
    def KFCV(self,fold):
        self.train_x, self.test_x = self.KFCV_single(fold,self.X) 
        self.train_y, self.test_y = self.KFCV_single(fold,self.Y) 

    def KFCV_single(self,fold,argxy):
        train = []
        test= []
        count = 1
        for i in range(len(argxy)):
            if count != fold:
                train.append(argxy[i])
            else:
                test.append(argxy[i])
            count+=1
            if count == 11:
                count = 1
        self.train = np.matrix(train)  
        self.test = np.matrix(test) 
        return train, test 




temp = genetic_alg()
temp.run_alg()
