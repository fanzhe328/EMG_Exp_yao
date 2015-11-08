import numpy as np
# from data_load import load_dataset1
import random


def proportion_simu(train, target, proportion=1.0):
    train_porportion = train * proportion
    train_simu = np.concatenate((train, train_porportion), axis=0)
    target_simu = np.concatenate((target, target), axis=0)
    return train_simu, target_simu


if __name__ == '__main__':
    # train, target = load_dataset1()
    # train_simu, target_simu = proportion_simu(train[1], target, 0.9)
    # print train_simu.shape, target_simu.shape
    iteration = 10
    while(iteration > 0):
        print random.randrange(1, 51)
        print random.randrange(1, 51)
        print random.randrange(1, 51)
        print random.randrange(1, 51)
        print random.randrange(1, 51)
        print random.randrange(1, 51)
        iteration -= 1
