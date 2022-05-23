import numpy as np
import random
import copy
import mld_metrics

from sklearn.datasets import make_multilabel_classification

def calculate_distance(a, b):
    return np.linalg.norm(a - b)


def distances_one_all(sample_idx, elements_idxs, x):
    distances = []
    for elem_idx in elements_idxs:
        distances.append((elem_idx, calculate_distance(x[sample_idx, :], x[elem_idx, :])))

    return distances

def new_sample(sample, ref_neighbor, neighbors, x, y):
    synth_sample = np.zeros(x.shape[1])

    for feature_idx in range(len(synth_sample)):
        # missing to add when feature is not numeric. In that case it should
        # put the most frequent value in the neighbors
        diff = x[ref_neighbor, feature_idx] - x[sample, feature_idx]
        offset = diff * random.random()
        value = x[sample, feature_idx] + offset

        synth_sample[feature_idx] = value

    labels_counts = y[sample, :]
    labels_counts = np.add(labels_counts, np.sum(y[neighbors, :], axis=0)) # todo 这里？
    labels = labels_counts > (len(neighbors) + 1) / 2
    ori_label = y[sample]
    return synth_sample, labels, ori_label

def sortSamples(v):
    return v[1]

def SCUMBLE(x, y, k):
    mean_ir = mld_metrics.mean_ir(y)

    y_new = copy.deepcopy(y)
    x_new = copy.deepcopy(x)
    y_new_ori = copy.deepcopy(y)
    min_set = set()
    label_IRLbl = []
    scumble_sample = []
    for label in range(y.shape[1]): # 所有可能的 样本类别
        ir_label = mld_metrics.ir_per_label(label, y) # 最多类别数量 / 每个类别数量
        label_IRLbl.append(ir_label)
    scumbleD = []
    for sample_idx in range(y.shape[0]):
        this_label = y[sample_idx]
        IRLblil = 1.
        count = 0.
        IRLbli = []
        for label in range(len(this_label)):
            if this_label[label] == 1:
                count += 1.0
                IRLblil *= label_IRLbl[label]
                IRLbli.append(label_IRLbl[label])
        if count == 0.:
            print (this_label)
        IRLblil_pow = pow(IRLblil, 1/count)
        scumble = 1 - (1 / np.average(IRLblil)) * IRLblil_pow
        scumble_sample.append(scumble)
        z = 1
        scumbleD.append(scumble)


    scumbleD = np.average(scumbleD)
    return scumble_sample, scumbleD

def REMEDIAL(x,y):
    mean_ir = mld_metrics.mean_ir(y)
    label_IRLbl = []
    scumble_sample = []
    for label in range(y.shape[1]):  # 所有可能的 样本类别
        ir_label = mld_metrics.ir_per_label(label, y)  # 最多类别数量 / 每个类别数量
        label_IRLbl.append(ir_label)
    IRMean = np.mean(label_IRLbl)
    scumbleD = []
    for sample_idx in range(y.shape[0]):
        this_label = y[sample_idx]
        IRLblil = 1.
        count = 0.
        IRLbli = []
        for label in range(len(this_label)):
            if this_label[label] == 1:
                count += 1.0
                IRLblil *= label_IRLbl[label]
                IRLbli.append(label_IRLbl[label])
        if count == 0.:
            print(this_label)
        IRLblil_pow = pow(IRLblil, 1 / count)
        scumble = 1 - (1 / np.average(IRLblil)) * IRLblil_pow
        scumble_sample.append(scumble)
        z = 1
        scumbleD.append(scumble)

    scumbleD = np.average(scumbleD)

    y_new = []
    x_new = []

    for sample_idx in range(y.shape[0]):
        thix_x = x[sample_idx]
        this_y = y[sample_idx]
        if scumble_sample[sample_idx] > scumbleD:
            y_min = copy.deepcopy(this_y)
            y_max = copy.deepcopy(this_y)
            for label_idx in range(y.shape[1]):
                if label_IRLbl[label_idx] <= mean_ir:
                    y_min[label_idx] = 0
                else:
                    y_max[label_idx] = 0
            x_new.append(thix_x)
            x_new.append(thix_x)
            y_new.append(y_min)
            y_new.append(y_max)

        else:
            x_new.append(thix_x)
            y_new.append(this_y)
    return np.array(x_new), np.array(y_new)



from sklearn.metrics.pairwise import pairwise_distances


# Example of usage

x, y = make_multilabel_classification(n_samples=1000, n_features=10, n_classes=8, allow_unlabeled =False)
print('Original samples generated (count): ')
print(y.shape[0])

print('Positive samples per class:')
print(np.sum(y, axis=0))

# Send the labels and the percentage to delete
#scumble_sample, scumbleD= SCUMBLE(x, y, 5)
x_new, y_new = REMEDIAL(x,y)
print('Synthetic samples generated (count): ')
print(y_new.shape[0])

z = 1
