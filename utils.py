import csv
import numpy as np
import matplotlib.pyplot as plt
from models import *
from scipy import stats


def check_dir_make(save_dir):
    import os
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)


# This function initializes the csv logger.
def init_log(save_dir, header):
    with open(save_dir + '.csv', 'w+') as f:
        writer = csv.writer(f)
        writer.writerow(header)

# This function writes a message into the initialized csv file.


def write_log(save_dir, message):
    with open(save_dir + '.csv', 'a') as f:
        writer = csv.writer(f)
        writer.writerow(message)

# This function calculates the mean accuracy, max accuracy, the best epoch and the spearman correlation.


def calculate_stats(dir, save_name, rand=False):
    epochs, accs = [], []
    with open(dir + 'Logs/' + save_name + '/metrics_test.csv') as f:
        reader = csv.reader(f)
        next(reader, None)  # skip header
        for row in reader:
            epochs.append(int(row[0]))
            accs.append(float(row[1]))

    acc_mean = np.mean(accs)
    acc_std = np.std(accs)
    acc_max = np.max(accs)
    acc_min = np.min(accs)
    acc_argmin = np.argmin(accs)
    acc_argmax = np.argmax(accs)

    print(f'Acc: mean={acc_mean:.3f}, Std: std={acc_std:.3f}, max={acc_max:.3f} at epoch {acc_argmax+1}, min={acc_min:.3f} at epoch {acc_argmin+1}')

    if not rand:
        mean_losses = []
        with open(dir + 'Logs/' + save_name + '/loss.csv') as f:
            reader = csv.reader(f)
            next(reader, None)  # skip header
            for row in reader:
                mean_losses.append(float(row[2]))

        rho, p = stats.spearmanr(mean_losses, accs)
        print(f'Rho = {rho}, p = {p}')


# This function plots the training loss vs the test accuracy and saves it.
def plot(dir, save_name, start=0, stop=500, name=None):
    average_losses = []
    with open(dir + 'Logs/' + save_name + '/loss.csv') as f:
        reader = csv.reader(f)
        next(reader, None)  # skip header
        for row in reader:
            average_losses.append(float(row[2]))

    epochs, accs = [], []
    with open(dir + 'Logs/' + save_name + '/metrics_test.csv') as f:
        reader = csv.reader(f)
        next(reader, None)  # skip header
        for row in reader:
            epochs.append(int(row[0]))
            accs.append(float(row[1]))

    argmax = np.argmax(accs)

    plt.plot(epochs[start:stop], average_losses[start:stop], label='Mean Loss')
    plt.plot(epochs[start:stop], accs[start:stop], label='Test Accuracy')
    plt.vlines(argmax, ymin=0, ymax=1.5, colors='lightgrey')
    plt.legend(loc='upper right')
    plt.xlim(start, stop)
    plt.ylim(0, 1.5)
    plt.xlabel('Epochs')
    plt.title(name)

    plt.savefig(dir + 'Logs/' + save_name + '/plot.png', dpi=300)


# This function implements our random models.
# As our random models are not really models, we put it into utils.
# If the mode is all, it uniformly selects a label out of [0, 1, 2, 3, 4]
# If the mode is 02, it uniformly selects a label out of [0, 2]
# If the mode is distributed, it selects a label out of [0, 1, 2, 3, 4], however,
# this time the probabilities are dependent on the distribtuion of the labels.
def predict_randomly(mode='all'):
    _len = 103
    if mode == all:
        pred = np.random.randint(5, size=_len)
    elif mode == '02':
        pred = np.random.randint(2, size=_len) * 2
    elif mode == 'distributed':
        zeros = np.zeros(34)
        ones = np.ones(5)
        twos = np.ones(32) * 2
        thress = np.ones(19) * 3
        fours = np.ones(13) * 4
        pred = np.concatenate([zeros, ones, twos, thress, fours])
        np.random.shuffle(pred)

    return pred

# to encounter data imbalance 
def get_sampler(labels):

    from torch.utils.data import WeightedRandomSampler

    target = labels
    class_sample_count = np.array(
        [len(np.where(target == t)[0]) for t in np.unique(target)])
    weight = 1. / class_sample_count
    samples_weight = np.array([weight[t] for t in target])

    samples_weight = torch.from_numpy(samples_weight)
    samples_weigth = samples_weight.double()
    sampler = WeightedRandomSampler(samples_weight, len(samples_weight))

    return(sampler)


# mostly based on https://www.realpythonproject.com/understanding-accuracy-recall-precision-f1-scores-and-confusion-matrices/
# This function plots the confusion matrix and saves it.
def conf_plot(confusion, save_dir, name, ckpt):
    import seaborn as sns
    sns.heatmap(confusion, annot=True, xticklabels=[
                0, 1, 2, 3, 4], yticklabels=[0, 1, 2, 3, 4])
    plt.xlabel('Predicted')
    plt.ylabel('Label')
    plt.title(f'{name}, epoch {ckpt.split("e")[1]}')
    plt.savefig(save_dir + '.png', dpi=300)


def rocBinaryPlot(probs, labels, name):
    from sklearn.metrics import roc_curve
    from sklearn.metrics import roc_auc_score

    plt.plot([0, 1], [0, 1], linestyle='--', label='No Skill')
    fpr, tpr, _ = roc_curve(labels, probs)

    rAucScore = roc_auc_score(labels, probs)

    plt.plot(fpr, tpr, marker='.', label=name)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC AUC: '+str(rAucScore))
    plt.legend()
    plt.show()

    return rAucScore


def rocBinaryPlots(probsList, labels, nameList):
    from sklearn.metrics import roc_curve
    from sklearn.metrics import roc_auc_score

    lRAuc = []

    for probs, name in zip(probsList, nameList):

        fpr, tpr, _ = roc_curve(labels, probs)
        rAucScore = roc_auc_score(labels, probs)
        plt.plot(fpr, tpr, marker='.', label=name+' ('+str(rAucScore)+')')

        lRAuc.append(rAucScore)

    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc='lower right')
    plt.show()

    return lRAuc



def precRecallBinaryPlot(probs, labels, name):
    from sklearn.metrics import precision_recall_curve
    from sklearn.metrics import auc

    # calculate the no skill line as the proportion of the positive class
    no_skill = len(labels[labels == 1]) / len(labels)

    plt.plot([0, 1], [no_skill, no_skill], linestyle='--', label='No Skill')
    precision, recall, _ = precision_recall_curve(labels, probs)

    prAucScore = auc(recall, precision)

    plt.plot(recall, precision, marker='.', label=name)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precission Recall AUC: '+str(prAucScore))
    plt.legend()
    plt.show()

    return prAucScore


def precRecallBinaryPlots(probsList, labels, nameList):
    from sklearn.metrics import precision_recall_curve
    from sklearn.metrics import auc

    lPRAuc = []

    for probs, name in zip(probsList, nameList):
        precision, recall, _ = precision_recall_curve(labels, probs)
        prAucScore = auc(recall, precision)
        plt.plot(recall, precision, marker='.',
                 label=name+' ('+str(prAucScore)+')')

        lPRAuc.append(prAucScore)

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.legend(loc='lower right')
    plt.show()

    return lPRAuc


def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False


if __name__ == '__main__':
    # specify the directory
    dir = '..'
    save_name = 'resnet-e500-preTrained'
    rand = False
    name = 'ResNet18 with pretraining'

    #calculate_stats(dir, save_name, rand)
    plot(dir, save_name, 0, 500, name)
