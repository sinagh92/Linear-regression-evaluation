import numpy as np
import time
import csv
import random


def write_weights_in_file(file, w):
    """ write the output weights in file"""
    output_labels = ['w'+str(i) for i in range(1, D+1)]
    output_labels.append('w0')

    with open(file, 'wt') as out_file:
        tsv_writer = csv.writer(out_file, delimiter='\t')
        tsv_writer.writerow(output_labels)
        tsv_writer.writerow(w)


def extract_samples_and_lables(lines):
    """ Extract samples and labels from each line of the file"""
    labels = np.zeros(N)
    samples = np.zeros((N, D+1))
    sample = np.zeros(D+1)

    for i in range(3, len(lines)):
        f = lines[i].rstrip('\t').split()
        labels[i-3] = float(f[0])

        for j in range(0, D):
            sample[j] = float(f[j+1])
        sample[D] = 1
        samples[i-3] = sample
    return samples, labels


def learn_weights_by_sgd(samples, labels):
    """ Learn the weights by using stochastic gradient descent algorithm """
    w = [np.random.random_sample() for i in range(0, D+1)]
    T = 12
    eta = 0.0000001
    loss = 100
    while loss >= 70:
        for i in range(0, T):
            N_random = list(range(N))
            random.shuffle(N_random)
            for k in N_random:
                w = w + eta * \
                    ((labels[k] - np.matmul(samples[k], w)) * samples[k])
        loss = np.sum(np.square(labels - np.matmul(samples, w)))/(2*N)
    return w


def compute_loss(samples, w, labels, N):
    """ Compute the sum of squared error loss"""
    return np.sum(np.square(labels - np.matmul(samples, w)))/(2*N)


if __name__ == '__main__':
    # Start the timer
    time_start = time.clock()

    # Read the data from file
    lines = [line.rstrip('\n') for line in open('data_100k_300.tsv')]

    N = int(lines[0])  # Number of data points
    D = int(lines[1])  # Dimension of features

    samples, labels = extract_samples_and_lables(lines)

    weights = learn_weights_by_sgd(samples, labels)

    loss = compute_loss(samples, weights, labels, N)
    print('Loss is : {}'.format(loss))

    write_weights_in_file('q3_data_100k_300.tsv', weights)

    time_elapsed = (time.clock() - time_start)
    print('Elapsed time is : {}'.format(time_elapsed))
