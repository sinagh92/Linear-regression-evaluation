import numpy as np
import time
import csv


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


def learn_weights_by_gradient_descent(samples, labels):
    """ Learn the weights by using gradient descent algorithm """

    w = [np.random.random_sample() for i in range(0, D+1)]
    T = 200
    eta = 0.000001

    for i in range(0, T):
        for j in range(0, D+1):
            w[j] = w[j] + eta/N * \
                (np.sum((labels - np.matmul(samples, w))*samples[:, j]))
    return w


def compute_loss(samples, w, labels, N):
    """ Compute the sum of squared error loss"""
    return np.sum(np.square(labels - np.matmul(samples, w)))/(2*N)


if __name__ == '__main__':

    time_start = time.clock()

    # Read the data from file
    lines = [line.rstrip('\n') for line in open('data_10k_100.tsv')]

    N = int(lines[0])  # Number of data points
    D = int(lines[1])  # Dimension of features

    samples, labels = extract_samples_and_lables(lines)

    weights = learn_weights_by_gradient_descent(samples, labels)

    loss = compute_loss(samples, weights, labels, N)
    print('Loss is : {}'.format(loss))

    write_weights_in_file('q2_data_10k_100.tsv', weights)

    time_elapsed = (time.clock() - time_start)
    print('Elapsed time is : {}'.format(time_elapsed))
