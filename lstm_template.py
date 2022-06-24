"""
Minimal character-level LSTM model. Written by Ngoc Quan Pham
Code structure borrowed from the Vanilla RNN model from Andreij Karparthy @karparthy.
BSD License
"""
import numpy as np
from random import uniform
import sys


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def dsigmoid(y):
    return y * (1 - y)


def dtanh(x):
    return 1 - x * x


# The numerically stable softmax implementation
def softmax(x):
    # assuming x shape is [feature_size, batch_size]
    e_x = np.exp(x - np.max(x, axis = 0))
    return e_x / e_x.sum(axis = 0)


# data I/O
data = open('data/input.txt', 'r').read()  # should be simple plain text file
chars = sorted(list(set(data)))
data_size, vocab_size = len(data), len(chars)
print('data has %d characters, %d unique.' % (data_size, vocab_size))
char_to_ix = {ch: i for i, ch in enumerate(chars)}
ix_to_char = {i: ch for i, ch in enumerate(chars)}
std = 0.1

option = sys.argv[1]

# hyperparameters
emb_size = 16
hidden_size = 256  # size of hidden layer of neurons
seq_length = 128  # number of steps to unroll the RNN for
learning_rate = 5e-2
max_updates = 500000
batch_size = 32

concat_size = emb_size + hidden_size

# model parameters
# char embedding parameters
Wex = np.random.randn(emb_size, vocab_size) * std  # embedding layer

# LSTM parameters
Wf = np.random.randn(hidden_size, concat_size) * std  # forget gate
Wi = np.random.randn(hidden_size, concat_size) * std  # input gate
Wo = np.random.randn(hidden_size, concat_size) * std  # output gate
Wc = np.random.randn(hidden_size, concat_size) * std  # c term

bf = np.zeros((hidden_size, 1))  # forget bias
bi = np.zeros((hidden_size, 1))  # input bias
bo = np.zeros((hidden_size, 1))  # output bias
bc = np.zeros((hidden_size, 1))  # memory bias

# Output layer parameters
Why = np.random.randn(vocab_size, hidden_size) * std  # hidden to output
by = np.random.randn(vocab_size, 1) * std  # output bias

data_stream = np.asarray([char_to_ix[char] for char in data])
print(data_stream.shape)

bound = (data_stream.shape[0] // (seq_length * batch_size)) * (seq_length * batch_size)
cut_stream = data_stream[:bound]
cut_stream = np.reshape(cut_stream, (batch_size, -1))


def print_shape(str_arr, arr):
    print(f"Shape of {str_arr} is: {arr.shape}")


def forward(inputs, targets, memory):
    """
    inputs,targets are both list of integers.
    hprev is Hx1 array of initial hidden state
    returns the loss, gradients on model parameters, and last hidden state
    """
    hprev, cprev = memory
    # xs: inputs to the LSTM
    # wes: word embeddings of the LSTM
    # hs: hidden states
    # ys: output layers
    # ps: probability distributions
    # cs: cell memories
    # zs: input and hidden states concatenated
    # ins: input gates
    # c_s: candidate memory
    # ls: labels
    # os: output gate
    # fs: forget gate

    xs, wes, hs, ys, ps, cs, zs, ins, c_s, ls = {}, {}, {}, {}, {}, {}, {}, {}, {}, {}
    os, fs = {}, {}
    hs[-1] = np.copy(hprev)
    cs[-1] = np.copy(cprev)

    loss = 0
    input_length = inputs.shape[0]

    # forward pass
    # print("vocab_size: ", vocab_size)
    # print("batch_size: ", batch_size)
    for t in range(input_length):
        xs[t] = np.zeros((vocab_size, batch_size))  # encode in 1-of-k representation
        for b in range(batch_size):
            xs[t][inputs[t][b]][b] = 1

        # convert word indices to word embeddings
        wes[t] = np.dot(Wex, xs[t])

        # LSTM cell operation
        # first concatenate the input and h to get z
        zs[t] = np.row_stack((hs[t - 1], wes[t]))

        # compute the forget gate
        # f = sigmoid(Wf * z + bf)
        fs[t] = sigmoid(np.dot(Wf, zs[t]) + bf)

        # compute the input gate
        # i = sigmoid(Wi * z + bi)
        ins[t] = sigmoid(np.dot(Wi, zs[t]) + bi)

        # compute the candidate memory
        # c_ = tanh(Wc * z + bc)
        c_s[t] = np.tanh(np.dot(Wc, zs[t]) + bc)

        # new memory: applying forget gate on the previous memory
        # and then adding the input gate on the candidate memory
        # c_t = f * c_(t-1) + i * c_ib
        # cs[t] = np.multiply(fs[t], cs[t - 1]) + np.multiply(ins[t], c_s[t])
        cs[t] = np.multiply(fs[t], cs[t - 1]) + np.multiply(ins[t], c_s[t])

        # output gate
        # o = sigmoid(Wo * z + bo)
        os[t] = sigmoid(np.dot(Wo, zs[t]) + bo)
        hs[t] = np.multiply(os[t], np.tanh(cs[t]))

        # DONE LSTM

        # output layer - softmax and cross-entropy loss
        # unnormalized log probabilities for next chars
        # OLD: ys[t] = np.dot(Why, os[t]) + by
        ys[t] = np.dot(Why, hs[t]) + by
        # ys[t] = np.multiply(os[t], np.tanh(cs[t]))

        # softmax for probabilities for next chars
        ps[t] = softmax(ys[t])

        # label
        ls[t] = np.zeros((vocab_size, batch_size))
        for b in range(batch_size):
            ls[t][targets[t][b]][b] = 1

        # cross-entropy loss
        loss_t = np.sum(-np.log(ps[t]) * ls[t])
        loss += loss_t
        # loss += -np.log(ps[t][targets[t],0])

    # activations = ()
    activations = (xs, wes, hs, ys, ps, cs, zs, ins, c_s, ls, os, fs)
    memory = (hs[input_length - 1], cs[input_length - 1])

    return loss, activations, memory


def backward(activations, clipping = True):
    xs, wes, hs, ys, ps, cs, zs, ins, c_s, ls, os, fs = activations

    # backward pass: compute gradients going backwards-

    # xs: inputs to the LSTM
    # wes: word embeddings of the LSTM
    # hs: hidden states
    # ys: output layers
    # ps: probability distributions
    # cs: cell memories
    # zs: input and hidden states concatenated
    # ins: input gates
    # c_s: candidate memory
    # ls: labels

    # dWex: word embeddings matrix
    # dWhy: hidden state to output
    # dby: bias of output
    # dWf, dWi, dWc, dWo: Forget, input, candidate state aka g, output
    # dbf, dbi, dbc, dbo: biases of the gates
    # dhnext: h+1 state
    # dcnext: c+1 state

    # Here we allocate memory for the gradients
    dWex, dWhy = np.zeros_like(Wex), np.zeros_like(Why)
    dby = np.zeros_like(by)
    dWf, dWi, dWc, dWo = np.zeros_like(Wf), np.zeros_like(Wi), np.zeros_like(Wc), np.zeros_like(Wo)
    dbf, dbi, dbc, dbo = np.zeros_like(bf), np.zeros_like(bi), np.zeros_like(bc), np.zeros_like(bo)

    dhnext = np.zeros_like(hs[0])
    dcnext = np.zeros_like(cs[0])

    input_length = len(xs)
    #
    # print_shape("dby", dby)
    # print_shape("dWhy", dWhy)
    # print_shape("by", by)
    # print_shape("Why", Why)

    dy = np.zeros_like(ys[0])

    # back propagation through time starts here
    for t in reversed(range(input_length)):
        # computing the gradients here

        # zs[t]: 272x32
        # hs: 256x32
        # xs: 65x32
        # wes: 16x32
        # hs + ws = 256+16=272
        # ys: 65x32
        # os: 256x32
        # cs: 256x32
        # ins: 256x32
        # print_shape("zs[t]", zs[t])
        # print_shape("hs[t]", hs[t])
        # print_shape("xs[t]", xs[t])
        # print_shape("wes[t]", wes[t])
        # print_shape("ys[t]", ys[t])
        # print_shape("os[t]", os[t])
        # print_shape("cs[t]", cs[t])
        # print_shape("ins[t]", ins[t])
        # print_shape("dby", dby)
        # print_shape("dy", dy)
        # print_shape("c_s[t]", c_s[t])
        # print_shape("dbc", dbc)
        # print_shape("dbo", dbo)
        # print_shape("dbi", dbi)
        # print_shape("dbf", dbf)
        # print_shape("dWo", dWo)
        # print_shape("dWc", dWc)
        # print_shape("dWf", dWf)
        # print_shape("dWi", dWi)
        # print_shape("dWex", dWex)

        # dby, by: 65x1
        # dbo, dbi, dbc, dbf: 256x1
        # dWhy, Why: 65x256
        # dWi, dWo, dWc, dWf: 256x272

        # -- Output layer, loss function, softmax --
        # first gradient that is closest to the loss function, output softmax
        # dL/dy = dL/dp * dp/dy
        # dy: 65 x 32
        dy = ps[t] - ys[t]
        # print_shape("dy", dy)

        # Gradient of the bias of the output layer
        dby += np.sum(dy, axis = -1, keepdims = True)
        # print_shape("dby", dby)

        # Gradient of the weights of the output layer
        # dL/dWhy = dL/dp * dp/dy * dy/dWhy
        # dWhy: 65x256
        dWhy += np.dot(dy, hs[t].T)
        # print_shape("dWhy", dWhy)

        # -- Candidate state --

        # Gradient of the bias of the candidate state
        # Pre-calculations dL/dbc

        # dh/dcs = dh_dcs mit 256x32
        dh_dc = np.multiply(os[t], dtanh(cs[t]))

        # dh/d_c_s = dh/dc * dcs/dc_s
        # (256x32)
        dh_dc_s = np.multiply(dh_dc, ins[t])

        # dh/dbc = dh/dc * dcs/dc_s * dc_s/dbc
        # (256x32)
        dh_dbc = np.multiply(dh_dc_s, dtanh(c_s[t]))

        # dL/dh= dL/dp * dp/dy * dy/dh
        # (256x32)
        dh = np.dot(Why.T, dy)

        # Result dL/dbc
        # dL/dbc = dL/dp * dp/dy * dy/dh * dh/dc * dc/dc_s * dc_s/dbc
        # dbc_ (256, 32)
        # dbc (256, 1)
        dbc_ = np.multiply(dh, dh_dbc)
        dbc += np.sum(dbc_, axis = -1, keepdims = True)
        # print_shape("dbc_", dbc_)
        # print_shape("dbc", dbc)

        # Gradient of the weights of the candidate state
        # Pre-calculations dL/dWc

        # dc_s/dWc
        # (32x272)
        dc_s_dWc = np.dot(zs[t], dtanh(c_s[t]).T)  # UMSTRITTEN
        # print_shape("dc_s_dWc", dc_s_dWc)

        # dc/dWc = dc/dc_s * dc_s/dWc
        # 32x272
        dc_dWc = np.dot(ins[t].T, dc_s_dWc.T)  # UMSTRITTEN
        print_shape("dc_dWc", dc_dWc)
        print_shape("dWc", dWc)

        # dL/dc_s = dL/dp * dp/dy * dy/dh * dh/dc * dc/dc_s
        # (256x32)
        dc_s = np.multiply(dh, dh_dc_s)
        print_shape("OKEE dc_s", dc_s)
        print_shape("c_s", c_s[t])
        raise SystemError
        # Result dL/dWc
        # dL/dWc = dL/dp * dp/dy * dy/dh * dh/dc * dc/dc_s * dc_s/dWc
        dWc = np.dot(dc_s, dc_dWc)
        # print_shape("dWc", dWc)

        # dbo = np.dot(hs[t].T, np.tanh(cs[t]))

        # -- Input Gate --

        # Gradient of the bias of the input gate
        # Pre-calculations dL/dbi

        # dc/dbi = dc/di * di/dbi
        # (256x32)
        dc_dbi = np.multiply(c_s[t], dsigmoid(ins[t]))

        # dh/dbi = dh/dc * dc/di * di/dbi
        # (256x32)
        dh_dbi = np.multiply(dh_dc, dc_dbi)

        # Result dL/dbi
        # dL/dbi = dL/dp * dp/dy * dy/dh * dh/dc * dc/di * di/dbi
        # dbi_ 256x32
        # dbi 256x1
        dbi_ = np.multiply(dh, dh_dbi)
        dbi += np.sum(dbi_, axis = -1, keepdims = True)
        # print_shape("dbi_", dbi_)
        # print_shape("dbi", dbi)

        # Gradient of the weights of the input gate
        # Pre-calculations dL/dWi

        # di/dWi
        # (32x272)
        di_dWi = np.dot(zs[t], dsigmoid(ins[t]).T)  # UMSTRITTEN
        # print_shape("di_dWi", di_dWi)

        # dc/dWi = dc/di * di/dWi
        # 32x272
        dc_dWi = np.dot(c_s[t].T, di_dWi.T)  # UMSTRITTEN

        # dL/dc = dL/dp * dp/dy * dy/dh * dh/dc
        # 256x32
        dc = np.multiply(dh, dh_dc)

        # Result dL/dWi
        # dL/dWi = dL/dp * dy/dp * dy/dh * dh/dc * dc/di * di/dWi
        dWi = np.dot(dc, dc_dWi)
        # print_shape("dWi", dWi)

        # -- Forget gate --
        # Gradient of the bias of the forget gate
        # Pre calculations dL/dbf

        # dc/dbf = dc/df * df/dbf
        # (256x32)
        dc_dbf = np.multiply(cs[t - 1], dsigmoid(fs[t]))

        # Result dL/dbf
        # dL/dbf = dL/dp * dp/dy * dy/dh * dh/dc * dc/df * df/dbf
        dbf_ = np.multiply(dc, dc_dbf)
        dbf += np.sum(dbf_, axis = -1, keepdims = True)
        # print_shape("dbf", dbf)

        # Gradient of the weights of the forget gate
        # Pre calculations dL/dWf

        # df/dWf
        # (32x272)
        df_dWf = np.dot(zs[t], dsigmoid(fs[t]).T)  # UMSTRITTEN
        # print_shape("df_dWf", df_dWf)

        # dc/dWf = dc/di * df/dWf
        # 32x272
        dc_dWf = np.dot(cs[t - 1].T, df_dWf.T)  # UMSTRITTEN
        # print_shape("dc_dWf", dc_dWf)

        # Result dL/dWf
        # dL/dWf = dL/dp * dp/dy * dy/dh * dh/dc * dc/df * df/dWf
        dWf = np.dot(dc, dc_dWf)
        # print_shape("dWf", dWf)

        # -- Output gate --
        # Gradient of the bias of the output gate
        # Pre calculations dL/dbo

        # dh/dbo = dh/do * do/dbo
        # (256x32)
        dh_dbo = np.multiply(np.tanh(cs[t]), dsigmoid(os[t]))

        # Results dL/dbo
        # dL/dbo = dL/dp * dp/dy * dy/dh * dh/do * do/dbo
        dbo_ = np.multiply(dh, dh_dbo)
        dbo += np.sum(dbo_, axis = -1, keepdims = True)
        # print_shape("dbo_", dbo_)
        # print_shape("dbo", dbo)

        # Gradient of the weights of the output gate
        # Pre calculations dL/dWo

        # do/dWo
        # (272x256)
        do_dWo = np.dot(zs[t], dsigmoid(os[t]).T)

        # dL/do = dL/dp * dp/dy * dy/dh * dh/do
        # (256x32)
        do = np.dot(dh, np.tanh(cs[t]).T)

        # Results dL/dWo
        # dL/dWo = dL/dp * dp/dy * dy/dh * dh/do * do/dWo
        # print_shape("do_dWo", do_dWo)
        # print_shape("do", do)

        dWo = np.dot(do, do_dWo.T)
        # print_shape("dWo okay okay", dWo)

        # -- Pre input layer
        # Gradient of the word embeddings
        # convert word indices to word embeddings
        # dL/dWex = dL/dp * dp/dy * dy/dh * dh/do * do/dz * dz/dWes
        # 16x65

        # do_dz = np.multiply()
        # wes[t] = np.dot(Wex, xs[t])
        # dWex += np.dot(xs[t], )
        dWex += 0

    # clip to mitigate exploding gradients
    if clipping:
        for dparam in [dWex, dWf, dWi, dWo, dWc, dbf, dbi, dbo, dbc, dWhy, dby]:
            np.clip(dparam, -5, 5, out = dparam)

    gradients = (dWex, dWf, dWi, dWo, dWc, dbf, dbi, dbo, dbc, dWhy, dby)

    return gradients


def sample(memory, seed_ix, n):
    """
  sample a sequence of integers from the model
  h is memory state, seed_ix is seed letter for first time step
  """
    h, c = memory
    x = np.zeros((vocab_size, 1))
    x[seed_ix] = 1
    ixes = []
    for t in range(n):
        # @todo
        # forward pass again, but we do not have to store the activations now
        loss, _, memory = forward(seed_ix, memory)
        p = np.exp(y) / np.sum(np.exp(y))
        ix = np.random.choice(range(vocab_size), p = p.ravel())

        index = ix
        x = np.zeros((vocab_size, 1))
        x[index] = 1
        ixes.append(index)
    return ixes


if option == 'train':

    n, p = 0, 0
    n_updates = 0

    # momentum variables for Adagrad
    mWex, mWhy = np.zeros_like(Wex), np.zeros_like(Why)
    mby = np.zeros_like(by)

    mWf, mWi, mWo, mWc = np.zeros_like(Wf), np.zeros_like(Wi), np.zeros_like(Wo), np.zeros_like(Wc)
    mbf, mbi, mbo, mbc = np.zeros_like(bf), np.zeros_like(bi), np.zeros_like(bo), np.zeros_like(bc)

    smooth_loss = -np.log(1.0 / vocab_size) * seq_length  # loss at iteration 0

    data_length = cut_stream.shape[1]

    while True:
        # prepare inputs (we're sweeping from left to right in steps seq_length long)
        if p + seq_length + 1 >= data_length or n == 0:
            hprev = np.zeros((hidden_size, batch_size))  # reset RNN memory
            cprev = np.zeros((hidden_size, batch_size))
            p = 0  # go from start of data

        inputs = cut_stream[:, p:p + seq_length].T
        targets = cut_stream[:, p + 1:p + 1 + seq_length].T

        # # sample from the model now and then
        # if n % 200 == 0:
        #     h_zero = np.zeros((hidden_size, 1))  # reset RNN memory
        #     c_zero = np.zeros((hidden_size, 1))
        #     sample_ix = sample((h_zero, c_zero), inputs[0][0], 2000)
        #     txt = ''.join(ix_to_char[ix] for ix in sample_ix)
        #     print('----\n %s \n----' % (txt,))

        # forward seq_length characters through the net and fetch gradient
        loss, activations, memory = forward(inputs, targets, (hprev, cprev))
        hprev, cprev = memory
        gradients = backward(activations)

        dWex, dWf, dWi, dWo, dWc, dbf, dbi, dbo, dbc, dWhy, dby = gradients
        smooth_loss = smooth_loss * 0.999 + loss / batch_size * 0.001
        if n % 20 == 0:
            print('iter %d, loss: %f' % (n, smooth_loss))  # print progress

        # perform parameter update with Adagrad
        for param, dparam, mem in zip([Wf, Wi, Wo, Wc, bf, bi, bo, bc, Wex, Why, by],
                                      [dWf, dWi, dWo, dWc, dbf, dbi, dbo, dbc, dWex, dWhy, dby],
                                      [mWf, mWi, mWo, mWc, mbf, mbi, mbo, mbc, mWex, mWhy, mby]):
            mem += dparam * dparam
            param += -learning_rate * dparam / np.sqrt(mem + 1e-8)  # adagrad update

        p += seq_length  # move data pointer
        n += 1  # iteration counter
        n_updates += 1
        if n_updates >= max_updates:
            break

elif option == 'gradcheck':

    data_length = cut_stream.shape[1]

    p = 0
    # inputs = [char_to_ix[ch] for ch in data[p:p + seq_length]]
    # targets = [char_to_ix[ch] for ch in data[p + 1:p + seq_length + 1]]
    inputs = cut_stream[:, p:p + seq_length].T
    targets = cut_stream[:, p + 1:p + 1 + seq_length].T

    delta = 0.0001

    hprev = np.zeros((hidden_size, batch_size))
    cprev = np.zeros((hidden_size, batch_size))

    memory = (hprev, cprev)

    loss, activations, hprev = forward(inputs, targets, memory)
    gradients = backward(activations, clipping = False)
    dWex, dWf, dWi, dWo, dWc, dbf, dbi, dbo, dbc, dWhy, dby = gradients

    for weight, grad, name in zip([Wf, Wi, Wo, Wc, bf, bi, bo, bc, Wex, Why, by],
                                  [dWf, dWi, dWo, dWc, dbf, dbi, dbo, dbc, dWex, dWhy, dby],
                                  ['Wf', 'Wi', 'Wo', 'Wc', 'bf', 'bi', 'bo', 'bc', 'Wex', 'Why', 'by']):

        str_ = ("Dimensions dont match between weight and gradient %s and %s." % (weight.shape, grad.shape))
        assert (weight.shape == grad.shape), str_

        print(name)
        countidx = 0
        gradnumsum = 0
        gradanasum = 0
        relerrorsum = 0
        erroridx = []

        for i in range(weight.size):

            # evaluate cost at [x + delta] and [x - delta]
            w = weight.flat[i]
            weight.flat[i] = w + delta
            loss_positive, _, _ = forward(inputs, targets, memory)
            weight.flat[i] = w - delta
            loss_negative, _, _ = forward(inputs, targets, memory)
            weight.flat[i] = w  # reset old value for this parameter
            # fetch both numerical and analytic gradient
            grad_analytic = grad.flat[i]
            grad_numerical = (loss_positive - loss_negative) / (2 * delta)
            gradnumsum += grad_numerical
            gradanasum += grad_analytic
            rel_error = abs(grad_analytic - grad_numerical) / abs(grad_numerical + grad_analytic)
            if rel_error is None:
                rel_error = 0.
            relerrorsum += rel_error

            if rel_error > 0.001:
                print('WARNING %f, %f => %e ' % (grad_numerical, grad_analytic, rel_error))
                countidx += 1
                erroridx.append(i)

        print('For %s found %i bad gradients; with %i total parameters in the vector/matrix!' % (
            name, countidx, weight.size))
        print(' Average numerical grad: %0.9f \n Average analytical grad: %0.9f \n Average relative grad: %0.9f' % (
            gradnumsum / float(weight.size), gradanasum / float(weight.size), relerrorsum / float(weight.size)))
        print(' Indizes at which analytical gradient does not match numerical:', erroridx)
