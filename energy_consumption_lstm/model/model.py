from .utils import sigmoid, tanh
import numpy as np


class LSTM:
    def __init__(self, input_shape, output_shape):
        self.input_shape = input_shape
        self.layer_1 = self.Unit(input_shape, output_shape)

    def forward(self, x=.34):
        print(self.unit(self.cell_state, self.output_gate, x))

    class Unit:
        def __init__(self, input_shape, output_shape):
            self.forget_weights = np.random.random((input_shape, output_shape))
            self.cell_state = np.random.random((input_shape))
            self.output_gate = np.random.random(output_shape)

        def forward(self, x):
            np.zeros_like

        def unit(self, cell_state, output_gate, input_gate):
            dot_prod = np.dot(self.forget_weights, np.dot(output_gate, input_gate))
            f_t = sigmoid(dot_prod)
            cell_state = cell_state * f_t
            return cell_state


class ClassicRNN:
    def __init__(self, input_shape, output_shape, hidden_dim=10, bptt_truncate=5, learning_rate=0.0001,
                 min_clip_value=-1, max_clip_value=1):
        self.hidden_dim = hidden_dim
        self.bptt_truncate = bptt_truncate
        self.learning_rate = learning_rate
        self.min_clip_value = min_clip_value
        self.max_clip_value = max_clip_value
        self.input_shape = input_shape
        self.input_weights = np.random.uniform(0, 1, (self.hidden_dim, self.input_shape))
        self.hidden_weights = np.random.uniform(0, 1, (self.hidden_dim, self.hidden_dim))
        self.output_weights = np.random.uniform(0, 1, (output_shape, self.hidden_dim))
        self.output = np.zeros((self.hidden_dim, output_shape))
        self.layer_1 = self.Unit(input_shape, output_shape)

    def forward(self, x, y):
        for i in range(self.input_shape):
            new_input = np.zeros(x.shape)
            new_input[i] = x[i]
            input_prod = np.dot(self.input_weights, new_input)
            previous_output_prod = np.dot(self.hidden_weights, self.output)
            prod = input_prod + previous_output_prod
            out = sigmoid(prod)
            mulv = np.dot(self.output_weights, out)
            self.output = out
        loss = (y - mulv) ** 2 / 2
        return loss, mulv

    def forward_backward(self, x, y):
        layers = []
        dU = np.zeros(self.input_weights.shape)
        dV = np.zeros(self.output_weights.shape)
        dW = np.zeros(self.hidden_weights.shape)

        dU_t = np.zeros(self.input_weights.shape)
        dV_t = np.zeros(self.output_weights.shape)
        dW_t = np.zeros(self.hidden_weights.shape)

        dU_i = np.zeros(self.input_weights.shape)
        dW_i = np.zeros(self.hidden_weights.shape)
        for i in range(self.input_shape):
            new_input = np.zeros(x.shape)
            new_input[i] = x[i]
            input_prod = np.dot(self.input_weights, new_input)
            previous_output_prod = np.dot(self.hidden_weights, self.output)
            prod = input_prod + previous_output_prod
            out = sigmoid(prod)
            mulv = np.dot(self.output_weights, out)
            layers.append({'out': out, 'prev_out': self.output})
            self.output = out

        dmulv = (mulv - y)

        # backward pass
        for t in range(self.input_shape):
            dV_t = np.dot(dmulv, np.transpose(layers[t]['out']))
            dsv = np.dot(np.transpose(self.output_weights), dmulv)

            ds = dsv
            dadd = prod * (1 - prod) * ds

            dmulw = dadd * np.ones_like(previous_output_prod)

            dprev_s = np.dot(np.transpose(self.hidden_weights), dmulw)

            for i in range(t - 1, max(-1, t - self.bptt_truncate - 1), -1):
                ds = dsv + dprev_s
                dadd = prod * (1 - prod) * ds

                dmulw = dadd * np.ones_like(previous_output_prod)
                dmulu = dadd * np.ones_like(input_prod)

                dW_i = np.dot(self.hidden_weights, layers[t]['prev_out'])
                dprev_s = np.dot(np.transpose(self.hidden_weights), dmulw)

                new_input = np.zeros(x.shape)
                new_input[t] = x[t]
                dU_i = np.dot(self.input_weights, new_input)
                dx = np.dot(np.transpose(self.input_weights), dmulu)

                dU_t += dU_i
                dW_t += dW_i

            dV += dV_t
            dU += dU_t
            dW += dW_t

            if dU.max() > self.max_clip_value:
                dU[dU > self.max_clip_value] = self.max_clip_value
            if dV.max() > self.max_clip_value:
                dV[dV > self.max_clip_value] = self.max_clip_value
            if dW.max() > self.max_clip_value:
                dW[dW > self.max_clip_value] = self.max_clip_value

            if dU.min() < self.min_clip_value:
                dU[dU < self.min_clip_value] = self.min_clip_value
            if dV.min() < self.min_clip_value:
                dV[dV < self.min_clip_value] = self.min_clip_value
            if dW.min() < self.min_clip_value:
                dW[dW < self.min_clip_value] = self.min_clip_value

                # update
            self.input_weights -= self.learning_rate * dU
            self.output_weights -= self.learning_rate * dV
            self.hidden_weights -= self.learning_rate * dW

        loss = (y - mulv) ** 2 / 2
        return loss

    class Unit:
        def __init__(self, input_shape, output_shape):
            self.forget_weights = np.random.random((input_shape, output_shape))
            self.output_gate = np.random.random(output_shape)

        def forward(self, x):
            self.output_gate = self.unit(self.output_gate, x)
            return self.output_gate

        def unit(self, output_gate, input_gate):
            dot_prod = np.dot(self.forget_weights, np.dot(output_gate, input_gate))
            return tanh(dot_prod)
