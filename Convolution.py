class CNN():
    def __init__(self, X, Y):
        new = np.zeros((10, Y.shape[0]))
        new[Y, range(Y.shape[0])] = 1
        self.X = X
        self.Y = new
        self.filters = []
        self.W = []
        self.B = []
        self.activation = []

    def create_mini_batch(self, size=64):
        indices = np.random.choice(self.X.shape[0], size=size, replace=False)
        self.X_B = self.X[indices]
        self.Y_B = self.Y[:, indices]

    def create_filter_layer(self, filter_dims=(5, 1, 3, 3)):
        if not self.filters:
            if self.X_B.shape[1] != filter_dims[1]:
                raise Exception("The filter dimensions do not match")

            self.filters.append(np.random.randn(*filter_dims) * 0.01)
        else:
            if self.filters[-1].shape[0] != filter_dims[1]:
                raise Exception("The filter dimensions do not match")

            self.filters.append(np.random.randn(*filter_dims) * 0.01)

    def calculate_input_size(self):
        if self.filters:
            starting_dims = self.X_B.shape[2]
            for i in self.filters:
                starting_dims -= i.shape[2]
                starting_dims += 1
            return starting_dims ** 2 * i.shape[0]
        else:
            raise Exception("There are no filters")

    def padding(self, Z, padding=2):
        return np.pad(Z, pad_width=((0, 0), (0, 0), (padding, padding), (padding, padding)))

    def fully_connected_layer(self, layer_depth=10, activation="ReLU"):
        if not self.W:
            self.W.append(np.random.randn(layer_depth, self.calculate_input_size()) * 0.01)
            self.B.append(np.random.randn(layer_depth, 1))
            self.activation.append(activation)
        else:
            self.W.append(np.random.randn(layer_depth, self.W[-1].shape[0]) * 0.01)
            self.B.append(np.random.randn(layer_depth, 1))
            self.activation.append(activation)

    def convolution(self, Z, f):
        N, C, H, W = Z.shape
        NS, CS, HS, WS = Z.strides
        C_out, _, f_K, f_K = f.shape
        C_outs, _, _, FWS = f.strides
        f = np.ascontiguousarray(as_strided(f, shape=(C * f_K ** 2, C_out), strides=(FWS, C_outs)))
        inner_dims = f_K * f_K * C
        A = as_strided(Z, shape=(N, H - f_K + 1, W - f_K + 1, C, f_K, f_K)
                       , strides=(NS, HS, WS, CS, HS, WS)).reshape(-1, inner_dims)
        out = A @ f
        S, E = out.strides
        out = as_strided(out, shape=(N, C_out, (H - f_K + 1) * (W - f_K + 1)),
                         strides=(E * (W - f_K + 1) * (H - f_K + 1) * C_out, E, S)).reshape(N, C_out, H - f_K + 1,
                                                                                            W - f_K + 1)
        return out

    def convolution_backward(self, Z, dZ, f):
        N, _, H, H = dZ.shape
        dNs, dCs, dHs, dHs = dZ.strides
        N, C_in, _, _ = Z.shape
        NS, CS, HS, WS = Z.strides
        C_out, f_C_in, f_K, f_K = f.shape
        inner_dims = H * H * N
        dZ = as_strided(dZ, shape=(C_out, N, H ** 2), strides=(dCs, dNs, dHs)).reshape(C_out, inner_dims)
        Z = as_strided(Z, shape=(C_in, f_K, f_K, N, H, H), strides=(CS, HS, WS, NS, HS, WS)).reshape(-1, inner_dims).T
        out = dZ @ Z
        return out.reshape(f.shape)

    def convolution_backward_Z(self, dZ, f):
        dZ = self.padding(dZ)
        N, C, H, W = dZ.shape
        NS, CS, HS, WS = dZ.strides
        f = np.rot90(f, k=2, axes=(2, 3))
        _, f_C_in, f_K, f_K = f.shape
        C_outs, _, _, FWS = f.strides
        inner_dims = f_K * f_K
        f = f.reshape(C, f_C_in, f_K ** 2)
        dZ = as_strided(dZ, shape=(C, N, H - 2, W - 2, f_K, f_K), strides=(CS, NS, HS, WS, HS, WS)).reshape(C, -1,
                                                                                                            inner_dims)
        out = np.zeros((N, f_C_in, H - 2, W - 2))
        for c in range(C):
            out += (dZ[c] @ f[c].T).reshape(N, f_C_in, H - 2, W - 2)
        return out.reshape(N, f_C_in, H - 2, W - 2)

    def activations(self, Z, activation):
        if activation == "ReLU":
            return Z * (Z > 0)
        if activation == "softmax":
            return np.exp(Z - np.max(Z, axis=0, keepdims=True) + 1e-8) / np.sum(
                np.exp(Z - np.max(Z, axis=0, keepdims=True) + 1e-8), axis=0, keepdims=True)

    def calculate_loss(self, Z):
        self.loss = -np.mean(np.sum(np.log(Z + 1e-8) * self.Y_B, axis=0))

    def forward(self):
        self.Z_filter = [0 for i in range(len(self.filters))]
        self.Z_filter[0] = self.convolution(self.X_B, self.filters[0])
        self.Z = [0 for i in range(len(self.W))]
        self.A = [0 for i in range(len(self.W))]
        for i in range(1, len(self.filters)):
            self.Z_filter[i] = self.convolution(self.Z_filter[i - 1], self.filters[i])
        self.Z[0] = self.W[0] @ self.Z_filter[i].reshape(self.calculate_input_size(), -1)
        self.A[0] = self.activations(self.Z[0], self.activation[0])
        for i in range(1, len(self.W)):
            self.Z[i] = self.W[i] @ self.Z[i - 1]
            self.A[i] = self.activations(self.Z[i], self.activation[i])
        self.calculate_loss(self.A[-1])

    def backward(self):
        num = len(self.W) - 1
        self.dW, self.dB, self.dfilter = [0 for i in range(len(self.W))], [0 for i in range(len(self.W))], [0 for i in
                                                                                                            range(
                                                                                                                len(self.W))]
        dZ = (self.A[num] - self.Y_B) / self.Y_B.shape[0]
        self.dW[num] = dZ @ self.A[num - 1].T
        self.dB[num] = np.sum(dZ, axis=1, keepdims=True)
        dZ = self.W[num].T @ dZ * (1 * (self.Z[num - 1] > 0))
        for i in range(num - 1, 0, -1):
            dW[i] = dZ @ self.A[i - 1]
            dB[i] = np.sum(dZ, axis=1, keepdims=True)
            dZ = self.W[i].T @ dZ * (1 * (self.Z[i - 1] > 0))
        self.dW[0] = dZ @ self.Z_filter[-1].reshape(self.calculate_input_size(), -1).T
        dZ = (self.W[0].T @ dZ).reshape(self.Z_filter[-1].shape)
        self.dfilter = [0 for i in range(len(self.filters))]
        for i in range(len(self.filters) - 1, 0, -1):
            self.dfilter[i] = self.convolution_backward(self.Z_filter[i - 1], dZ, self.filters[i])
            dZ = self.convolution_backward_Z(dZ, self.filters[i])
        self.dfilter[0] = self.convolution_backward(self.X_B, dZ, self.filters[0])

    def update(self, learning_rate=0.01):
        for i in range(len(self.filters)):
            self.filters[i] = self.filters[i] - learning_rate * self.dfilter[i]
        for i in range(len(self.W)):
            self.W[i] = self.W[i] - learning_rate * self.dW[i]
            self.B[i] = self.B[i] - learning_rate * self.dB[i]














