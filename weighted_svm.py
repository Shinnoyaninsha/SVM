import numpy as np

class WeightedSVM:
    def __init__(self, kernel=None, learning_rate=0.1,n_iters=1000, c=0.01, degree=2, bias=1, gamma=0.1,weights=False):
        self.w = None #vecteur de poids
        self.learning_rate = learning_rate
        self.c = c
        self.gamma = gamma
        self.degree = degree        
        self.kernel = kernel
        self.b = bias if self.kernel else 0
        self.n_iters = n_iters
        self.alpha = np.array([])
        self.use_weights = weights
    
    def check_nb_labels(self, y):
        """
        Vérifier le nombre de labels à prédire pour choisir la méthode de prédiction.
        """
        _len = len(set(y))
        if _len == 2:
            self.labels = "binary"
        elif _len > 2 and self.multi in ['ovo', 'ovr']:
            self.labels = self.multi
        elif _len == 1:
            raise ValueError("Valeur unique dans la colonne cible")
        else:
            raise ValueError("Gestion des labels multiples non reconnues")
            
    def kernel_function(self, x1, x2):
        if self.kernel in ["linear", None]:
            return np.dot((self.w * x1), x2.T) + self.b
        if self.kernel == "poly":
            return ((np.dot(self.w * x1, x2.T) + self.b)) ** self.degree
        if self.kernel == "rbf":
            x1_2 = np.sum((x1 * self.w) ** 2, axis=1, keepdims=True)
            x2_2 = np.sum((x2 * self.w) ** 2, axis=1, keepdims=True).T
            cross_term = np.dot((x1 * self.w), (x2 * self.w).T)
            return np.exp(-self.gamma * (x1_2 - 2 * cross_term + x2_2))
        if self.kernel == "sigmoid":
            return np.tanh(self.gamma * np.dot((self.w * x1), x2.T) + self.b)
        raise ValueError(f"Noyau {self.kernel} non reconnu")

    
    def partial_derivative_w(self, w_idx, x1, x2):
        """
        Retourne la dérivée partielle de la fonction Kernel par rapport à un poids.
        """
        if self.kernel == "linear":
            return x1[w_idx] * x2[w_idx]
        elif self.kernel == "poly":
            return x1[w_idx] * x2[w_idx] * self.degree * (self.b+self.gamma * np.dot(self.w * x1, x2))**(self.degree-1)
        elif self.kernel == "rbf":
            return -self.gamma *(x1[w_idx]-x2[w_idx])**2 - self.kernel_function(x1,x2)
        elif self.kernel == "sigmoid":
            return (1-np.tanh(self.gamma * np.dot(self.w * x1, x2) + self.b)**2) * self.gamma * x1[w_idx] * x2[w_idx]
        
    def partial_derivative_vectorize(self, X):
        if self.kernel == "linear":
            return X[:,:,np.newaxis] * X[:, np.newaxis,:]
        elif self.kernel == "poly":
            return X[:,:,np.newaxis] * X[:, np.newaxis,:] * self.degree * (self.b + self.gamma * np.dot(X * self.w, X.T))**(self.degree-1)
        elif self.kernel == "rbf":
            delta = X[:, :, np.newaxis] - X[:, np.newaxis, :]
            return (-self.gamma) * (delta**2) - self.kernel_function(X, X)
        elif self.kernel == "sigmoid":
            dot_product = np.dot(X * self.w, X.T)
            tanh_term = np.tanh(self.gamma * dot_product + self.b)
            return (1 - tanh_term**2) * self.gamma * X[:, :, np.newaxis] * X[:, np.newaxis, :]
        
    def partial_derivative_wrt_weights(self, X):
        """
        Compute the partial derivative of the kernel with respect to the weight vector w,
        assuming X is transformed by element-wise multiplication with w inside the kernel.
        """
        Xw = X * self.w  # Apply weight vector inside the kernel

        if self.kernel == "linear":
            # ∂/∂w of (Xw @ Xw.T) = X * X
            return X[:, np.newaxis, :] * X[np.newaxis, :, :] 

        elif self.kernel == "poly":
            # K = (b + gamma * Xw @ Xw.T)^d
            dot = np.dot(Xw, X.T)
            base = self.b + self.gamma * dot
            power_term = self.degree * (base ** (self.degree - 1))

            # ∂/∂w of (Xw @ Xw.T) = X * X
            return power_term[:, :, np.newaxis] * self.gamma * X[np.newaxis, :, :] * X[:, np.newaxis, :]

        elif self.kernel == "rbf":
            # Elementwise-weighted input
            Xw = X * self.w  # shape: (n_samples, n_features)

            # Compute squared distances
            Xw_i = Xw[:, np.newaxis, :]  # shape: (n, 1, f)
            Xw_j = Xw[np.newaxis, :, :]  # shape: (1, n, f)
            delta = Xw_i - Xw_j  # shape: (n, n, f)
            sq_dist = np.sum(delta ** 2, axis=-1)  # shape: (n, n)
            K = np.exp(-self.gamma * sq_dist)  # shape: (n, n)

            # Compute ∂/∂w of ||Xw_i - Xw_j||^2:
            # ∂/∂w_k [||Xw_i - Xw_j||^2] = 2 * (X_i - X_j)^2 * w_k
            X_i = X[:, np.newaxis, :]  # shape: (n, 1, f)
            X_j = X[np.newaxis, :, :]  # shape: (1, n, f)
            delta_X = X_i - X_j  # shape: (n, n, f)

            # Square and multiply by w (broadcasted)
            weighted_grad = delta_X ** 2 * self.w  # shape: (n, n, f)

            # Final gradient
            grad = -2 * self.gamma * K[:, :, np.newaxis] * weighted_grad  # shape: (n, n, f)
            return grad


        elif self.kernel == "sigmoid":
            # K = tanh(gamma * Xw @ Xw.T + b)
            dot = np.dot(Xw, X.T)
            inner = self.gamma * dot + self.b
            tanh_term = np.tanh(inner)
            sech2 = 1 - tanh_term ** 2  # derivative of tanh

            # ∂/∂w of (Xw @ Xw.T) = X * X
            return sech2[:, :, np.newaxis] * self.gamma * X[np.newaxis, :, :] * X[:, np.newaxis, :]

        else:
            raise ValueError(f"Unsupported kernel: {self.kernel}")
        
    def compute_dL_dK(self,alpha, y):
        alpha_y = alpha * y 
        return -np.outer(alpha_y, alpha_y)
        
    def update_weights(self, X, alpha, y):
        dL_dK = self.compute_dL_dK(alpha, y)
        dK_dw = self.partial_derivative_wrt_weights(X)
        dL_dw = np.sum(dL_dK[:,:,np.newaxis] * dK_dw, axis=(0,1))
        self.w -= self.learning_rate * dL_dw

    
    def fit(self, X, y):
        n_samples, n_features = X.shape
        y_ = np.where(y <= 0, -1, 1)
        self.X_train = X
        self.y_train = y_

        if self.gamma == 'scale':
            self.gamma = 1 / (self.X_train.shape[1] * self.X_train.var())
        elif self.gamma == 'auto':
            self.gamma = 1 / self.X_train.shape[1]

        self.w = np.ones(n_features)
        self.alpha = np.zeros(n_samples)
        self.support_weights = self.alpha * self.y_train

        if self.kernel is None:
            for _ in range(self.n_iters):
                for idx, x_i in enumerate(X):
                    condition = y_[idx] * (np.dot(x_i, self.w) - self.b) >= 1
                    if condition:
                        self.w -= 2 * self.learning_rate * self.w * self.c
                    else:
                        self.w -= self.learning_rate * (2 * self.w * self.c - y[idx] * x_i)
                        self.b -= self.learning_rate * y_[idx]
        else:
            for iteration in range(self.n_iters):
                self.K = self.kernel_function(X, X)
                grad_alpha = np.zeros(n_samples)
                grad_alpha = 1 - self.y_train * np.sum(self.alpha * self.y_train * self.K, axis=0)  
                # self.update_weights(X, self.alpha, y_) if self.use_weights else 0
                self.alpha += self.learning_rate * grad_alpha
                self.alpha = np.clip(self.alpha, 0, self.c)

                if iteration % 100 ==0:
                    print(f"iteration {iteration}")                    
                    
            self.support_vectors_ = X[self.alpha > 1e-5]
            self.support_vector_labels_ = self.y_train[self.alpha > 1e-5]
            self.alpha = self.alpha[self.alpha > 1e-5]
            self.support_weights = self.alpha * self.support_vector_labels_
            if self.kernel is not None:
                total_bias = 0
                for sv_w,sv_y, sv in zip(self.support_weights,self.support_vector_labels_, self.support_vectors_):
                    total_bias += sv_y - sum(sv_w * self.kernel_function(sv, x) for x in self.X_train)
                self.b = total_bias / len(self.support_vectors_)
    

    def predict(self, X):
        if self.kernel is None:
            approx = np.dot(X, self.w) - self.b 
        else:
            y_pred = []
            for x in X :
                prediction = 0
                for j in range(len(self.alpha)):
                    prediction += self.alpha[j] * self.y_train[j] * self.kernel_function(x, self.X_train[j]) + self.b
                y_pred.append(prediction)
            approx = np.array(y_pred)
        
        return np.where(approx <= 0, -1, 1)