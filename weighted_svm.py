import numpy as np

class WeightedSVM:
    def __init__(self, kernel=None, learning_rate=0.1,n_iters=1000, c=0.01, degree=2, bias=1, gamma=0.1):
        self.w = None #vecteur de poids
        self.learning_rate = learning_rate
        self.c = c
        self.gamma = gamma
        self.degree = degree        
        self.kernel = kernel
        self.b = bias if self.kernel else 0
        self.n_iters = n_iters
        self.alpha = np.array([])
    
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
        """
        Méthode permettant de mesurer la similarité entre deux points x1 et x2.
        """
        if self.kernel in ["linear", None]:
            return np.dot(self.w * x1, x2)+self.b #insertion du vecteur de poids : multiplication de l'importance de chaque variable pour la comparaison
        if self.kernel == "poly":
            return (self.gamma * np.dot(self.w * x1, x2)+self.b)**self.degree 
        if self.kernel == "rbf":
            return np.exp(-self.gamma * np.sum((((x1-x2)**2)*self.w))) #la multiplication par le vecteur poids hors du carré permet de garder le signe de l'importance du poids : on fait la somme des carrés qui est égale au carré de la distance euclidienne utilisée normalement dans le kernel rbf
        if self.kernel == "sigmoid":
            return np.tanh(self.gamma * np.dot(self.w*x1, x2)+self.b)
        else:
            raise ValueError
    
    def make_kernel_matrix(self, X):
        n_samples = X.shape[0]
        K = np.zeros((n_samples, n_samples))
        for i in range(n_samples):
            for j in range(n_samples):
                K[i,j] = self.kernel_function(X[i], X[j])
        return K
    
    
    def partial_derivative_w(self, w_idx, x1, x2):
        """
        Retourne la dérivée partielle de la fonction Kernel par rapport à un poids.
        """
        if self.kernel == "linear":
            return x1[w_idx] * x2[w_idx]
        elif self.kernel == "poly":
            return x1[w_idx] * x2[w_idx] * self.degree * (self.b+self.gamma * np.dot(self.w * x1, x2)**(self.degree-1))
        elif self.kernel == "rbf":
            return -self.gamma *(x1[w_idx]-x2[w_idx])**2 - self.kernel_function(x1,x2)
        elif self.kernel == "sigmoid":
            return (1-np.tanh(self.gamma * np.dot(self.w * x1, x2) + self.b)**2) * self.gamma * x1[w_idx] * x2[w_idx]
    
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
        self.support_vectors_ = X[self.alpha > 1e-5]
        self.support_vector_labels_ = self.y_train[self.alpha > 1e-5]
        self.K = self.kernel_matrix(self.X_train)

        if self.kernel is None:
            for _ in range(self.n_iters):
                for idx, x_i in enumerate(X):
                    condition = y_[idx] * (np.dot(x_i, self.w) - self.b) >= 1
                    if condition:
                        self.w -= 2 * self.learning_rate * self.w * self.c
                    else:
                        self.w -= self.learning_rate * (2 * self.w * self.c - y[idx * x_i])
                        self.b -= self.learning_rate * y_[idx]
        else:
            for iteration in range(self.n_iters):
                grad_alpha = np.ones(n_samples)
                grad_w = np.zeros(n_features)
                for i in range(n_samples):
                    grad_alpha[i] = 1 - self.y_train[i] * np.sum(self.alpha * self.y_train * self.K[:,i])
                    mul_alpha = self.alpha * self.alpha[i]
                    mul_y = self.y_train * self.y_train[i]
                    for j in range(n_features):
                        partial_derivatives = np.array([self.partial_derivative_w(j, X[i], X[k]) for k in range(n_samples)])
                        grad_w[j] += 0.5 * np.sum(mul_alpha * mul_y * partial_derivatives)

                old_alpha = self.alpha.copy()
                self.alpha += self.learning_rate * grad_alpha
                self.alpha = np.clip(self.alpha, 0, self.c)
                self.w += self.learning_rate * grad_w

                if np.linalg.norm(self.alpha - old_alpha) < self.tol:
                    print(f"Arrêt de convergence à l'itération {iteration}")
                    break
                    
                    
            self.support_vectors_ = X[self.alpha > 1e-5]
            self.support_vector_labels_ = self.y_train[self.alpha > 1e-5]
            self.alpha = self.alpha[self.alpha > 1e-5]
            self.support_weights = self.alpha * self.support_vector_labels_
            if self.kernel is not None:
                total_bias = 0
                for sv_w,sv_y, sv in zip(self.support_weights,self.support_vector_labels_, self.support_vectors_):
                    total_bias += sv_y - sum(sv_w * self._kernel(sv, x) for x in self.X_train)
                # Average the bias over all support vectors
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