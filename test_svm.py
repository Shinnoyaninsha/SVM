import numpy as np 

class SVM_C:
    def __init__(self,kernel=None, learning_rate=0.001, c=1, n_iters=1000, degree=3, bias=1, gamma='scale', optimizer="gd", tol=1e-5, max_passes=20):
        self.w = None
        self.b = None
        self.learning_rate = learning_rate
        self.c = c #soft margin parameter
        self.gamma = gamma #paramètre de pénalisation du kernel rbf
        self.degree = degree #paramètre du degré du kernel poly
        self.poly_bias = bias #paramètre du biais du kernel poly
        self.n_iters = n_iters
        self.kernel = kernel
        self.alpha = None
        self.support_vectors_ = np.array([])
        self.optimizer = optimizer
        self.tol = tol
        self.max_passes = max_passes

    def kernel_matrix(self, X):
        n_s = X.shape[0]
        K = np.zeros((n_s, n_s)) # crée une matrice de 0s
        for i in range(n_s):
            for j in range(n_s):
                K[i, j] = self._kernel(X[i], X[j])
        return K
    
    def linear_fit(self, X, y):
        for _ in range(self.n_itersn):
            for idx, x_i in enumerate(X):
                condition = y[idx] * (np.dot(x_i, self.w) - self.b) >= 1
                if condition:
                    self.w -= self.learning_rate * self.w * self.c
                else:
                    self.w -= self.learning_rate * (self.learning_rate * 2 * self.c - y[idx] * x_i)
                    self.b -= self.learning_rate * y[idx]
    
    def gradient_fit(self, X, y, n_s):
        for iteration in range(self.n_iters):
            grad_alpha = np.ones(n_s)
            for i in range(n_s):
                grad_alpha[i] = 1 - y[i] * np.sum(self.alpha * y * self.K[:,i])
            old_alpha = self.alpha.copy()
            self.alpha += self.learning_rate * grad_alpha

            self.alpha = np.clip(self.alpha, 0, self.c)

            if np.linalg.norm(self.alpha - old_alpha) < self.tol:
                print(f"Convergence à l'itération {iteration}")
                break

    def fit(self, X, y):
        n_samples, n_features = X.shape
        #Conversion des labels en -1, 1 pour le SVM qui utilise le signe du résultat pour la classification
        y_ = np.where(y <= 0, -1, 1)
        if self.gamma == 'scale':
            self.gamma = 1 / (X.shape[1] * X.var())
        elif self.gamma == 'auto':
            self.gamma = 1 / X.shape[1]
        #calcul de la matrice des kernel entre tous les couples de points de l'entrainement
        self.K = self.kernel_matrix(X)

        #initialisation de tous les poids à 0
        self.w = np.zeros(n_features)
        self.b = 0
        self.alpha = np.zeros(n_samples)
        self.support_vectors_ = X[self.alpha > 1e-5]
        self.support_vector_labels_ = y[self.alpha > 1e-5]

        #descente de gradient, avec utilisation de la hinge loss comme fonction de coût à minimiser
        #si la condition de supériorité à 1 n'est pas satisfaite => soit la prédiction est fausse, soit la marge n'est pas suffisante
        #si la condition est satisfaite, on actualise par le terme de régularisation
        if self.kernel is None: #si la SVM n'utilise pas de kernel, on utilise la descente de gradient pour actualiser le vecteur de poids w
            for _ in range(self.n_iters):
                self.linear_fit(X, y_, self.n_iters)


        else: #si la SVM utilise un kernel, on utilise soit la smo soit la descente de gradient pour optimiser les valeurs alpha
            if self.optimizer == "gd":
                self.gradient_fit(X, y_, n_samples)

            elif self.optimizer == "smo":
                passes = 0
                self.errors = self.decision_function(X) - y
                for iter in range(self.n_iters):
                    alpha_changed = 0
                    alpha_prev = self.alpha.copy()
                    for i in range(n_s):
                        E_i = self.decision_function(X[i]) - y[i]
                        if (y[i] * E_i < -self.tol and self.alpha[i] < self.c) or (y[i] * E_i > self.tol and self.alpha[i] > 0):
                            j = np.argmax(np.abs(self.errors - self.errors[i]))
                            j = j if j !=i else (i+1)% n_s
                            E_j = self.decision_function(X[j]) - y[j]

                            alpha_i_old, alpha_j_old = self.alpha[i], self.alpha[j]

                            if y[i] != y[j]:
                                L = max(0, self.alpha[j] - self.alpha[i])
                                H = min(self.c, self.c + self.alpha[j] - self.alpha[i])
                            else:
                                L = max(0, self.alpha[i] + self.alpha[j] - self.c)
                                H = min(self.c, self.alpha[i] + self.alpha[j])

                            if L == H:
                                continue

                            eta = 2.0 * self.K[i, j] - self.K[i, i] - self.K[j, j]
                            if eta >=0:
                                continue
                            
                            self.alpha[j] -= y[j]* (E_i - E_j)/eta
                            self.alpha[j] = np.clip(self.alpha[j], L, H)
                            
                            if abs(self.alpha[j] - alpha_j_old) < 1e-5:
                                continue

                            self.alpha[i] += y[i] * y[j] * (alpha_j_old - self.alpha[j])

                            b1 = self.b - E_i - y[i] * (self.alpha[i] - alpha_i_old) * self.K[i, i] - y[j] * (self.alpha[j] - alpha_j_old) * self.K[i, j]
                            b2 = self.b - E_j - y[i] * (self.alpha[i] - alpha_i_old) * self.K[i, j] - y[j] * (self.alpha[j] - alpha_j_old) * self.K[j, j]
                            self.b = b1 if 0 < self.alpha[i] < self.c else (b2 if 0 < self.alpha[j] < self.c else (b1 + b2) / 2)

                            alpha_changed += 1
                            self.errors = self.decision_function(X) - y

                    passes = passes + 1 if alpha_changed == 0 else 0
                    if passes == self.max_passes:
                        break

                    diff = np.linalg.norm(self.alpha - alpha_prev)
                    if diff < self.tol:
                        break

            self.support_vectors_ = X[self.alpha > 1e-5]
            self.support_vector_labels_ = y[self.alpha > 1e-5]
            self.alpha = self.alpha[self.alpha > 1e-5]            
            self.support_weights = self.alpha * self.support_vector_labels_
            #Le biais est déjà calculé pendant l'entrainement via smo
            if self.optimizer == "gd" and self.kernel is not None:
                total_bias = 0
                for sv_w,sv_y, sv in zip(self.support_weights,self.support_vector_labels_, self.support_vectors_):
                    total_bias += sv_y - sum(sv_w * self._kernel(sv, x) for x in X)
                # Average the bias over all support vectors
                self.b = total_bias / len(self.support_vectors_)



    def decision_function(self, X):
        if self.kernel is None:
            return np.dot(X, self.w)-self.b
        else:
            decision = self.b
            for alpha, sv_y, sv in zip(self.alpha, self.support_vector_labels_, self.support_vectors_):
                decision += alpha * sv_y * self._kernel(sv, X)
            return decision
        

    def predict(self, X):
        approx = np.array([self.decision_function(x) for x in X])
        result = np.where(np.array(approx) <= 0, -1, 1)
        #la fonction de prédiction est la somme des produits entre le lagrangien, qui est la force de similarité entre le point X et le vecteur support, le résultat de la fonction kernel entre le point X et le vecteur support, et l'étiquette attachée au vecteur support
        return result   
    
    def get_score(self,X, y):
            y = np.where(y <= 0, -1, 1)
            erreurs = y == self.predict(X)
            return np.sum(erreurs)/len(erreurs)

    def draw_decision_function(self, x):
        if self.kernel is None:
            return -(self.w[0] * x - self.b) / self.w[1]
            #permet d'afficher la fonction de décision

    def marges(self, x):
        if self.kernel is None:
            return (self.decision_function(x) +1/self.w[1], self.decision_function(x) -1/self.w[1])
            #permet d'afficher les marges, situées à une distance 1 de la frontière de décision

    def _kernel(self, x1, x2):
        if self.kernel =="linear":
            return np.dot(x1, x2)
        elif self.kernel=="poly":
            return (self.poly_bias + self.gamma * np.dot(x1,x2)) ** self.degree
        elif self.kernel == "rbf":
            return np.exp(-self.gamma * np.linalg.norm(x1 - x2) ** 2)
        elif self.kernel == "cosine": #n'existe pas dans la littérature ou la pratique, a été créé en référence au cours de NLP ; à tester sur des classifications de texte
            return self.gamma * np.dot(x1, x2) / (np.linalg.norm(x1) * np.linalg.norm(x2))
        elif self.kernel == "sigmoid":
            return np.tanh(self.gamma * np.dot(x1, x2) + self.poly_bias)
        elif self.kernel is None:
            return 1
        else:
            raise ValueError("Kernel non reconnu")
