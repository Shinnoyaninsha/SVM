import numpy as np 

class WSVM_C:
    def __init__(self,kernel=None, learning_rate=0.001, c_param=0.01, n_iters=1000, degree=3, bias=1, gamma=0.1):
        self.w = None
        self.b = None
        self.learning_rate = learning_rate
        self.c = c_param #soft margin parameter
        self.gamma = gamma #paramètre de pénalisation du kernel rbf
        self.degree = degree #paramètre du degré du kernel poly
        self.poly_bias = bias #paramètre du biais du kernel poly
        self.n_iters = n_iters
        self.kernel = kernel
        self.alpha = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        #Conversion des labels en -1, 1 pour le SVM qui utilise le signe du résultat pour la classification
        y_ = np.where(y <= 0, -1, 1)
        self.X_train = X
        self.y_train = y_

        #initialisation de tous les poids à 0
        self.w = np.zeros(n_features)
        self.b = 0
        self.alpha = np.zeros(n_samples)

        #descente de gradient, avec utilisation de la hinge loss comme fonction de coût à minimiser
        #si la condition de supériorité à 1 n'est pas satisfaite => soit la prédiction est fausse, soit la marge n'est pas suffisante
        #si la condition est satisfaite, on actualise par le terme de régularisation
        if self.kernel is None: #si la SVM n'utilise pas de kernel, on utilise la descente de gradient pour actualiser le vecteur de poids w
            for _ in range(self.n_iters):
                for idx, x_i in enumerate(X):
                    condition = y_[idx] * (np.dot(x_i, self.w) - self.b) >= 1
                    if condition:
                        self.w -= 2 * self.learning_rate * self.w * self.c
                    else:
                        self.w -=  self.learning_rate * (self.w *2* self.c - y[idx] * x_i)
                        self.b -=  self.learning_rate * y_[idx]
        else: #si la SVM utilise un kernel, on utilise la descente de gradient pour actualiser les coefficients alpha à partir des formules du Lagrangien
            for _ in range(self.n_iters):
                for i in range(n_samples):
                    prediction = 0
                    for j in range(n_samples):
                        #la prédiction se fait à partir de la somme de la contribution de chacun des points des vecteurs supports, matéralisée par le kernel et le coefficient alpha
                        prediction += self.alpha[j] * self.y_train[j] * self._kernel(self.X_train[i], self.X_train[j])
                    if self.y_train[i] * prediction < 1:
                        self.alpha[i] += self.learning_rate * (1 - self.y_train[i] * prediction)
                    else:
                        self.alpha[i] -= self.learning_rate * (self.c * self.alpha[i])
            # Mise à jour du biais
            self.support_vectors = [(i, a) for i, a in enumerate(self.alpha) if 0 < a < self.c]

            if self.support_vectors:
                # Use the first support vector to calculate the bias (could also average over all support vectors)
                idx, _ = self.support_vectors[0]
                self.b = self.y_train[idx] - sum(
                    self.alpha[j] * self.y_train[j] * self._kernel(self.X_train[idx], self.X_train[j])
                    for j in range(n_samples)
                )



    def predict(self, X):
        #le calcul de la prévision est différent en fonction de l'utilisation de kernel
        if self.kernel is None:
            approx = np.dot(X, self.w) - self.b
            return np.where(approx <= 0, -1, 1)
            #la fonction de prédiction est la somme des produits du vecteur poids et de la matrice des features
            #la fonction de prédiction retourne un réel, dont le signe permet de déterminer la classe prédite, si prediction >1, classe positive, sinon classe négative
        else:
            y_pred = []
            for x in X:
                prediction = 0
                for j in range(len(self.alpha)):
                    prediction += self.alpha[j] * self.y_train[j] * self._kernel(x, self.X_train[j]) + self.b
                y_pred.append(prediction)
            result = np.where(np.array(y_pred) <= 0, -1, 1)
                #la fonction de prédiction est la somme des produits entre le lagrangien, qui est la force de similarité entre le point X et le vecteur support, le résultat de la fonction kernel entre le point X et le vecteur support, et l'étiquette attachée au vecteur support
            return result

    def decision_function(self, x):
        if self.kernel is None:
            return -(self.w[0] * x - self.b) / self.w[1]
            #permet d'afficher la fonction de décision

    def marges(self, x):
        if self.kernel is None:
            return (self.decision_function(x) +1/self.w[1], self.decision_function(x) -1/self.w[1])
            #permet d'afficher les marges, situées à une distance 1 de la frontière de décision

    def _kernel(self, x1, x2):
        if self.kernel in ["linear", "poly"]:
            return np.dot(x1, x2 + self.poly_bias)** self.degree
        elif self.kernel == "rbf":
            return np.exp(-np.linalg.norm(x1 - x2) ** 2 / (2 * (self.c ** 2)))
        elif self.kernel == "cosine": #n'existe pas dans la littérature ou la pratique, a été créé en référence au cours de NLP ; à tester sur des classifications de texte
            return np.dot(x1, x2) / (np.linalg.norm(x1) * np.linalg.norm(x2))
        elif self.kernel == "sigmoid":
            return np.tanh(np.dot(x1, x2) + self.b)
        elif self.kernel is None:
            return 1
        else:
            raise ValueError("Kernel non reconnu")
