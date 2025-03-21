import numpy as np

class WeightedSVM:
    def __init__(self, kernel=None, learning_rate=0.1,n_iters=1000, c=0.01, degree=2, bias=1, gamma=0.1):
        self.w = None #vecteur de poids
        self.b = None #biais des svm sans kernel
        self.learning_rate = learning_rate
        self.c = c
        self.gamma = gamma
        self.degree = degree
        self.kernel_bias = bias
        self.n_iters = n_iters
        self.kernel = kernel
        self.alpha = None #vecteur de coefficients alpha
    
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
            return np.dot(self.w * x1, x2)+self.kernel_bias #insertion du vecteur de poids : multiplication de l'importance de chaque variable pour la comparaison
        if self.kernel == "poly":
            return (self.gamma * np.dot(self.w * x1, x2)+self.kernel_bias)**self.degree 
        if self.kernel == "rbf":
            return np.exp(-self.gamma * np.sum((((x1-x2)**2)*self.w))) #la multiplication par le vecteur poids hors du carré permet de garder le signe de l'importance du poids : on fait la somme des carrés qui est égale au carré de la distance euclidienne utilisée normalement dans le kernel rbf
        if self.kernel == "sigmoid":
            return np.tanh(self.gamma * np.dot(self.w*x1, x2)+self.kernel_bias)
        else:
            raise ValueError
    
    def partial_derivative_w(self, w_idx, x1, x2):
        """
        Retourne la dérivée partielle de la fonction Kernel par rapport à un poids.
        """
        if self.kernel == "linear":
            return x1[w_idx] * x2[w_idx]
        elif self.kernel == "poly":
            return x1[w_idx] * x2[w_idx] * self.degree * (self.kernel_bias+self.gamma * np.dot(self.w * x1, x2)**(self.degree-1))
        elif self.kernel == "rbf":
            return -self.gamma *(x1[w_idx]-x2[w_idx])**2 - self.kernel_function(x1,x2)
        elif self.kernel == "sigmoid":
            return (1-np.tanh(self.gamma * np.dot(self.w * x1, x2) + self.kernel_bias)**2) * self.gamma * x1[w_idx] * x2[w_idx]
    
    def partial_derivative_gamma(self, x1, x2):
        