import numpy as np

class NaiveBayes :
    def fit(self, X, y):
        n_samples, n_features = X.shape
        self._classes = np.unique(y)
        n_classes = len(self._classes)

        self._mean = np.zeros((n_classes , n_features), dtype= float)
        self._variance = np.zeros((n_classes , n_features), dtype= float)
        self._priors = np.zeros(n_classes , dtype= float)

        for idx , C in enumerate(self._classes):
            X_c = X[y  == C]
            self._mean[idx,:] = X_c.mean(axis = 0)
            self._variance[idx,:] = X_c.var(axis = 0)
            self._priors[idx] = X_c.shape[0] / float(n_samples)

    def predict(self , X):
        y_pred = [self.predicthelper(x) for x in X]
        return np.array(y_pred)
    
    def predicthelper(self, X):
        posteriors = []
        for idx , c in enumerate(self._classes):
            prior = np.log(self._priors[idx])
            posterior = np.sum(np.log(self._probDensity(idx, X)))
            posterior = posterior * prior
            posteriors.append(posterior)

        return self._classes[np.argmax(posteriors)]
    
    def _probDensity(self ,class_idx , X):
        mean = self._mean[class_idx]
        var = self._variance[class_idx]
        
        numerator = np.exp(-((X-mean) ** 2) / (2*var))
        denominator = np.sqrt(2 * np.pi * var)

        return numerator / denominator