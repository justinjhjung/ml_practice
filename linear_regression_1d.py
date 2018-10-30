import numpy as np

class LinearRegression_1D:
    def __init__(self):
        self.theta0 = np.random.rand()
        self.theta1 = np.random.rand()
    
    def fit(self, x, y, normalize=True, eta = 1e-3, diff_obj = 1e-2, verbose = False):
        '''
        Fit linear model

        Parameters
        ----------
        x : 1D_array,
            x data
        y : array,
            y data
        eta : float, 
              learning rate
        diff_obj : float, 
                   minimum difference that can be allowed 
                   between the previous error and the current one 
                   for the early stopping
        verbose : bool,
                  verbosity

        Return
        ------
        Does not return anything but the fitted instance.
        '''
        diff = 1
        if normalize == True:
            x = self._preprocess(x)
        error = self._obj_function(x, y)
        
        count = 0
        while diff > diff_obj:
            temp0 = self.theta0 - eta * np.sum((self.predict(x) - y))
            temp1 = self.theta1 - eta * np.sum((self.predict(x) - y) * x)
            
            # Update with temp objects
            self.theta0 = temp0
            self.theta1 = temp1
            
            # Calculate current error
            current_error = self._obj_function(x, y)
            diff = error - current_error
            error = current_error
            
            if verbose == True:
                count += 1
                log = 'Trial number {} : theta0 : {:.3f}, theta1 = {:.3f}, diff = {:.4f}'
                print(log.format(count, self.theta0, self.theta1, diff))
    
    def predict(self, x):
        return self.theta0 + self.theta1 * x

    def _obj_function(self, x, y):
        return 0.5 * np.sum(y - self.predict(x)) ** 2

    def _preprocess(self, x):
        mu = np.mean(x)
        sigma = np.std(x)
        return (x - mu) / sigma

if __name__ == '__main__':
    X = np.array([1,2,3])
    y = np.array([1,2,2])
    
    lm_1d = LinearRegression_1D()
    lm_1d.fit(X,y, normalize=False, diff_obj=1e-10)
    print('y = {} + {} * x'.format(lm_1d.theta0, lm_1d.theta1))
