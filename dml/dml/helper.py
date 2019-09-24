import numpy as np
import pandas as pd
from sklearn.utils.multiclass import type_of_target

def assure_2d_array(x):
    if x.ndim == 1:
        x = x.reshape(-1,1)
    elif x.ndim > 2:
        raise ValueError('Only one- or two-dimensional arrays are allowed')
    return x

def check_binary_vector(x, variable_name=''):
    # only a single treatment variable is allowed
    assert x.ndim == 1
    
    # assure D binary
    assert type_of_target(x) == 'binary', 'variable ' + variable_name  + ' must be binary'
    
    if np.any(np.power(x,2) - x != 0):
        raise ValueError('variable ' + variable_name  + ' must be binary with values 0 and 1')


class DoubleMLData():
    def __init__(self,
                 data,
                 X_cols,
                 y_col,
                 d_cols,
                 z_col=None):
        self.data = data
        self.X_cols = X_cols
        self.y_col = y_col
        self.d_cols = d_cols
        self.z_col = z_col
        self.extract_y_z()
        # by default, we intialize to the first treatment variable
        self.extract_X_d(d_cols[0])
    
    @property
    def X(self):
        return self._X.values
    
    @property
    def y(self):
        return self._y.values
    
    @property
    def d(self):
        return self._d.values
    
    @property
    def z(self):
        if self.z_col is not None:
            return self._z.values
        else:
            return None
    
    @property 
    def all_variables(self):
        return self.data.columns
    
    @property 
    def n_treat(self):
        return len(self.d_cols)
    
    @property 
    def n_obs(self):
        return self.data.shape[0]
    
    @property
    def X_cols(self):
        return self._X_cols
    
    @X_cols.setter
    def X_cols(self, X_cols):
        print(X_cols)
        print(self.all_variables)
        assert isinstance(X_cols, list)
        assert set(X_cols).issubset(set(self.all_variables))
        self._X_cols = X_cols
    
    @property
    def d_cols(self):
        return self._d_cols
    
    @d_cols.setter
    def d_cols(self, d_cols):
        assert isinstance(d_cols, list)
        assert set(d_cols).issubset(set(self.all_variables))
        self._d_cols = d_cols
    
    @property
    def y_col(self):
        return self._y_col
    
    @y_col.setter
    def y_col(self, y_col):
        assert isinstance(y_col, str)
        assert y_col in self.all_variables
        self._y_col = y_col
    
    @property
    def z_col(self):
        return self._z_col
    
    @z_col.setter
    def z_col(self, z_col):
        if z_col is not None:
            assert isinstance(z_col, str)
            assert z_col in self.all_variables
            self._z_col = z_col
        else:
            self._z_col = None
    
    def extract_y_z(self):
        self._y = self.data.loc[:, self.y_col]
        if self.z_col is None:
            self._z = None
        else:
            self._z = self.data.loc[:, self.z_col]
    
    def extract_X_d(self, treatment_var):
        assert treatment_var in self.d_cols
        xd_list = self.X_cols + self.d_cols
        xd_list.remove(treatment_var)
        self._d = self.data.loc[:, treatment_var]
        self._X = self.data.loc[:, xd_list]
        
    
def double_ml_data_from_arrays(X, y, d, z=None):
    X = assure_2d_array(X)
    d = assure_2d_array(d)
    
    # assert single y and z variable here
    y_col = 'y'
    if z is None:
        z_col = None
    else:
        z_col = 'z'
    
    if d.shape[1] == 1:
        d_cols = ['d']
    else:
        d_cols = [f'd{i+1}' for i in np.arange(d.shape[1])]
    
    X_cols = [f'X{i+1}' for i in np.arange(X.shape[1])]
    
    print(X_cols)
    if z is None:
        data = pd.DataFrame(np.column_stack((X, y, d)),
                            columns = X_cols + [y_col] + d_cols)
    else:
        data = pd.DataFrame(np.column_stack((X, y, d, z)),
                            columns = X_cols + [y_col] + d_cols + [z_col])
    return DoubleMLData(data, X_cols, y_col, d_cols, z_col)
    