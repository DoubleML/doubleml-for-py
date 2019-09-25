import numpy as np
import pandas as pd
from .helper import assure_2d_array


class DoubleMLData:
    def __init__(self,
                 data,
                 x_cols,
                 y_col,
                 d_cols,
                 z_col=None):
        self.data = data
        self.x_cols = x_cols
        self.y_col = y_col
        self.d_cols = d_cols
        self.z_col = z_col
        self.extract_y_z()
        # by default, we initialize to the first treatment variable
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
    def x_cols(self):
        return self._x_cols
    
    @x_cols.setter
    def x_cols(self, value):
        assert isinstance(value, list)
        assert set(value).issubset(set(self.all_variables))
        self._x_cols = value
    
    @property
    def d_cols(self):
        return self._d_cols
    
    @d_cols.setter
    def d_cols(self, value):
        assert isinstance(value, list)
        assert set(value).issubset(set(self.all_variables))
        self._d_cols = value
    
    @property
    def y_col(self):
        return self._y_col
    
    @y_col.setter
    def y_col(self, value):
        assert isinstance(value, str)
        assert value in self.all_variables
        self._y_col = value
    
    @property
    def z_col(self):
        return self._z_col
    
    @z_col.setter
    def z_col(self, value):
        if value is not None:
            assert isinstance(value, str)
            assert value in self.all_variables
            self._z_col = value
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
        xd_list = self.x_cols + self.d_cols
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
    
    x_cols = [f'X{i+1}' for i in np.arange(X.shape[1])]

    if z is None:
        data = pd.DataFrame(np.column_stack((X, y, d)),
                            columns=x_cols + [y_col] + d_cols)
    else:
        data = pd.DataFrame(np.column_stack((X, y, d, z)),
                            columns=x_cols + [y_col] + d_cols + [z_col])
    return DoubleMLData(data, x_cols, y_col, d_cols, z_col)

