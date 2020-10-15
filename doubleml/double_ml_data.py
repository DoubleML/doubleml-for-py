import numpy as np
import pandas as pd
import io

from ._helper import assure_2d_array


class DoubleMLData:
    def __init__(self,
                 data,
                 y_col,
                 d_cols,
                 x_cols=None,
                 z_cols=None,
                 use_other_treat_as_covariate=True):
        self.data = data
        self.y_col = y_col
        self.d_cols = d_cols
        self.z_cols = z_cols
        self.use_other_treat_as_covariate = use_other_treat_as_covariate
        if x_cols is not None:
            self.x_cols = x_cols
        else:
            # x_cols defaults to all columns but y_col, d_cols and z_cols
            if self.z_cols is not None:
                y_d_z = set.union(set(self.y_col), set(self.d_cols), set(self.z_cols))
                self.x_cols = [col for col in self.data.columns if col not in y_d_z]
            else:
                y_d = set.union(set(self.y_col), set(self.d_cols))
                self.x_cols = [col for col in self.data.columns if col not in y_d]
        self._set_y_z()
        # by default, we initialize to the first treatment variable
        self._set_x_d(self.d_cols[0])

    def __repr__(self):
        buf = io.StringIO()
        self.data.info(verbose=False, buf=buf)
        data_info = buf.getvalue()
        return f'=== DoubleMLData Object ===\n' \
               f'y_col: {self.y_col}\n' \
               f'd_cols: {self.d_cols}\n' \
               f'x_cols: {self.x_cols}\n' \
               f'z_cols: {self.z_cols}\n' \
               f'data:\n {data_info}'

    @classmethod
    def from_arrays(cls, X, y, d, z=None):
        X = assure_2d_array(X)
        d = assure_2d_array(d)

        # assert single y variable here
        y_col = 'y'
        if z is None:
            z_cols = None
        else:
            if z.shape[1] == 1:
                z_cols = ['z']
            else:
                z_cols = [f'z{i + 1}' for i in np.arange(z.shape[1])]

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
                                columns=x_cols + [y_col] + d_cols + z_cols)

        return cls(data, y_col, d_cols, x_cols, z_cols)

    @property
    def x(self):
        return self._X.values
    
    @property
    def y(self):
        return self._y.values
    
    @property
    def d(self):
        return self._d.values
    
    @property
    def z(self):
        if self.z_cols is not None:
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
    def n_instr(self):
        return len(self.z_cols)
    
    @property 
    def n_obs(self):
        return self.data.shape[0]
    
    @property
    def x_cols(self):
        return self._x_cols
    
    @x_cols.setter
    def x_cols(self, value):
        if isinstance(value, str):
            value = [value]
        if not isinstance(value, list):
            raise TypeError('x_cols must be a list')
        assert set(value).issubset(set(self.all_variables))
        self._x_cols = value
    
    @property
    def d_cols(self):
        return self._d_cols
    
    @d_cols.setter
    def d_cols(self, value):
        if isinstance(value, str):
            value = [value]
        if not isinstance(value, list):
            raise TypeError('d_cols must be a list')
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
    def z_cols(self):
        return self._z_cols
    
    @z_cols.setter
    def z_cols(self, value):
        if value is not None:
            if isinstance(value, str):
                value = [value]
            if not isinstance(value, list):
                raise TypeError('z_cols must be a list')
            assert set(value).issubset(set(self.all_variables))
            self._z_cols = value
        else:
            self._z_cols = None
    
    def _set_y_z(self):
        self._y = self.data.loc[:, self.y_col]
        if self.z_cols is None:
            self._z = None
        else:
            self._z = self.data.loc[:, self.z_cols]
    
    def _set_x_d(self, treatment_var):
        assert treatment_var in self.d_cols
        if self.use_other_treat_as_covariate:
            xd_list = self.x_cols + self.d_cols
            xd_list.remove(treatment_var)
        else:
            xd_list = self.x_cols
        self._d = self.data.loc[:, treatment_var]
        self._X = self.data.loc[:, xd_list]
