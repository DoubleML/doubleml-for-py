from rdrobust import rdrobust, rdbwselect
import numpy as np
from sklearn.model_selection import cross_val_predict
import scipy.stats as stats
from doubleml.utils.resampling import DoubleMLResampling
from doubleml.utils._estimation import _get_cond_smpls

class RDFlex():
    """Flexible adjustment with double machine learning for regression discontinuity designs

    Parameters
    ----------
    obj_dml_data : :class:`DoubleMLData` object
        The :class:`DoubleMLData` object providing the data and specifying the variables for the causal model.

    ml_g : estimator implementing ``fit()`` and ``predict()``
        A machine learner implementing ``fit()`` and ``predict()`` methods (e.g.
        :py:class:`sklearn.ensemble.RandomForestRegressor`) for the nuisance function :math:`g_0(X) = E[Y|X]`.

    ml_m : classifier implementing ``fit()`` and ``predict_proba()`` or None
        A machine learner implementing ``fit()`` and ``predict_proba()`` methods (e.g.
        :py:class:`sklearn.ensemble.RandomForestClassifier`) for the nuisance function :math:`m_0(X) = E[D|X]`.
        Or None, in case of a non-fuzzy design.

    n_folds : int
        Number of folds.
        Default is ``5``.

    n_rep : int
        Number of repetitons for the sample splitting.
        Default is ``1``.

    cutoff : float or int
        A float or intspecifying the cutoff in the score.
        Default is ``0``.

    h_fs : float or int or None
        Initial bandwidth in the first stage estimation. If ``None``, then the optimal bandwidth without
        covariates will be used.
        Default is ``None``.

    fs_kernel : str
        Kernel for the first stage estimation. ``uniform``, ``triangular`` and ``epanechnikov``are supported.
        Default is ``uniform``.

    **kwargs : 

    Examples
    --------

    Notes
    -----

    """

    def __init__(self, obj_dml_data, ml_g, ml_m=None, cutoff=0, n_folds=5, n_rep=1, h_fs=None, fs_kernel="uniform", **kwargs):
        self.data = {}
        
        if obj_dml_data.d is not None and ml_m is None:
            raise ValueError("If D is specified (Fuzzy Design), a classifier 'ml_m' must be provided.")
        self._dml_data = obj_dml_data

        self._dml_data -= cutoff
        self.data["T"] = (0.5*(np.sign(X)+1)).astype(bool)

        self.ml_g = ml_g
        self.ml_m = ml_m
        self.n_folds = n_folds
        self.n_rep = n_rep
        self.h_fs = h_fs
        self.alpha = alpha
        self.kwargs = kwargs

        self.initial_bw = rdbwselect(y=Y, x=X, fuzzy=D).bws.values.flatten()
        self.w = self._set_weights(fs_kernel=fs_kernel, 
                                   bw=2 * self.initial_bw.max() * h_fs)
        self.w_mask = self.w.astype(bool)
        
        self.smpls = DoubleMLResampling(n_folds=n_folds, n_rep=n_rep, n_obs=self.w_mask.sum(), 
                                        stratify=D[self.w_mask]).split_samples()
        
        self._initialize_reps(n_obs = self.w_mask.sum(), n_rep = n_rep)

    def __str__(self):
        ci_conventional = [round(ci, 3) for ci in self.ci[0, :]]
        ci_robust = [round(ci, 3) for ci in self.ci[2, :]]
        col_format = "{:<20} {:>8} {:>8} {:>8} {:>8} to {:<8}"

        header = (
            "Method                  Coef.     S.E.    P>|t|            95% CI\n"
            "-----------------------------------------------------------------"
        )

        conventional_row = col_format.format(
            "Conventional",
            round(self.coef[0], 3),
            round(self.se[0], 3),
            round(self.p_value[0], 3),
            ci_conventional[0],
            ci_conventional[1]
        )

        robust_row = col_format.format(
            "Robust",
            "-",
            "-",
            round(self.p_value[2], 3),
            ci_robust[0],
            ci_robust[1]
        )

        result = f"{header}\n{conventional_row}\n{robust_row}"
        
        return result

    def fit(self, n_jobs=-1):
        for i_rep in range(self.n_rep):
            self._i_rep = i_rep
            eta_Y, eta_D = self._fit_nuisance_models(n_jobs)
            self.M_Y[:,i_rep] = self.data["Y"][self.w_mask] - eta_Y
            if self.data["D"] is not None:
                self.M_D[:,i_rep] = self.data["D"][self.w_mask] - eta_D
            self._fit_rdd()
        
        self.aggregate_over_splits()

        return self
    
    def _fit_nuisance_models(self, n_jobs):
        T = self.data["T"][self.w_mask]
        X = np.c_[T, self.data["Z"][self.w_mask]]
        D = self.data["D"][self.w_mask]
        Y = self.data["Y"][self.w_mask]

        _check_left, _check_right = self._check_fuzzyness(w=self.w_mask,min_smpls=2*self.n_folds)
        if _check_left:
            pred_treat_d0 = cross_val_predict(estimator=self.ml_m, X=self.data["Z"][self.w_mask], y=self.data["D"][self.w_mask], cv=smpls_d0, 
                                              n_jobs=n_jobs, method="predict_proba", params = {"sample_weight": self.w[self.w_mask]})[:,1]
        else:
            pred_treat_d0 = np.average(self.data["D"][~self.data["T"]], weights=self.w[~self.data["T"]])
        if _check_right:
            pred_treat_d1 = cross_val_predict(estimator=self.ml_m, X=self.data["Z"][self.w_mask], y=self.data["D"][self.w_mask], cv=smpls_d1, 
                                              n_jobs=n_jobs, method="predict_proba", params = {"sample_weight": self.w[self.w_mask]})[:,1]
        else:
            pred_treat_d1 = np.average(self.data["D"][self.data["T"]], weights=self.w[self.data["T"]])

        pred_outcome_d0 = cross_val_predict(estimator=self.ml_g, X=X, y=sY, cv=smpls, n_jobs=n_jobs, method="predict", 
                                            params = {"sample_weight": self.w[self.w_mask]})

        eta_D = .5 * (pred_treat_d0 + pred_treat_d1)
        eta_Y = .5 * (pred_outcome_d0 + pred_outcome_d1)

        return eta_Y, eta_D
        

    def _fit_rdd(self):
        _rdd_res = rdrobust(y=self.M_Y[:,self._i_rep], x=self.data["X"][self.w_mask], 
                            fuzzy=self.M_D[:,self._i_rep], **self.kwargs)
        self.coefs[:,self._i_rep] = _rdd_res.coef.values.flatten()
        self.ses[:,self._i_rep] = _rdd_res.se.values.flatten()
        self.cis[:,:,self._i_rep] = _rdd_res.ci.values
        self.rdd_res.append(_rdd_res)
        return
    
    def _set_weights(self, bw, fs_kernel="uniform"):
        if fs_kernel == "uniform":
            return (np.abs(self.data["X"]) < bw)
        if fs_kernel == "triangular":
            return np.maximum(0, (bw - np.abs(self.data["X"])) / bw)
        if fs_kernel == "epanechnikov":
            return np.where(np.abs(self.data["X"])<bw,.75*(1-self.data["X"]/bw)**2,0)
        
    def _check_fuzzyness(self, w, min_smpls):
        return ((self.data["D"][w][self.data["X"][w]<0].sum() > min_smpls), 
                ((self.data["D"][w][self.data["X"][w]>0] - 1).sum() < -(min_smpls)))
    
    def _initialize_reps(self, n_obs, n_rep):
        self.M_Y = np.empty(shape=(n_obs, n_rep))
        self.M_D = np.empty(shape=(n_obs, n_rep)) if self.data["D"] is not None else None
        self.rdd_res = []
        self.coefs = np.empty(shape=(3,n_rep))
        self.ses = np.empty(shape=(3,n_rep))
        self.cis = np.empty(shape=(3,2,n_rep))
        return

    def aggregate_over_splits(self):
        self.coef = np.median(self.coefs, axis=1)
        self.ci = np.median(self.cis, axis=2)
        med_se = np.median(self.ses, axis=1)
        self.se = [np.sqrt(np.median(med_se[i]**2 + (self.coefs[i, :] - self.coef[i])**2)) for i in range(3)]

        z = self.coef / self.se
        self.p_value = 2*stats.norm.cdf(-np.abs(z))