from rpy2.robjects import pandas2ri
from rpy2 import robjects
pandas2ri.activate()

# The R code to fit the DML model
r_MLPLR = robjects.r('''
        library('dml')
        library('mlr3learners')
        library('data.table')
        library('mlr3')

        f <- function(data, inf_model, dml_procedure) {
            data = data.table(data)
            mlmethod <- list(mlmethod_m = 'regr.lm',
                             mlmethod_g = 'regr.lm')
            params <- list(params_m = list(),
                           params_g = list())

            double_mlplr_obj = DoubleMLPLR$new(n_folds = 2,
                                     ml_learners = mlmethod,
                                     params = params,
                                     dml_procedure = dml_procedure, inf_model = inf_model)
            double_mlplr_obj$fit(data, y = 'y', d = 'd')
            return(list(coef = double_mlplr_obj$coef,
                        se = double_mlplr_obj$se))
        }
        ''')

r_MLPLIV = robjects.r('''
        library('dml')
        library('mlr3learners')
        library('data.table')
        library('mlr3')

        f <- function(data, inf_model, dml_procedure) {
            data = data.table(data)
            mlmethod <- list(mlmethod_m = 'regr.lm',
                             mlmethod_g = 'regr.lm',
                             mlmethod_r = 'regr.lm')
            params <- list(params_m = list(),
                           params_g = list(),
                           params_r = list())

            double_mlpliv_obj = DoubleMLPLIV$new(n_folds = 2,
                                     ml_learners = mlmethod,
                                     params = params,
                                     dml_procedure = dml_procedure, inf_model = inf_model)
            double_mlpliv_obj$fit(data, y = 'y', d = 'd', z = 'z')
            return(list(coef = double_mlpliv_obj$coef,
                        se = double_mlpliv_obj$se))
        }
        ''')

r_IRM = robjects.r('''
        library('dml')
        library('mlr3learners')
        library('data.table')
        library('mlr3')

        f <- function(data, inf_model, dml_procedure) {
            data = data.table(data)
            mlmethod <- list(mlmethod_m = 'classif.log_reg',
                             mlmethod_g0 = 'regr.lm',
                             mlmethod_g1 = 'regr.lm')
            params <- list(params_m = list(),
                           params_g0 = list(),
                           params_g1 = list())

            double_mlirm_obj = DoubleMLIRM$new(n_folds = 2,
                                     ml_learners = mlmethod,
                                     params = params,
                                     dml_procedure = dml_procedure, inf_model = inf_model)
            double_mlirm_obj$fit(data, y = 'y', d = 'd')
            return(list(coef = double_mlirm_obj$coef,
                        se = double_mlirm_obj$se))
        }
        ''')

r_IIVM = robjects.r('''
        library('dml')
        library('mlr3learners')
        library('data.table')
        library('mlr3')

        f <- function(data, inf_model, dml_procedure) {
            data = data.table(data)
            mlmethod <- list(mlmethod_p = 'classif.log_reg',
                             mlmethod_mu0 = 'regr.lm',
                             mlmethod_mu1 = 'regr.lm',
                             mlmethod_m0 = 'classif.log_reg',
                             mlmethod_m1 = 'classif.log_reg')
            params <- list(params_p = list(),
                           params_mu0 = list(),
                           params_mu1 = list(),
                           params_m0 = list(),
                           params_m1 = list())

            double_mliivm_obj = DoubleMLIIVM$new(n_folds = 2,
                                     ml_learners = mlmethod,
                                     params = params,
                                     dml_procedure = dml_procedure, inf_model = inf_model)
            double_mliivm_obj$fit(data, y = 'y', d = 'd', z = 'z')
            return(list(coef = double_mliivm_obj$coef,
                        se = double_mliivm_obj$se))
        }
        ''')

