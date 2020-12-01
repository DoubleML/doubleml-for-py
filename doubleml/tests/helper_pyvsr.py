import pytest

rpy2 = pytest.importorskip("rpy2")
from rpy2 import robjects
from rpy2.robjects import ListVector
from rpy2.robjects.vectors import IntVector


def export_smpl_split_to_r(smpls):
    n_smpls = len(smpls)
    all_train = ListVector.from_length(n_smpls)
    all_test = ListVector.from_length(n_smpls)

    for idx, (train, test) in enumerate(smpls):
        all_train[idx] = IntVector(train + 1)
        all_test[idx] = IntVector(test + 1)

    return all_train, all_test


# The R code to fit the DML model
r_MLPLR = robjects.r('''
        library('DoubleML')
        library('mlr3learners')
        library('data.table')
        library('mlr3')

        f <- function(data, score, dml_procedure, train_ids, test_ids) {
            data = data.table(data)
            mlmethod_m = 'regr.lm'
            mlmethod_g = 'regr.lm'
            
            Xnames = names(data)[names(data) %in% c("y", "d") == FALSE]
            data_ml = double_ml_data_from_data_frame(data, y_col = "y", 
                                                     d_cols = "d", x_cols = Xnames)
            
            double_mlplr_obj = DoubleMLPLR$new(data_ml,
                                               n_folds = 2,
                                               ml_g = mlmethod_g,
                                               ml_m = mlmethod_m,
                                               dml_procedure = dml_procedure,
                                               score = score)
            
            smpls = list(list(train_ids=train_ids, test_ids=test_ids))
            double_mlplr_obj$set_sample_splitting(smpls)
            
            double_mlplr_obj$fit()
            return(list(coef = double_mlplr_obj$coef,
                        se = double_mlplr_obj$se))
        }
        ''')


r_MLPLIV = robjects.r('''
        library('DoubleML')
        library('mlr3learners')
        library('data.table')
        library('mlr3')

        f <- function(data, score, dml_procedure, train_ids, test_ids) {
            data = data.table(data)
            mlmethod_g = 'regr.lm'
            mlmethod_m = 'regr.lm'
            mlmethod_r = 'regr.lm'
            
            Xnames = names(data)[names(data) %in% c("y", "d", "Z1") == FALSE]
            data_ml = double_ml_data_from_data_frame(data, y_col = "y", 
                                                     d_cols = "d", x_cols = Xnames,
                                                     z_col = "Z1")
            
            double_mlpliv_obj = DoubleMLPLIV$new(data_ml,
                                                 n_folds = 2,
                                                 ml_g = mlmethod_g,
                                                 ml_m = mlmethod_m,
                                                 ml_r = mlmethod_r,
                                                 dml_procedure = dml_procedure,
                                                 score = score)
            
            smpls = list(list(train_ids=train_ids, test_ids=test_ids))
            double_mlpliv_obj$set_sample_splitting(smpls)
            
            double_mlpliv_obj$fit()
            return(list(coef = double_mlpliv_obj$coef,
                        se = double_mlpliv_obj$se))
        }
        ''')


r_MLPLIV_PARTIAL_X = robjects.r('''
        library('DoubleML')
        library('mlr3learners')
        library('data.table')
        library('mlr3')

        f <- function(data, score, dml_procedure, train_ids, test_ids) {
            data = data.table(data)
            mlmethod_g = 'regr.lm'
            mlmethod_m = 'regr.lm'
            mlmethod_r = 'regr.lm'

            Xnames = names(data)[grepl('X', names(data))]
            Znames = names(data)[grepl('Z', names(data))]
            data_ml = double_ml_data_from_data_frame(data, y_col = "y", 
                                                     d_cols = "d", x_cols = Xnames,
                                                     z_col = Znames)

            double_mlpliv_obj = DoubleML:::DoubleMLPLIV.partialX(data_ml,
                                                      n_folds = 2,
                                                      ml_g = mlmethod_g,
                                                      ml_m = mlmethod_m,
                                                      ml_r = mlmethod_r,
                                                      dml_procedure = dml_procedure,
                                                      score = score)

            smpls = list(list(train_ids=train_ids, test_ids=test_ids))
            double_mlpliv_obj$set_sample_splitting(smpls)

            double_mlpliv_obj$fit()
            return(list(coef = double_mlpliv_obj$coef,
                        se = double_mlpliv_obj$se))
        }
        ''')


r_MLPLIV_PARTIAL_Z = robjects.r('''
        library('DoubleML')
        library('mlr3learners')
        library('data.table')
        library('mlr3')

        f <- function(data, score, dml_procedure, train_ids, test_ids) {
            data = data.table(data)
            mlmethod_r = 'regr.lm'

            Xnames = names(data)[grepl('X', names(data))]
            Znames = names(data)[grepl('Z', names(data))]
            data_ml = double_ml_data_from_data_frame(data, y_col = "y", 
                                                     d_cols = "d", x_cols = Xnames,
                                                     z_col = Znames)

            double_mlpliv_obj = DoubleML:::DoubleMLPLIV.partialZ(data_ml,
                                                      n_folds = 2,
                                                      ml_r = mlmethod_r,
                                                      dml_procedure = dml_procedure,
                                                      score = score)

            smpls = list(list(train_ids=train_ids, test_ids=test_ids))
            double_mlpliv_obj$set_sample_splitting(smpls)

            double_mlpliv_obj$fit()
            return(list(coef = double_mlpliv_obj$coef,
                        se = double_mlpliv_obj$se))
        }
        ''')


r_MLPLIV_PARTIAL_XZ = robjects.r('''
        library('DoubleML')
        library('mlr3learners')
        library('data.table')
        library('mlr3')

        f <- function(data, score, dml_procedure, train_ids, test_ids) {
            data = data.table(data)
            mlmethod_g = 'regr.lm'
            mlmethod_m = 'regr.lm'
            mlmethod_r = 'regr.lm'

            Xnames = names(data)[grepl('X', names(data))]
            Znames = names(data)[grepl('Z', names(data))]
            data_ml = double_ml_data_from_data_frame(data, y_col = "y", 
                                                     d_cols = "d", x_cols = Xnames,
                                                     z_col = Znames)

            double_mlpliv_obj = DoubleML:::DoubleMLPLIV.partialXZ(data_ml,
                                                       n_folds = 2,
                                                       ml_g = mlmethod_g,
                                                       ml_m = mlmethod_m,
                                                       ml_r = mlmethod_r,
                                                       dml_procedure = dml_procedure,
                                                       score = score)

            smpls = list(list(train_ids=train_ids, test_ids=test_ids))
            double_mlpliv_obj$set_sample_splitting(smpls)

            double_mlpliv_obj$fit()
            return(list(coef = double_mlpliv_obj$coef,
                        se = double_mlpliv_obj$se))
        }
        ''')


r_IRM = robjects.r('''
        library('DoubleML')
        library('mlr3learners')
        library('data.table')
        library('mlr3')

        f <- function(data, score, dml_procedure, train_ids, test_ids) {
            data = data.table(data)
            mlmethod_g = 'regr.lm'
            mlmethod_m = 'classif.log_reg'
            
            Xnames = names(data)[names(data) %in% c("y", "d") == FALSE]
            data_ml = double_ml_data_from_data_frame(data, y_col = "y", 
                                                     d_cols = "d", x_cols = Xnames)
            
            double_mlirm_obj = DoubleMLIRM$new(data_ml,
                                               n_folds = 2,
                                               ml_g = mlmethod_g,
                                               ml_m = mlmethod_m,
                                               dml_procedure = dml_procedure,
                                               score = score)
            
            smpls = list(list(train_ids=train_ids, test_ids=test_ids))
            double_mlirm_obj$set_sample_splitting(smpls)
            
            double_mlirm_obj$fit()
            return(list(coef = double_mlirm_obj$coef,
                        se = double_mlirm_obj$se))
        }
        ''')


r_IIVM = robjects.r('''
        library('DoubleML')
        library('mlr3learners')
        library('data.table')
        library('mlr3')

        f <- function(data, score, dml_procedure, train_ids, test_ids) {
            data = data.table(data)
            ml_g = 'regr.lm'
            ml_m = 'classif.log_reg'
            ml_r = 'classif.log_reg'
            
            Xnames = names(data)[names(data) %in% c("y", "d", "z") == FALSE]
            data_ml = double_ml_data_from_data_frame(data, y_col = "y", 
                                                     d_cols = "d", x_cols = Xnames,
                                                     z_col = "z")

            double_mliivm_obj = DoubleMLIIVM$new(data_ml,
                                                 n_folds = 2,
                                                 ml_g = ml_g,
                                                 ml_m = ml_m,
                                                 ml_r = ml_r,
                                                 dml_procedure = dml_procedure,
                                                 score = score)
            
            smpls = list(list(train_ids=train_ids, test_ids=test_ids))
            double_mliivm_obj$set_sample_splitting(smpls)
            
            double_mliivm_obj$fit()
            return(list(coef = double_mliivm_obj$coef,
                        se = double_mliivm_obj$se))
        }
        ''')

