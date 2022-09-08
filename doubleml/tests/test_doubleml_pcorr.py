import numpy as np
import pandas as pd
import pytest
import math

from io import StringIO

from sklearn.base import clone

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

from doubleml.double_ml_pcorr import DoubleMLPartialCorr


@pytest.fixture(scope='module',
                params=['orthogonal',
                        'corr'])
def score(request):
    return request.param


@pytest.fixture(scope='module',
                params=[LinearRegression(),
                        RandomForestRegressor(max_depth=4, n_estimators=30)])
def learner(request):
    return request.param


@pytest.fixture(scope='module',
                params=['dml1', 'dml2'])
def dml_procedure(request):
    return request.param


res = """
family,n_obs,dim_x,theta,coef,se,learner,dml_procedure,score
Clayton,500,20,3.0,0.785016556210597,0.0211990772585593,LinearRegression(),dml1,orthogonal
Clayton,500,20,3.0,0.7908882399437702,0.0563084007258299,LinearRegression(),dml1,corr
Clayton,500,20,3.0,0.800307235189689,0.053219834843907,"RandomForestRegressor(max_depth=4, n_estimators=30)",dml1,corr
Clayton,500,20,3.0,0.8003125231052852,0.0532198348423308,"RandomForestRegressor(max_depth=4, n_estimators=30)",dml2,corr
Clayton,500,20,3.0,0.796516226493404,0.0181382680476087,"RandomForestRegressor(max_depth=4, n_estimators=30)",dml2,orthogonal
Clayton,500,20,3.0,0.7907665380609445,0.0563083999368849,LinearRegression(),dml2,corr
Clayton,500,20,3.0,0.7849008864706752,0.0212020071857546,LinearRegression(),dml2,orthogonal
Clayton,500,20,3.0,0.7965084349304221,0.0181384537525967,"RandomForestRegressor(max_depth=4, n_estimators=30)",dml1,orthogonal
Clayton,751,21,5.0,0.8663752833433993,0.0120950826509857,LinearRegression(),dml1,orthogonal
Clayton,751,21,5.0,0.8683372443805593,0.0462303421666814,LinearRegression(),dml1,corr
Clayton,751,21,5.0,0.855930432426392,0.0463848624320309,"RandomForestRegressor(max_depth=4, n_estimators=30)",dml1,corr
Clayton,751,21,5.0,0.8556551190056527,0.0463848590632257,"RandomForestRegressor(max_depth=4, n_estimators=30)",dml2,corr
Clayton,751,21,5.0,0.8525387471195819,0.012787979435383,"RandomForestRegressor(max_depth=4, n_estimators=30)",dml2,orthogonal
Clayton,751,21,5.0,0.8677849679248986,0.0462303287030357,LinearRegression(),dml2,corr
Clayton,751,21,5.0,0.8651549018446776,0.012116301667269,LinearRegression(),dml2,orthogonal
Clayton,751,21,5.0,0.8532520057321199,0.0127750042306664,"RandomForestRegressor(max_depth=4, n_estimators=30)",dml1,orthogonal
Frank,500,20,3.0,0.424124411381283,0.0367139058000936,LinearRegression(),dml1,orthogonal
Frank,500,20,3.0,0.4264231319647027,0.0447352733073517,LinearRegression(),dml1,corr
Frank,500,20,3.0,0.4742784627975791,0.0455963323657532,"RandomForestRegressor(max_depth=4, n_estimators=30)",dml1,corr
Frank,500,20,3.0,0.4746348281803957,0.0455963240103948,"RandomForestRegressor(max_depth=4, n_estimators=30)",dml2,corr
Frank,500,20,3.0,0.473496263143277,0.0344206136001644,"RandomForestRegressor(max_depth=4, n_estimators=30)",dml2,orthogonal
Frank,500,20,3.0,0.4264935520917238,0.0447352729748001,LinearRegression(),dml2,corr
Frank,500,20,3.0,0.4243197622860055,0.0367122239798731,LinearRegression(),dml2,orthogonal
Frank,500,20,3.0,0.4727692289848898,0.0344302065207366,"RandomForestRegressor(max_depth=4, n_estimators=30)",dml1,orthogonal
Frank,751,21,5.0,0.5753212860041824,0.025487622543942,LinearRegression(),dml1,orthogonal
Frank,751,21,5.0,0.5772991278083659,0.0384447779959259,LinearRegression(),dml1,corr
Frank,751,21,5.0,0.5548643328373402,0.0403248850358926,"RandomForestRegressor(max_depth=4, n_estimators=30)",dml1,corr
Frank,751,21,5.0,0.5540329121113372,0.0403248498726468,"RandomForestRegressor(max_depth=4, n_estimators=30)",dml2,corr
Frank,751,21,5.0,0.5502533193394885,0.027930811380898,"RandomForestRegressor(max_depth=4, n_estimators=30)",dml2,orthogonal
Frank,751,21,5.0,0.5767517238847002,0.0384447620284203,LinearRegression(),dml2,corr
Frank,751,21,5.0,0.5742018235523945,0.0254991708288727,LinearRegression(),dml2,orthogonal
Frank,751,21,5.0,0.5521881988158363,0.027908678621988,"RandomForestRegressor(max_depth=4, n_estimators=30)",dml1,orthogonal
Gaussian,500,20,0.3,0.2974870983668217,0.0404053010737537,LinearRegression(),dml1,orthogonal
Gaussian,500,20,0.3,0.2987979145908756,0.0458516418186316,LinearRegression(),dml1,corr
Gaussian,500,20,0.3,0.3543174280443483,0.0466734389029534,"RandomForestRegressor(max_depth=4, n_estimators=30)",dml1,corr
Gaussian,500,20,0.3,0.3542034373035431,0.0466734380677551,"RandomForestRegressor(max_depth=4, n_estimators=30)",dml2,corr
Gaussian,500,20,0.3,0.3528439798459306,0.0390184206295324,"RandomForestRegressor(max_depth=4, n_estimators=30)",dml2,orthogonal
Gaussian,500,20,0.3,0.2987765394240376,0.0458516417887375,LinearRegression(),dml2,corr
Gaussian,500,20,0.3,0.2974879361330424,0.0404052911536074,LinearRegression(),dml2,orthogonal
Gaussian,500,20,0.3,0.35311775280105,0.0390143487677432,"RandomForestRegressor(max_depth=4, n_estimators=30)",dml1,orthogonal
Gaussian,378,21,0.7,0.6875540170617134,0.0275024041543685,LinearRegression(),dml1,orthogonal
Gaussian,378,21,0.7,0.6908754530447432,0.0603774443910145,LinearRegression(),dml1,corr
Gaussian,378,21,0.7,0.6695909533692574,0.0632544202958506,"RandomForestRegressor(max_depth=4, n_estimators=30)",dml1,corr
Gaussian,378,21,0.7,0.6696160275206153,0.0632544202564085,"RandomForestRegressor(max_depth=4, n_estimators=30)",dml2,corr
Gaussian,378,21,0.7,0.6671083711961121,0.029210880627205,"RandomForestRegressor(max_depth=4, n_estimators=30)",dml2,orthogonal
Gaussian,378,21,0.7,0.6915260179791072,0.0603774165825769,LinearRegression(),dml2,corr
Gaussian,378,21,0.7,0.6889333522673382,0.0274566179439899,LinearRegression(),dml2,orthogonal
Gaussian,378,21,0.7,0.667082509962698,0.029211829308024,"RandomForestRegressor(max_depth=4, n_estimators=30)",dml1,orthogonal
Gumbel,500,20,3.0,0.8392447475281859,0.0145737243046634,LinearRegression(),dml1,orthogonal
Gumbel,500,20,3.0,0.845171571318042,0.0549583150399893,LinearRegression(),dml1,corr
Gumbel,500,20,3.0,0.852841187599964,0.0536140883027092,"RandomForestRegressor(max_depth=4, n_estimators=30)",dml1,corr
Gumbel,500,20,3.0,0.8527724568572526,0.053614088038411,"RandomForestRegressor(max_depth=4, n_estimators=30)",dml2,corr
Gumbel,500,20,3.0,0.8486497573110123,0.013701373222273,"RandomForestRegressor(max_depth=4, n_estimators=30)",dml2,orthogonal
Gumbel,500,20,3.0,0.8451685222607324,0.0549583150394818,LinearRegression(),dml2,corr
Gumbel,500,20,3.0,0.839221039754254,0.01457436795236,LinearRegression(),dml2,orthogonal
Gumbel,500,20,3.0,0.8487723689472468,0.0136980651595235,"RandomForestRegressor(max_depth=4, n_estimators=30)",dml1,orthogonal
Gumbel,751,21,5.0,0.934706044179728,0.0054254344005855,LinearRegression(),dml1,orthogonal
Gumbel,751,21,5.0,0.9371898811671948,0.048437478657045,LinearRegression(),dml1,corr
Gumbel,751,21,5.0,0.9219833793147598,0.0485256701676625,"RandomForestRegressor(max_depth=4, n_estimators=30)",dml1,corr
Gumbel,751,21,5.0,0.922044417379134,0.0485256700068038,"RandomForestRegressor(max_depth=4, n_estimators=30)",dml2,corr
Gumbel,751,21,5.0,0.9187743518566605,0.0066796510608135,"RandomForestRegressor(max_depth=4, n_estimators=30)",dml2,orthogonal
Gumbel,751,21,5.0,0.937162289568195,0.048437478624717,LinearRegression(),dml2,corr
Gumbel,751,21,5.0,0.9347117809727916,0.0054252845412264,LinearRegression(),dml2,orthogonal
Gumbel,751,21,5,0.9186411833435042,0.006683309097016636,"RandomForestRegressor(max_depth=4, n_estimators=30)",dml1,orthogonal
"""


@pytest.fixture(scope='module')
def dml_pcorr_fixture(generate_data_partial_copula, learner, score, dml_procedure):
    n_folds = 2

    # collect data
    data = generate_data_partial_copula
    dml_data = data['data']
    pars = data['pars']

    # Set machine learning methods for m & g
    ml_g = clone(learner)
    ml_m = clone(learner)

    np.random.seed(3141)
    dml_pcorr = DoubleMLPartialCorr(dml_data,
                                    ml_g, ml_m,
                                    n_folds=n_folds,
                                    dml_procedure=dml_procedure,
                                    score=score)

    dml_pcorr.fit()

    s = StringIO(res)
    xx = pd.read_csv(s)

    # xx = pd.read_csv('res.csv')
    # df = pd.DataFrame(pars, index=['family', 'n_obs', 'dim_x', 'theta']).transpose()
    # df['coef'] = dml_pcorr.coef
    # df['se'] = dml_pcorr.se
    # df['learner'] = str(learner)
    # df['dml_procedure'] = dml_procedure
    # df['score'] = score
    # df = pd.concat((xx, df))
    # df.to_csv('res.csv', index=False)

    df = xx[(xx.family == pars[0]) & (xx.n_obs == pars[1]) & (xx.dim_x == pars[2]) & (xx.theta == pars[3]) &
            (xx.learner == str(learner)) & (xx.dml_procedure == dml_procedure) &
            (xx.score == score)]
    df.reset_index(inplace=True)
    assert df.shape[0] == 1

    res_dict = {'coef': dml_pcorr.coef,
                'se': dml_pcorr.se,
                'coef_stored': df.coef[0],
                'se_stored': df.se[0]
                }

    return res_dict


@pytest.mark.ci
def test_dml_pcorr_coef(dml_pcorr_fixture):
    assert math.isclose(dml_pcorr_fixture['coef'],
                        dml_pcorr_fixture['coef_stored'],
                        rel_tol=1e-9, abs_tol=1e-4)


@pytest.mark.ci
def test_dml_pcorr_se(dml_pcorr_fixture):
    assert math.isclose(dml_pcorr_fixture['se'],
                        dml_pcorr_fixture['se_stored'],
                        rel_tol=1e-9, abs_tol=1e-4)
