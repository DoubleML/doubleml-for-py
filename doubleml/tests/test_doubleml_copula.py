import numpy as np
import pandas as pd
import pytest
import math

from io import StringIO

from sklearn.base import clone

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

from doubleml.double_ml_copula import DoubleMLPartialCopula


@pytest.fixture(scope='module',
                params=[('orthogonal', False),
                        ('likelihood', False)])
def score_par_initial(request):
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
family,n_obs,dim_x,theta,coef,se,learner,par_initial,dml_procedure,score
Clayton,500,20,3.0,2.7235384061180303,0.2175374331387619,LinearRegression(),False,dml1,orthogonal
Clayton,500,20,3.0,2.6301557110790417,0.1461284613644576,LinearRegression(),False,dml1,likelihood
Clayton,500,20,3.0,2.398580836325331,0.1491760579834111,"RandomForestRegressor(max_depth=4, n_estimators=30)",False,dml1,likelihood
Clayton,500,20,3.0,2.3925664886220392,0.1486541893083326,"RandomForestRegressor(max_depth=4, n_estimators=30)",False,dml2,likelihood
Clayton,500,20,3.0,2.424166374484788,0.1807699421043955,"RandomForestRegressor(max_depth=4, n_estimators=30)",False,dml2,orthogonal
Clayton,500,20,3.0,2.6189236468320347,0.1452796012016269,LinearRegression(),False,dml2,likelihood
Clayton,500,20,3.0,2.714996221302836,0.2161549473484053,LinearRegression(),False,dml2,orthogonal
Clayton,500,20,3.0,2.414018698021544,0.1793328694545627,"RandomForestRegressor(max_depth=4, n_estimators=30)",False,dml1,orthogonal
Clayton,751,21,5.0,4.139888266293474,0.2399577808209823,LinearRegression(),False,dml1,orthogonal
Clayton,751,21,5.0,4.019696980633763,0.1634100723897752,LinearRegression(),False,dml1,likelihood
Clayton,751,21,5.0,3.485683323507639,0.16026707937611,"RandomForestRegressor(max_depth=4, n_estimators=30)",False,dml1,likelihood
Clayton,751,21,5.0,3.479603153990169,0.1598499593318482,"RandomForestRegressor(max_depth=4, n_estimators=30)",False,dml2,likelihood
Clayton,751,21,5.0,3.5361614678448694,0.2089459533522841,"RandomForestRegressor(max_depth=4, n_estimators=30)",False,dml2,orthogonal
Clayton,751,21,5.0,4.010406700723393,0.1628188532919586,LinearRegression(),False,dml2,likelihood
Clayton,751,21,5.0,4.136795950635423,0.2395429404448309,LinearRegression(),False,dml2,orthogonal
Clayton,751,21,5.0,3.5294333658996884,0.2080848599959124,"RandomForestRegressor(max_depth=4, n_estimators=30)",False,dml1,orthogonal
Frank,500,20,3.0,2.930048142571822,0.2939434292043041,LinearRegression(),False,dml1,orthogonal
Frank,500,20,3.0,2.927123148226359,0.2825371460533261,LinearRegression(),False,dml1,likelihood
Frank,500,20,3.0,3.251858186402817,0.2870001754916468,"RandomForestRegressor(max_depth=4, n_estimators=30)",False,dml1,likelihood
Frank,500,20,3.0,3.2279973585607373,0.2862720684377259,"RandomForestRegressor(max_depth=4, n_estimators=30)",False,dml2,likelihood
Frank,500,20,3.0,3.245723021185581,0.3008270873042967,"RandomForestRegressor(max_depth=4, n_estimators=30)",False,dml2,orthogonal
Frank,500,20,3.0,2.922554624781972,0.2824098018659317,LinearRegression(),False,dml2,likelihood
Frank,500,20,3.0,2.9276274625587155,0.2938607707713638,LinearRegression(),False,dml2,orthogonal
Frank,500,20,3.0,3.297184656741689,0.3027910405636311,"RandomForestRegressor(max_depth=4, n_estimators=30)",False,dml1,orthogonal
Frank,751,21,5.0,4.4523708384041285,0.2748805248812199,LinearRegression(),False,dml1,orthogonal
Frank,751,21,5.0,4.449013597163932,0.2583477293935847,LinearRegression(),False,dml1,likelihood
Frank,751,21,5.0,4.1984124952517625,0.2623879358067892,"RandomForestRegressor(max_depth=4, n_estimators=30)",False,dml1,likelihood
Frank,751,21,5.0,4.180071218645194,0.2618193916424874,"RandomForestRegressor(max_depth=4, n_estimators=30)",False,dml2,likelihood
Frank,751,21,5.0,4.209737882383287,0.279326673321059,"RandomForestRegressor(max_depth=4, n_estimators=30)",False,dml2,orthogonal
Frank,751,21,5.0,4.444806138624854,0.2582082661048024,LinearRegression(),False,dml2,likelihood
Frank,751,21,5.0,4.462033749040959,0.2752608094846291,LinearRegression(),False,dml2,orthogonal
Frank,751,21,5.0,4.195190756828999,0.278771340971097,"RandomForestRegressor(max_depth=4, n_estimators=30)",False,dml1,orthogonal
Gaussian,500,20,0.3,0.2974911186903773,0.0401554555250891,LinearRegression(),False,dml1,orthogonal
Gaussian,500,20,0.3,0.296406693950865,0.038714555373594,LinearRegression(),False,dml1,likelihood
Gaussian,500,20,0.3,0.3521836344589286,0.0365682446905633,"RandomForestRegressor(max_depth=4, n_estimators=30)",False,dml1,likelihood
Gaussian,500,20,0.3,0.3521229945984814,0.0365732714637029,"RandomForestRegressor(max_depth=4, n_estimators=30)",False,dml2,likelihood
Gaussian,500,20,0.3,0.3534679975168641,0.0387182142556661,"RandomForestRegressor(max_depth=4, n_estimators=30)",False,dml2,orthogonal
Gaussian,500,20,0.3,0.2964373236069877,0.0387124199509896,LinearRegression(),False,dml2,likelihood
Gaussian,500,20,0.3,0.2970763622858367,0.0401832653240211,LinearRegression(),False,dml2,orthogonal
Gaussian,500,20,0.3,0.3588635130759974,0.0382782972478518,"RandomForestRegressor(max_depth=4, n_estimators=30)",False,dml1,orthogonal
Gaussian,378,21,0.7,0.7017266506560914,0.0244877060522756,LinearRegression(),False,dml1,orthogonal
Gaussian,378,21,0.7,0.6863462795296122,0.0223595578290998,LinearRegression(),False,dml1,likelihood
Gaussian,378,21,0.7,0.6661102539254198,0.0231153725706949,"RandomForestRegressor(max_depth=4, n_estimators=30)",False,dml1,likelihood
Gaussian,378,21,0.7,0.6661023202164957,0.0231164421352081,"RandomForestRegressor(max_depth=4, n_estimators=30)",False,dml2,likelihood
Gaussian,378,21,0.7,0.6674862916857788,0.0288598237814438,"RandomForestRegressor(max_depth=4, n_estimators=30)",False,dml2,orthogonal
Gaussian,378,21,0.7,0.6863295509782996,0.0223618816240352,LinearRegression(),False,dml2,likelihood
Gaussian,378,21,0.7,0.6878611652522506,0.0271749382955932,LinearRegression(),False,dml2,orthogonal
Gaussian,378,21,0.7,0.6795912306813494,0.0265433570612625,"RandomForestRegressor(max_depth=4, n_estimators=30)",False,dml1,orthogonal
Gumbel,500,20,3.0,2.812645379083889,0.1291543208072068,LinearRegression(),False,dml1,orthogonal
Gumbel,500,20,3.0,2.8338930018508286,0.1051435706908404,LinearRegression(),False,dml1,likelihood
Gumbel,500,20,3.0,2.8507003769223864,0.1030070913177987,"RandomForestRegressor(max_depth=4, n_estimators=30)",False,dml1,likelihood
Gumbel,500,20,3.0,2.842496732457448,0.1023872361096882,"RandomForestRegressor(max_depth=4, n_estimators=30)",False,dml2,likelihood
Gumbel,500,20,3.0,2.87844502643444,0.1322575755330831,"RandomForestRegressor(max_depth=4, n_estimators=30)",False,dml2,orthogonal
Gumbel,500,20,3.0,2.828533639712606,0.104735360659339,LinearRegression(),False,dml2,likelihood
Gumbel,500,20,3.0,2.8226125536946225,0.1303678070803468,LinearRegression(),False,dml2,orthogonal
Gumbel,500,20,3.0,2.882119840187083,0.1327187911999591,"RandomForestRegressor(max_depth=4, n_estimators=30)",False,dml1,orthogonal
Gumbel,751,21,5.0,4.518890637398259,0.1815731496010701,LinearRegression(),False,dml1,orthogonal
Gumbel,751,21,5.0,4.534566090182032,0.1361091749919077,LinearRegression(),False,dml1,likelihood
Gumbel,751,21,5.0,3.9000489634283344,0.1155315480694146,"RandomForestRegressor(max_depth=4, n_estimators=30)",False,dml1,likelihood
Gumbel,751,21,5.0,3.8991741088842,0.1154790191624282,"RandomForestRegressor(max_depth=4, n_estimators=30)",False,dml2,likelihood
Gumbel,751,21,5.0,3.951706154032148,0.157664546355151,"RandomForestRegressor(max_depth=4, n_estimators=30)",False,dml2,orthogonal
Gumbel,751,21,5.0,4.530517591714636,0.1358647771160421,LinearRegression(),False,dml2,likelihood
Gumbel,751,21,5.0,4.508031003168783,0.1804489047436976,LinearRegression(),False,dml2,orthogonal
Gumbel,751,21,5,3.9536446969074674,0.1578680402519666,"RandomForestRegressor(max_depth=4, n_estimators=30)",False,dml1,orthogonal
"""


@pytest.fixture(scope='module')
def dml_partial_copula_fixture(generate_data_partial_copula, learner, score_par_initial, dml_procedure):
    n_folds = 2

    score = score_par_initial[0]
    par_initial = score_par_initial[1]

    # collect data
    data = generate_data_partial_copula
    dml_data = data['data']
    pars = data['pars']
    copula_family = pars[0]

    # Set machine learning methods for m & g
    ml_g = clone(learner)
    ml_m = clone(learner)

    np.random.seed(3141)
    dml_copula = DoubleMLPartialCopula(dml_data, copula_family,
                                       ml_g, ml_m,
                                       n_folds=n_folds,
                                       dml_procedure=dml_procedure,
                                       score=score)

    dml_copula.fit()

    s = StringIO(res)
    xx = pd.read_csv(s)

    # xx = pd.read_csv('res.csv')
    # df = pd.DataFrame(pars, index=['family', 'n_obs', 'dim_x', 'theta']).transpose()
    # df['coef'] = dml_copula.coef
    # df['se'] = dml_copula.se
    # df['learner'] = str(learner)
    # df['par_initial'] = par_initial
    # df['dml_procedure'] = dml_procedure
    # df['score'] = score
    # df = pd.concat((xx, df))
    # df.to_csv('res.csv', index=False)

    df = xx[(xx.family == pars[0]) & (xx.n_obs == pars[1]) & (xx.dim_x == pars[2]) & (xx.theta == pars[3]) &
            (xx.learner == str(learner)) & (xx.dml_procedure == dml_procedure) &
            (xx.par_initial == par_initial) & (xx.score == score)]
    df.reset_index(inplace=True)
    assert df.shape[0] == 1

    res_dict = {'coef': dml_copula.coef,
                'se': dml_copula.se,
                'coef_stored': df.coef[0],
                'se_stored': df.se[0]
                }

    return res_dict


@pytest.mark.ci
def test_dml_partial_copula_coef(dml_partial_copula_fixture):
    assert math.isclose(dml_partial_copula_fixture['coef'],
                        dml_partial_copula_fixture['coef_stored'],
                        rel_tol=1e-9, abs_tol=1e-4)


@pytest.mark.ci
def test_dml_partial_copula_se(dml_partial_copula_fixture):
    assert math.isclose(dml_partial_copula_fixture['se'],
                        dml_partial_copula_fixture['se_stored'],
                        rel_tol=1e-9, abs_tol=1e-4)
