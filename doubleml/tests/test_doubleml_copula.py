import numpy as np
import pandas as pd
import pytest
import math

from io import StringIO

from sklearn.base import clone

from sklearn.linear_model import LinearRegression, Lasso
from sklearn.svm import LinearSVR

from doubleml.double_ml_copula import DoubleMLPartialCopula


@pytest.fixture(scope='module',
                params=[('orthogonal', False),
                        ('likelihood', False)])
def score_par_initial(request):
    return request.param


@pytest.fixture(scope='module',
                params=[LinearRegression(),
                        Lasso(alpha=0.1),
                        LinearSVR()])
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
Clayton,500,20,3.0,2.851887019770146,0.1569776123927299,Lasso(alpha=0.1),False,dml1,likelihood
Clayton,500,20,3.0,2.847402963817901,0.15664242606727,Lasso(alpha=0.1),False,dml2,likelihood
Clayton,500,20,3.0,2.895908670750728,0.2300398045551561,Lasso(alpha=0.1),False,dml2,orthogonal
Clayton,500,20,3.0,2.6189236468320347,0.1452796012016269,LinearRegression(),False,dml2,likelihood
Clayton,500,20,3.0,2.0383746305862864,0.1203287104905856,LinearSVR(),False,dml2,likelihood
Clayton,500,20,3.0,2.216493965739504,0.1778689413158268,LinearSVR(),False,dml2,orthogonal
Clayton,500,20,3.0,2.1005572424198,0.1252591395621895,LinearSVR(),False,dml1,likelihood
Clayton,500,20,3.0,2.2398344449737104,0.1816594162356163,LinearSVR(),False,dml1,orthogonal
Clayton,500,20,3.0,2.714996221302836,0.2161549473484053,LinearRegression(),False,dml2,orthogonal
Clayton,500,20,3.0,2.9311252090197817,0.2358248333418028,Lasso(alpha=0.1),False,dml1,orthogonal
Clayton,751,21,5.0,4.139888266293474,0.2399577808209823,LinearRegression(),False,dml1,orthogonal
Clayton,751,21,5.0,4.019696980633763,0.1634100723897752,LinearRegression(),False,dml1,likelihood
Clayton,751,21,5.0,4.322653952080724,0.1771793166636119,Lasso(alpha=0.1),False,dml1,likelihood
Clayton,751,21,5.0,4.318343920817255,0.1768966564709376,Lasso(alpha=0.1),False,dml2,likelihood
Clayton,751,21,5.0,4.421425918278544,0.2578856752501956,Lasso(alpha=0.1),False,dml2,orthogonal
Clayton,751,21,5.0,4.010406700723393,0.1628188532919586,LinearRegression(),False,dml2,likelihood
Clayton,751,21,5.0,3.2229023535284,0.1357606584710087,LinearSVR(),False,dml2,likelihood
Clayton,751,21,5.0,3.419751970915858,0.1928357288204158,LinearSVR(),False,dml2,orthogonal
Clayton,751,21,5.0,3.254578975213466,0.1378039997293081,LinearSVR(),False,dml1,likelihood
Clayton,751,21,5.0,3.4753036722838324,0.2000428531674425,LinearSVR(),False,dml1,orthogonal
Clayton,751,21,5.0,4.136795950635423,0.2395429404448309,LinearRegression(),False,dml2,orthogonal
Clayton,751,21,5.0,4.406708799676572,0.2558527370868762,Lasso(alpha=0.1),False,dml1,orthogonal
Frank,500,20,3.0,2.930048142571822,0.2939434292043041,LinearRegression(),False,dml1,orthogonal
Frank,500,20,3.0,2.927123148226359,0.2825371460533261,LinearRegression(),False,dml1,likelihood
Frank,500,20,3.0,3.0647597963851654,0.2840395945092878,Lasso(alpha=0.1),False,dml1,likelihood
Frank,500,20,3.0,3.052990256817812,0.2836937670856637,Lasso(alpha=0.1),False,dml2,likelihood
Frank,500,20,3.0,3.056441345828248,0.2951791919842307,Lasso(alpha=0.1),False,dml2,orthogonal
Frank,500,20,3.0,2.922554624781972,0.2824098018659317,LinearRegression(),False,dml2,likelihood
Frank,500,20,3.0,3.05128841453294,0.2832928041439224,LinearSVR(),False,dml2,likelihood
Frank,500,20,3.0,3.053242945056086,0.2933074484338382,LinearSVR(),False,dml2,orthogonal
Frank,500,20,3.0,3.057080209597426,0.2834620970690372,LinearSVR(),False,dml1,likelihood
Frank,500,20,3.0,3.1292266023680404,0.2959488051402639,LinearSVR(),False,dml1,orthogonal
Frank,500,20,3.0,2.9276274625587155,0.2938607707713638,LinearRegression(),False,dml2,orthogonal
Frank,500,20,3.0,3.12869159004671,0.297811191333834,Lasso(alpha=0.1),False,dml1,orthogonal
Frank,751,21,5.0,4.4523708384041285,0.2748805248812199,LinearRegression(),False,dml1,orthogonal
Frank,751,21,5.0,4.449013597163932,0.2583477293935847,LinearRegression(),False,dml1,likelihood
Frank,751,21,5.0,4.579971053326542,0.2713177357584226,Lasso(alpha=0.1),False,dml1,likelihood
Frank,751,21,5.0,4.569161668678281,0.2709306772507521,Lasso(alpha=0.1),False,dml2,likelihood
Frank,751,21,5.0,4.587110281777761,0.2841849464288453,Lasso(alpha=0.1),False,dml2,orthogonal
Frank,751,21,5.0,4.444806138624854,0.2582082661048024,LinearRegression(),False,dml2,likelihood
Frank,751,21,5.0,4.368167112864361,0.247475953151133,LinearSVR(),False,dml2,likelihood
Frank,751,21,5.0,4.367650293440152,0.2676409333102593,LinearSVR(),False,dml2,orthogonal
Frank,751,21,5.0,4.3699738873587535,0.2475313004873348,LinearSVR(),False,dml1,likelihood
Frank,751,21,5.0,4.343520969039804,0.2667391372042391,LinearSVR(),False,dml1,orthogonal
Frank,751,21,5.0,4.462033749040959,0.2752608094846291,LinearRegression(),False,dml2,orthogonal
Frank,751,21,5.0,4.579583885149293,0.2838764547823507,Lasso(alpha=0.1),False,dml1,orthogonal
Gaussian,500,20,0.3,0.2974911186903773,0.0401554555250891,LinearRegression(),False,dml1,orthogonal
Gaussian,500,20,0.3,0.296406693950865,0.038714555373594,LinearRegression(),False,dml1,likelihood
Gaussian,500,20,0.3,0.3164879049837943,0.0373190245666502,Lasso(alpha=0.1),False,dml1,likelihood
Gaussian,500,20,0.3,0.3164859383480843,0.0373191713444344,Lasso(alpha=0.1),False,dml2,likelihood
Gaussian,500,20,0.3,0.3171494160519004,0.0391956991066414,Lasso(alpha=0.1),False,dml2,orthogonal
Gaussian,500,20,0.3,0.2964373236069877,0.0387124199509896,LinearRegression(),False,dml2,likelihood
Gaussian,500,20,0.3,0.3036259312738747,0.0380915716265215,LinearSVR(),False,dml2,likelihood
Gaussian,500,20,0.3,0.3052977248568985,0.0394197926643504,LinearSVR(),False,dml2,orthogonal
Gaussian,500,20,0.3,0.3038017772271014,0.038079587343178,LinearSVR(),False,dml1,likelihood
Gaussian,500,20,0.3,0.3154429664299929,0.0387211775099045,LinearSVR(),False,dml1,orthogonal
Gaussian,500,20,0.3,0.2970763622858367,0.0401832653240211,LinearRegression(),False,dml2,orthogonal
Gaussian,500,20,0.3,0.3238632220804833,0.0387131439531298,Lasso(alpha=0.1),False,dml1,orthogonal
Gaussian,378,21,0.7,0.7017266506560914,0.0244877060522756,LinearRegression(),False,dml1,orthogonal
Gaussian,378,21,0.7,0.6863462795296122,0.0223595578290998,LinearRegression(),False,dml1,likelihood
Gaussian,378,21,0.7,0.6961430217323711,0.0212174801650862,Lasso(alpha=0.1),False,dml1,likelihood
Gaussian,378,21,0.7,0.6961414028548646,0.0212177016540916,Lasso(alpha=0.1),False,dml2,likelihood
Gaussian,378,21,0.7,0.6970704042183374,0.0264334477718775,Lasso(alpha=0.1),False,dml2,orthogonal
Gaussian,378,21,0.7,0.6863295509782996,0.0223618816240352,LinearRegression(),False,dml2,likelihood
Gaussian,378,21,0.7,0.6147977381750503,0.0257061331000723,LinearSVR(),False,dml2,likelihood
Gaussian,378,21,0.7,0.6201206629686802,0.0300118750107243,LinearSVR(),False,dml2,orthogonal
Gaussian,378,21,0.7,0.6148034065671478,0.0257054061149328,LinearSVR(),False,dml1,likelihood
Gaussian,378,21,0.7,0.6797569761879663,0.0209972451245596,LinearSVR(),False,dml1,orthogonal
Gaussian,378,21,0.7,0.6878611652522506,0.0271749382955932,LinearRegression(),False,dml2,orthogonal
Gaussian,378,21,0.7,0.7089139334200555,0.0240909737463711,Lasso(alpha=0.1),False,dml1,orthogonal
Gumbel,500,20,3.0,2.812645379083889,0.1291543208072068,LinearRegression(),False,dml1,orthogonal
Gumbel,500,20,3.0,2.8338930018508286,0.1051435706908404,LinearRegression(),False,dml1,likelihood
Gumbel,500,20,3.0,2.9513086675090605,0.1105432170808855,Lasso(alpha=0.1),False,dml1,likelihood
Gumbel,500,20,3.0,2.938817088433657,0.1095862757684273,Lasso(alpha=0.1),False,dml2,likelihood
Gumbel,500,20,3.0,2.9253405936690813,0.1382550395477466,Lasso(alpha=0.1),False,dml2,orthogonal
Gumbel,500,20,3.0,2.828533639712606,0.104735360659339,LinearRegression(),False,dml2,likelihood
Gumbel,500,20,3.0,2.6892640616246704,0.0914059054194332,LinearSVR(),False,dml2,likelihood
Gumbel,500,20,3.0,2.676385985033522,0.1127662920657106,LinearSVR(),False,dml2,orthogonal
Gumbel,500,20,3.0,2.694222997184153,0.09176440946438,LinearSVR(),False,dml1,likelihood
Gumbel,500,20,3.0,2.713508661117639,0.1170115107095381,LinearSVR(),False,dml1,orthogonal
Gumbel,500,20,3.0,2.8226125536946225,0.1303678070803468,LinearRegression(),False,dml2,orthogonal
Gumbel,500,20,3.0,2.9413505246826066,0.140264882569633,Lasso(alpha=0.1),False,dml1,orthogonal
Gumbel,751,21,5.0,4.518890637398259,0.1815731496010701,LinearRegression(),False,dml1,orthogonal
Gumbel,751,21,5.0,4.534566090182032,0.1361091749919077,LinearRegression(),False,dml1,likelihood
Gumbel,751,21,5.0,4.68841746266263,0.150201060533572,Lasso(alpha=0.1),False,dml1,likelihood
Gumbel,751,21,5.0,4.677439524581505,0.1494974477722136,Lasso(alpha=0.1),False,dml2,likelihood
Gumbel,751,21,5.0,4.627286015234464,0.1896923610842009,Lasso(alpha=0.1),False,dml2,orthogonal
Gumbel,751,21,5.0,4.530517591714636,0.1358647771160421,LinearRegression(),False,dml2,likelihood
Gumbel,751,21,5.0,3.807421270260601,0.1025914306383968,LinearSVR(),False,dml2,likelihood
Gumbel,751,21,5.0,3.8276944098411607,0.1389235206884187,LinearSVR(),False,dml2,orthogonal
Gumbel,751,21,5.0,3.8328751482698618,0.1040337270209033,LinearSVR(),False,dml1,likelihood
Gumbel,751,21,5.0,3.8627862225555214,0.1422812478351028,LinearSVR(),False,dml1,orthogonal
Gumbel,751,21,5.0,4.508031003168783,0.1804489047436976,LinearRegression(),False,dml2,orthogonal
Gumbel,751,21,5,4.643256105631871,0.19138588966754735,Lasso(alpha=0.1),False,dml1,orthogonal
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
