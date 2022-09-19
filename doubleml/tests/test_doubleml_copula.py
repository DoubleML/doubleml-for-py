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
                        LinearSVR(max_iter=10000)])
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
Clayton,500,20,3.0,2.04177396248945,0.1204374264249355,LinearSVR(max_iter=10000),False,dml2,likelihood
Clayton,500,20,3.0,2.216639544010257,0.1780570028085972,LinearSVR(max_iter=10000),False,dml2,orthogonal
Clayton,500,20,3.0,2.1087802351865355,0.1257377545123566,LinearSVR(max_iter=10000),False,dml1,likelihood
Clayton,500,20,3.0,2.243665606014625,0.1824538010945179,LinearSVR(max_iter=10000),False,dml1,orthogonal
Clayton,500,20,3.0,2.714996221302836,0.2161549473484053,LinearRegression(),False,dml2,orthogonal
Clayton,500,20,3.0,2.9311252090197817,0.2358248333418028,Lasso(alpha=0.1),False,dml1,orthogonal
Clayton,751,21,5.0,4.139888266293474,0.2399577808209823,LinearRegression(),False,dml1,orthogonal
Clayton,751,21,5.0,4.019696980633763,0.1634100723897752,LinearRegression(),False,dml1,likelihood
Clayton,751,21,5.0,4.322653952080724,0.1771793166636119,Lasso(alpha=0.1),False,dml1,likelihood
Clayton,751,21,5.0,4.318343920817255,0.1768966564709376,Lasso(alpha=0.1),False,dml2,likelihood
Clayton,751,21,5.0,4.421425918278544,0.2578856752501956,Lasso(alpha=0.1),False,dml2,orthogonal
Clayton,751,21,5.0,4.010406700723393,0.1628188532919586,LinearRegression(),False,dml2,likelihood
Clayton,751,21,5.0,3.25934610721292,0.1354812915547291,LinearSVR(max_iter=10000),False,dml2,likelihood
Clayton,751,21,5.0,3.4596815330525,0.1943411209478199,LinearSVR(max_iter=10000),False,dml2,orthogonal
Clayton,751,21,5.0,3.286855396442884,0.1372466849791128,LinearSVR(max_iter=10000),False,dml1,likelihood
Clayton,751,21,5.0,3.4945552770300674,0.1988259901619252,LinearSVR(max_iter=10000),False,dml1,orthogonal
Clayton,751,21,5.0,4.136795950635423,0.2395429404448309,LinearRegression(),False,dml2,orthogonal
Clayton,751,21,5.0,4.406708799676572,0.2558527370868762,Lasso(alpha=0.1),False,dml1,orthogonal
Frank,500,20,3.0,2.930048142571822,0.2939434292043041,LinearRegression(),False,dml1,orthogonal
Frank,500,20,3.0,2.927123148226359,0.2825371460533261,LinearRegression(),False,dml1,likelihood
Frank,500,20,3.0,3.064759796385161,0.2840395945092889,Lasso(alpha=0.1),False,dml1,likelihood
Frank,500,20,3.0,3.052990256817799,0.2836937670856621,Lasso(alpha=0.1),False,dml2,likelihood
Frank,500,20,3.0,3.056441345828248,0.2951791919842307,Lasso(alpha=0.1),False,dml2,orthogonal
Frank,500,20,3.0,2.922554624781972,0.2824098018659317,LinearRegression(),False,dml2,likelihood
Frank,500,20,3.0,3.062179435627723,0.2832375847961998,LinearSVR(max_iter=10000),False,dml2,likelihood
Frank,500,20,3.0,3.064192299990512,0.2931180099999145,LinearSVR(max_iter=10000),False,dml2,orthogonal
Frank,500,20,3.0,3.068563590317771,0.2834246641581959,LinearSVR(max_iter=10000),False,dml1,likelihood
Frank,500,20,3.0,3.1410839484918656,0.2957927330786973,LinearSVR(max_iter=10000),False,dml1,orthogonal
Frank,500,20,3.0,2.9276274625587155,0.2938607707713638,LinearRegression(),False,dml2,orthogonal
Frank,500,20,3.0,3.1286915900467074,0.2978111913338333,Lasso(alpha=0.1),False,dml1,orthogonal
Frank,751,21,5.0,4.4523708384041285,0.2748805248812199,LinearRegression(),False,dml1,orthogonal
Frank,751,21,5.0,4.449013597163932,0.2583477293935847,LinearRegression(),False,dml1,likelihood
Frank,751,21,5.0,4.579971053326512,0.2713177357584048,Lasso(alpha=0.1),False,dml1,likelihood
Frank,751,21,5.0,4.569161668678299,0.2709306772507462,Lasso(alpha=0.1),False,dml2,likelihood
Frank,751,21,5.0,4.587110281777852,0.284184946428858,Lasso(alpha=0.1),False,dml2,orthogonal
Frank,751,21,5.0,4.444806138624854,0.2582082661048024,LinearRegression(),False,dml2,likelihood
Frank,751,21,5.0,4.3644328205913485,0.2480259877103876,LinearSVR(max_iter=10000),False,dml2,likelihood
Frank,751,21,5.0,4.359607413317368,0.2678764147279798,LinearSVR(max_iter=10000),False,dml2,orthogonal
Frank,751,21,5.0,4.3661382886606095,0.2480785122359267,LinearSVR(max_iter=10000),False,dml1,likelihood
Frank,751,21,5.0,4.334510596882472,0.2669368123926763,LinearSVR(max_iter=10000),False,dml1,orthogonal
Frank,751,21,5.0,4.462033749040959,0.2752608094846291,LinearRegression(),False,dml2,orthogonal
Frank,751,21,5.0,4.579583885149329,0.2838764547823612,Lasso(alpha=0.1),False,dml1,orthogonal
Gaussian,500,20,0.3,0.2974911186903789,0.0401554555250889,LinearRegression(),False,dml1,orthogonal
Gaussian,500,20,0.3,0.296406693950865,0.038714555373594,LinearRegression(),False,dml1,likelihood
Gaussian,500,20,0.3,0.3164879049837943,0.0373190245666502,Lasso(alpha=0.1),False,dml1,likelihood
Gaussian,500,20,0.3,0.3164859383480843,0.0373191713444344,Lasso(alpha=0.1),False,dml2,likelihood
Gaussian,500,20,0.3,0.3171494160516647,0.0391956991066582,Lasso(alpha=0.1),False,dml2,orthogonal
Gaussian,500,20,0.3,0.2964373236069877,0.0387124199509896,LinearRegression(),False,dml2,likelihood
Gaussian,500,20,0.3,0.3034570579001908,0.0379953723993331,LinearSVR(max_iter=10000),False,dml2,likelihood
Gaussian,500,20,0.3,0.305584223802907,0.0392804944531069,LinearSVR(max_iter=10000),False,dml2,orthogonal
Gaussian,500,20,0.3,0.3036432136199186,0.0379827139754758,LinearSVR(max_iter=10000),False,dml1,likelihood
Gaussian,500,20,0.3,0.3154713789118484,0.0386011992382176,LinearSVR(max_iter=10000),False,dml1,orthogonal
Gaussian,500,20,0.3,0.297076362285839,0.0401832653240209,LinearRegression(),False,dml2,orthogonal
Gaussian,500,20,0.3,0.3238632220804097,0.0387131439531351,Lasso(alpha=0.1),False,dml1,orthogonal
Gaussian,378,21,0.7,0.7017266506560831,0.0244877060522771,LinearRegression(),False,dml1,orthogonal
Gaussian,378,21,0.7,0.6863462795296091,0.0223595578291002,LinearRegression(),False,dml1,likelihood
Gaussian,378,21,0.7,0.6961430217323712,0.0212174801650862,Lasso(alpha=0.1),False,dml1,likelihood
Gaussian,378,21,0.7,0.6961414028548646,0.0212177016540916,Lasso(alpha=0.1),False,dml2,likelihood
Gaussian,378,21,0.7,0.6970704042183374,0.0264334477718775,Lasso(alpha=0.1),False,dml2,orthogonal
Gaussian,378,21,0.7,0.6863295509782996,0.0223618816240352,LinearRegression(),False,dml2,likelihood
Gaussian,378,21,0.7,0.6187219544207153,0.0255059995818854,LinearSVR(max_iter=10000),False,dml2,likelihood
Gaussian,378,21,0.7,0.6239942177512532,0.0297897718560052,LinearSVR(max_iter=10000),False,dml2,orthogonal
Gaussian,378,21,0.7,0.6187392552809212,0.0255037721405918,LinearSVR(max_iter=10000),False,dml1,likelihood
Gaussian,378,21,0.7,0.6811987055498495,0.0210615737279121,LinearSVR(max_iter=10000),False,dml1,orthogonal
Gaussian,378,21,0.7,0.6878611652522505,0.0271749382955933,LinearRegression(),False,dml2,orthogonal
Gaussian,378,21,0.7,0.7089139334200556,0.0240909737463711,Lasso(alpha=0.1),False,dml1,orthogonal
Gumbel,500,20,3.0,2.812645379083889,0.1291543208072068,LinearRegression(),False,dml1,orthogonal
Gumbel,500,20,3.0,2.8338930018508286,0.1051435706908404,LinearRegression(),False,dml1,likelihood
Gumbel,500,20,3.0,2.9513086675090605,0.1105432170808855,Lasso(alpha=0.1),False,dml1,likelihood
Gumbel,500,20,3.0,2.938817088433657,0.1095862757684273,Lasso(alpha=0.1),False,dml2,likelihood
Gumbel,500,20,3.0,2.9253405936690813,0.1382550395477466,Lasso(alpha=0.1),False,dml2,orthogonal
Gumbel,500,20,3.0,2.828533639712606,0.104735360659339,LinearRegression(),False,dml2,likelihood
Gumbel,500,20,3.0,2.6793516035227607,0.0907163222314676,LinearSVR(max_iter=10000),False,dml2,likelihood
Gumbel,500,20,3.0,2.670705226361701,0.1122845518475913,LinearSVR(max_iter=10000),False,dml2,orthogonal
Gumbel,500,20,3.0,2.684554649985541,0.0910916285517505,LinearSVR(max_iter=10000),False,dml1,likelihood
Gumbel,500,20,3.0,2.7071478275823675,0.116453702749288,LinearSVR(max_iter=10000),False,dml1,orthogonal
Gumbel,500,20,3.0,2.8226125536946225,0.1303678070803468,LinearRegression(),False,dml2,orthogonal
Gumbel,500,20,3.0,2.9413505246826066,0.140264882569633,Lasso(alpha=0.1),False,dml1,orthogonal
Gumbel,751,21,5.0,4.518890637398259,0.1815731496010701,LinearRegression(),False,dml1,orthogonal
Gumbel,751,21,5.0,4.534566090182032,0.1361091749919077,LinearRegression(),False,dml1,likelihood
Gumbel,751,21,5.0,4.68841746266263,0.150201060533572,Lasso(alpha=0.1),False,dml1,likelihood
Gumbel,751,21,5.0,4.677439524581505,0.1494974477722136,Lasso(alpha=0.1),False,dml2,likelihood
Gumbel,751,21,5.0,4.627286015234464,0.1896923610842009,Lasso(alpha=0.1),False,dml2,orthogonal
Gumbel,751,21,5.0,4.530517591714636,0.1358647771160421,LinearRegression(),False,dml2,likelihood
Gumbel,751,21,5.0,3.826438615600424,0.1033521414957656,LinearSVR(max_iter=10000),False,dml2,likelihood
Gumbel,751,21,5.0,3.848904357636379,0.1399199999148842,LinearSVR(max_iter=10000),False,dml2,orthogonal
Gumbel,751,21,5.0,3.8529609512109286,0.1048621605595004,LinearSVR(max_iter=10000),False,dml1,likelihood
Gumbel,751,21,5.0,3.893807860516736,0.1442452705243862,LinearSVR(max_iter=10000),False,dml1,orthogonal
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


@pytest.mark.filterwarnings("ignore:The likelihood score function")
@pytest.mark.ci
def test_dml_partial_copula_coef(dml_partial_copula_fixture):
    assert math.isclose(dml_partial_copula_fixture['coef'],
                        dml_partial_copula_fixture['coef_stored'],
                        rel_tol=1e-9, abs_tol=1e-4)


@pytest.mark.filterwarnings("ignore:The likelihood score function")
@pytest.mark.ci
def test_dml_partial_copula_se(dml_partial_copula_fixture):
    assert math.isclose(dml_partial_copula_fixture['se'],
                        dml_partial_copula_fixture['se_stored'],
                        rel_tol=1e-9, abs_tol=1e-4)
