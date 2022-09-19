import numpy as np
import pandas as pd
import pytest
import math

from io import StringIO

from sklearn.base import clone

from sklearn.linear_model import LinearRegression, Lasso
from sklearn.svm import LinearSVR

from doubleml.double_ml_pcorr import DoubleMLPartialCorr


@pytest.fixture(scope='module',
                params=['orthogonal',
                        'corr'])
def score(request):
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
family,n_obs,dim_x,theta,coef,se,learner,dml_procedure,score
Clayton,500,20,3.0,0.785016556210597,0.0211990772585593,LinearRegression(),dml1,orthogonal
Clayton,500,20,3.0,0.7908882399437702,0.0563084007258299,LinearRegression(),dml1,corr
Clayton,500,20,3.0,0.7929717990184653,0.0561041705361666,Lasso(alpha=0.1),dml1,corr
Clayton,500,20,3.0,0.792923253219766,0.0561041704101541,Lasso(alpha=0.1),dml2,corr
Clayton,500,20,3.0,0.7903504862926403,0.020685888860453,Lasso(alpha=0.1),dml2,orthogonal
Clayton,500,20,3.0,0.7907665380609445,0.0563083999368849,LinearRegression(),dml2,corr
Clayton,500,20,3.0,0.7836655764338814,0.054874952616929,LinearSVR(max_iter=10000),dml2,corr
Clayton,500,20,3.0,0.7664260975500558,0.0222577782782518,LinearSVR(max_iter=10000),dml2,orthogonal
Clayton,500,20,3.0,0.7839093407218738,0.0548749558649809,LinearSVR(max_iter=10000),dml1,corr
Clayton,500,20,3.0,0.7664077548467634,0.0222582125883368,LinearSVR(max_iter=10000),dml1,orthogonal
Clayton,500,20,3.0,0.7849008864706752,0.0212020071857546,LinearRegression(),dml2,orthogonal
Clayton,500,20,3.0,0.790440612484851,0.0206836553979105,Lasso(alpha=0.1),dml1,orthogonal
Clayton,751,21,5.0,0.8663752833433993,0.0120950826509857,LinearRegression(),dml1,orthogonal
Clayton,751,21,5.0,0.8683372443805593,0.0462303421666814,LinearRegression(),dml1,corr
Clayton,751,21,5.0,0.8692645524139941,0.0446443559594651,Lasso(alpha=0.1),dml1,corr
Clayton,751,21,5.0,0.8685531585872669,0.0446443329149975,Lasso(alpha=0.1),dml2,corr
Clayton,751,21,5.0,0.8651777992802181,0.0126015712359625,Lasso(alpha=0.1),dml2,orthogonal
Clayton,751,21,5.0,0.8677849679248986,0.0462303287030357,LinearRegression(),dml2,corr
Clayton,751,21,5.0,0.8626391141960391,0.0465316519156563,LinearSVR(max_iter=10000),dml2,corr
Clayton,751,21,5.0,0.8557724590559994,0.0116017819984263,LinearSVR(max_iter=10000),dml2,orthogonal
Clayton,751,21,5.0,0.86330367613655,0.0465316712363759,LinearSVR(max_iter=10000),dml1,corr
Clayton,751,21,5.0,0.8570053087582163,0.0115763901919477,LinearSVR(max_iter=10000),dml1,orthogonal
Clayton,751,21,5.0,0.8651549018446776,0.012116301667269,LinearRegression(),dml2,orthogonal
Clayton,751,21,5.0,0.8668022657550633,0.0125788471566516,Lasso(alpha=0.1),dml1,orthogonal
Frank,500,20,3.0,0.424124411381283,0.0367139058000936,LinearRegression(),dml1,orthogonal
Frank,500,20,3.0,0.4264231319647027,0.0447352733073517,LinearRegression(),dml1,corr
Frank,500,20,3.0,0.4447186101167423,0.0446497525826323,Lasso(alpha=0.1),dml1,corr
Frank,500,20,3.0,0.4448318122373296,0.0446497517216303,Lasso(alpha=0.1),dml2,corr
Frank,500,20,3.0,0.4438516485510532,0.0355923658192673,Lasso(alpha=0.1),dml2,orthogonal
Frank,500,20,3.0,0.4264935520917238,0.0447352729748001,LinearRegression(),dml2,corr
Frank,500,20,3.0,0.4449518352780356,0.045226406028012,LinearSVR(max_iter=10000),dml2,corr
Frank,500,20,3.0,0.4374536918809536,0.0367304357316029,LinearSVR(max_iter=10000),dml2,orthogonal
Frank,500,20,3.0,0.4450249174946293,0.0452264063822942,LinearSVR(max_iter=10000),dml1,corr
Frank,500,20,3.0,0.4371521611139571,0.0367330316568197,LinearSVR(max_iter=10000),dml1,orthogonal
Frank,500,20,3.0,0.4243197622860055,0.0367122239798731,LinearRegression(),dml2,orthogonal
Frank,500,20,3.0,0.4436080773652,0.0355948479525148,Lasso(alpha=0.1),dml1,orthogonal
Frank,751,21,5.0,0.5753212860041824,0.025487622543942,LinearRegression(),dml1,orthogonal
Frank,751,21,5.0,0.5772991278083659,0.0384447779959259,LinearRegression(),dml1,corr
Frank,751,21,5.0,0.5792148371626971,0.0382569893808161,Lasso(alpha=0.1),dml1,corr
Frank,751,21,5.0,0.5782822571962621,0.0382569429828534,Lasso(alpha=0.1),dml2,corr
Frank,751,21,5.0,0.5747000563710413,0.0266364323792145,Lasso(alpha=0.1),dml2,orthogonal
Frank,751,21,5.0,0.5767517238847002,0.0384447620284203,LinearRegression(),dml2,corr
Frank,751,21,5.0,0.5774848035425141,0.0382183497476541,LinearSVR(max_iter=10000),dml2,corr
Frank,751,21,5.0,0.5728547929011871,0.0245962716529683,LinearSVR(max_iter=10000),dml2,orthogonal
Frank,751,21,5.0,0.5777872454804736,0.0382183547968706,LinearSVR(max_iter=10000),dml1,corr
Frank,751,21,5.0,0.5733604234653946,0.0245905657594367,LinearSVR(max_iter=10000),dml1,orthogonal
Frank,751,21,5.0,0.5742018235523945,0.0254991708288727,LinearRegression(),dml2,orthogonal
Frank,751,21,5.0,0.5766721107776791,0.0266206851990655,Lasso(alpha=0.1),dml1,orthogonal
Gaussian,500,20,0.3,0.2974870983668217,0.0404053010737537,LinearRegression(),dml1,orthogonal
Gaussian,500,20,0.3,0.2987979145908756,0.0458516418186316,LinearRegression(),dml1,corr
Gaussian,500,20,0.3,0.3182755348610983,0.0458758734525925,Lasso(alpha=0.1),dml1,corr
Gaussian,500,20,0.3,0.3182098099997944,0.0458758731701098,Lasso(alpha=0.1),dml2,corr
Gaussian,500,20,0.3,0.3171504656319521,0.0394197143578954,Lasso(alpha=0.1),dml2,orthogonal
Gaussian,500,20,0.3,0.2987765394240376,0.0458516417887375,LinearRegression(),dml2,corr
Gaussian,500,20,0.3,0.3128724309785949,0.0459261423938387,LinearSVR(max_iter=10000),dml2,corr
Gaussian,500,20,0.3,0.3080098665499862,0.0402927656801698,LinearSVR(max_iter=10000),dml2,orthogonal
Gaussian,500,20,0.3,0.3128144475402148,0.0459261426134384,LinearSVR(max_iter=10000),dml1,corr
Gaussian,500,20,0.3,0.3077216583156347,0.0402961525294087,LinearSVR(max_iter=10000),dml1,orthogonal
Gaussian,500,20,0.3,0.2974879361330424,0.0404052911536074,LinearRegression(),dml2,orthogonal
Gaussian,500,20,0.3,0.3172895298878174,0.0394177720066707,Lasso(alpha=0.1),dml1,orthogonal
Gaussian,378,21,0.7,0.6875540170617134,0.0275024041543685,LinearRegression(),dml1,orthogonal
Gaussian,378,21,0.7,0.6908754530447432,0.0603774443910145,LinearRegression(),dml1,corr
Gaussian,378,21,0.7,0.6996464642272304,0.0617925334748206,Lasso(alpha=0.1),dml1,corr
Gaussian,378,21,0.7,0.6995496640419805,0.0617925328731849,Lasso(alpha=0.1),dml2,corr
Gaussian,378,21,0.7,0.6967852293174709,0.026906115046322,Lasso(alpha=0.1),dml2,orthogonal
Gaussian,378,21,0.7,0.6915260179791072,0.0603774165825769,LinearRegression(),dml2,corr
Gaussian,378,21,0.7,0.6345202619394801,0.0578110670454579,LinearSVR(max_iter=10000),dml2,corr
Gaussian,378,21,0.7,0.6233984802137702,0.0312341351069788,LinearSVR(max_iter=10000),dml2,orthogonal
Gaussian,378,21,0.7,0.634486898218449,0.0578110671218641,LinearSVR(max_iter=10000),dml1,corr
Gaussian,378,21,0.7,0.6235242300842792,0.0312302183732054,LinearSVR(max_iter=10000),dml1,orthogonal
Gaussian,378,21,0.7,0.6889333522673382,0.0274566179439899,LinearRegression(),dml2,orthogonal
Gaussian,378,21,0.7,0.6970444148960535,0.0268968274647395,Lasso(alpha=0.1),dml1,orthogonal
Gumbel,500,20,3.0,0.8392447475281859,0.0145737243046634,LinearRegression(),dml1,orthogonal
Gumbel,500,20,3.0,0.845171571318042,0.0549583150399893,LinearRegression(),dml1,corr
Gumbel,500,20,3.0,0.8532471799070727,0.0558927767823582,Lasso(alpha=0.1),dml1,corr
Gumbel,500,20,3.0,0.8532755620745481,0.0558927767391248,Lasso(alpha=0.1),dml2,corr
Gumbel,500,20,3.0,0.8504792429825914,0.0137941798376175,Lasso(alpha=0.1),dml2,orthogonal
Gumbel,500,20,3.0,0.8451685222607324,0.0549583150394818,LinearRegression(),dml2,corr
Gumbel,500,20,3.0,0.8524046644238776,0.0541947925704087,LinearSVR(max_iter=10000),dml2,corr
Gumbel,500,20,3.0,0.8344010611981282,0.0138376423998269,LinearSVR(max_iter=10000),dml2,orthogonal
Gumbel,500,20,3.0,0.8524471942304361,0.0541947926705352,LinearSVR(max_iter=10000),dml1,corr
Gumbel,500,20,3.0,0.8341035987070853,0.0138462384717157,LinearSVR(max_iter=10000),dml1,orthogonal
Gumbel,500,20,3.0,0.839221039754254,0.01457436795236,LinearRegression(),dml2,orthogonal
Gumbel,500,20,3.0,0.850434800637281,0.0137954960274928,Lasso(alpha=0.1),dml1,orthogonal
Gumbel,751,21,5.0,0.934706044179728,0.0054254344005855,LinearRegression(),dml1,orthogonal
Gumbel,751,21,5.0,0.9371898811671948,0.048437478657045,LinearRegression(),dml1,corr
Gumbel,751,21,5.0,0.9385534006035824,0.0475309091588444,Lasso(alpha=0.1),dml1,corr
Gumbel,751,21,5.0,0.9385326185037648,0.0475309091402445,Lasso(alpha=0.1),dml2,corr
Gumbel,751,21,5.0,0.935346109230908,0.0055524576546628,Lasso(alpha=0.1),dml2,orthogonal
Gumbel,751,21,5.0,0.937162289568195,0.048437478624717,LinearRegression(),dml2,corr
Gumbel,751,21,5.0,0.9247012948542036,0.0480531029382231,LinearSVR(max_iter=10000),dml2,corr
Gumbel,751,21,5.0,0.920180398146392,0.0059909006673761,LinearSVR(max_iter=10000),dml2,orthogonal
Gumbel,751,21,5.0,0.9243470564660702,0.0480531082701033,LinearSVR(max_iter=10000),dml1,corr
Gumbel,751,21,5.0,0.9191178470592878,0.0060212533832936,LinearSVR(max_iter=10000),dml1,orthogonal
Gumbel,751,21,5.0,0.9347117809727916,0.0054252845412264,LinearRegression(),dml2,orthogonal
Gumbel,751,21,5,0.9353391395330848,0.00555262813597057,Lasso(alpha=0.1),dml1,orthogonal
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


@pytest.mark.filterwarnings("ignore:The classical score function")
@pytest.mark.ci
def test_dml_pcorr_coef(dml_pcorr_fixture):
    assert math.isclose(dml_pcorr_fixture['coef'],
                        dml_pcorr_fixture['coef_stored'],
                        rel_tol=1e-9, abs_tol=1e-4)


@pytest.mark.filterwarnings("ignore:The classical score function")
@pytest.mark.ci
def test_dml_pcorr_se(dml_pcorr_fixture):
    assert math.isclose(dml_pcorr_fixture['se'],
                        dml_pcorr_fixture['se_stored'],
                        rel_tol=1e-9, abs_tol=1e-4)
