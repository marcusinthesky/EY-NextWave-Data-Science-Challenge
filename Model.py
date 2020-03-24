import os
import abc
import datetime
from toolz.curried import *
import pandas as pd
import holoviews as hv
from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures, \
                                  StandardScaler, \
                                  FunctionTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline, make_union


def Imputer(model_: abc.ABCMeta, 
            data_: pd.DataFrame, 
            X_:  list, y_: list)-> pd.DataFrame:
    
    def predict(model__: abc.ABCMeta, 
                data__: pd.DataFrame, 
                X__:  list, y__: list) -> pd.np.array:
        not_nul = data_.loc[:,X__+ y__].notnull().all(1)
        
        model_instance = model__.fit(X = data__.loc[not_nul,X__],
                                     y = data__.loc[not_nul,y__].values.ravel())
        
        return model_instance.predict(X=data_.loc[:,X__])
    
    
    return data_.assign(**{ys+'_predicted': predict(model__ = model_, 
                                                    data__ = data_,
                                                    X__ = X_,
                                                    y__ = list(ys))
                           for ys in y_})

def basis(x:pd.np.array, knot:float = 12*60*60)-> pd.np.array:
    x = x - knot
    x = x * (x>0.0)
    x = x**2 * pd.np.log(x, where= x > 0.0)
    
    return x.astype(pd.np.float64)


if __name__ == '__main__':
    path = os.path.join('Data','data_test.csv')
    data = pd.read_csv(path, index_col = 'Unnamed: 0')

    id_ = ['hash','trajectory_id','vmax','vmin','vmean']
    midnight = pd.to_datetime('00:00:00')

    views = data.pipe(partial(pd.wide_to_long,
                              stubnames = ['x','y','time'],
                              i = id_,
                              j = 'pop',
                              sep = '_',
                              suffix = '(!?entry|exit)'))\
                .reset_index()\
                .replace(['entry','exit'],[0,1])\
                .assign(time = lambda d: d.time\
                                          .pipe(pd.to_datetime)\
                                          .subtract(midnight)\
                                          .dt.total_seconds(),
                        missing = lambda d: d.x.isna().astype(pd.np.int))

    pipeline = make_pipeline(make_union(PolynomialFeatures(degree=3, include_bias=False),
                                    FunctionTransformer(basis,
                                                        validate=False)),
                         linear_model.BayesianRidge(fit_intercept=True))

    imputed = views.groupby(['hash'])\
                    .apply(lambda g: Imputer(model_=pipeline, data_=g, X_=['time'], y_=['y','x']))\
                    .reset_index(drop=True)
    
    imputed_path = pipe(datetime.datetime.now().timestamp(), 
                           partial(round, ndigits=0), 
                           int, 
                           lambda x: os.path.join('Data',f'imputed_{x}.csv'))
    imputed.to_csv(imputed_path, index=False, sep=',')
    
    quantile = 0.25
    output_path = pipe(datetime.datetime.now().timestamp(), 
                           partial(round, ndigits=0), 
                           int, 
                           lambda x: os.path.join('Data',f'submission_{x}.csv'))

    imputed.assign(city = lambda d: ((d.x_predicted <d.x.quantile(0.5+quantile)) &\
                                     (d.x_predicted >d.x.quantile(0.5-quantile)) &\
                                     (d.y_predicted <d.y.quantile(0.5+quantile)) &\
                                     (d.y_predicted >d.y.quantile(0.5-quantile))).astype(pd.np.int))\
            .where(lambda x: x.missing==1)\
            .dropna(axis=0, how='all')\
            .loc[:,['trajectory_id','city']]\
            .rename(columns={'trajectory_id':'id','city':'target'})\
            .assign(target = lambda x: x.target.astype(int))\
            .to_csv(output_path, index=False, sep=',')