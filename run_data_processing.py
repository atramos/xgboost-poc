from candidate_data_processors import dpp8,trainer
from sagemaker_sklearn_extension.externals import read_csv_data
import sys
import xgboost as xgb


X, y = read_csv_data(source='training-set.csv',                         target_column_index=dpp8.HEADER.target_column_index,output_dtype='O')
model = trainer.train(X,y,                                              dpp8.HEADER,                                                 dpp8.build_feature_transform(),dpp8.build_label_transform())
d=model.transform(X)

trf=dpp8.build_label_transform()
trf.fit(y)
y= trf.transform(y)

dtrain = xgb.DMatrix(d[1:,:], label=y[1:])
param = {'max_depth':2, 'eta':1 }
bst = xgb.train(param, dtrain)


