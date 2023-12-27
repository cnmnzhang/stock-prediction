import pandas as pd
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.metrics import mean_squared_error

# Read train data
df = pd.read_csv('./data/train.csv')
# Randomly use 30000 of your dataframe
# df = df_full.sample(n=30000, random_state=1)


train, test = train_test_split(df, test_size=0.2, random_state=42)
# Create DMatrix on train data
dtrain = xgb.DMatrix(data=train[['store', 'item']],
                     label=train['sales'])

# Define xgboost parameters
params = {  'objective': 'reg:linear',
            'max_depth': 2,
            'verbosity': 0}

# Train xgboost model
xg_depth_2 = xgb.train(params=params, dtrain=dtrain)


# Define xgboost parameters
params = {  'objective': 'reg:linear',
            'max_depth': 8,
            'verbosity': 0}

# Train xgboost model
xg_depth_8 = xgb.train(params=params, dtrain=dtrain)

# Define xgboost parameters
params = {  'objective': 'reg:linear',
            'max_depth': 15,
            'verbosity': 0}

# Train xgboost model
xg_depth_15 = xgb.train(params=params, dtrain=dtrain)

# Define xgboost parameters
params = {  'objective': 'reg:linear',
            'max_depth': 30,
            'verbosity': 0}

# Train xgboost model
xg_depth_30 = xgb.train(params=params, dtrain=dtrain)


dtrain_ = xgb.DMatrix(data=train[['store', 'item']])
dtest = xgb.DMatrix(data=test[['store', 'item']])
print(len(train), len(test))

# For each of 3 trained models
for model in [xg_depth_2, xg_depth_8, xg_depth_15,xg_depth_30]:
    # Make predictions
    train_pred = model.predict(dtrain_)     
    test_pred = model.predict(dtest)          
    
    # Calculate metrics
    mse_train = mean_squared_error(train['sales'], train_pred)                  
    mse_test = mean_squared_error(test['sales'], test_pred)
    print('MSE Train: {:.3f}. MSE Test: {:.3f}'.format(mse_train, mse_test))