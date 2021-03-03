import pandas as pd
import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt
import warnings
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor

warnings.filterwarnings('ignore')

# Loading Dataset
train_data = pd.read_csv('Train.csv')
# print(train_data.head())

# Analysing the dataset
# print(train_data.describe())
# print(train_data.info())
# print(train_data.apply(lambda x: len(x.unique()))) # checking unique values in dataset


# Preprocessing the dataset


# checking for categorical attributes
cat_col = []
for q in train_data.dtypes.index:
    if train_data.dtypes[q] == 'object':
        cat_col.append(q)
# print(cat_col)
cat_col.remove('Item_Identifier')
cat_col.remove('Outlet_Identifier')
# print(cat_col)

# printing the categorical columns
# for i in cat_col:
# print(i)
# print(train_data[i].value_counts())
# print()

# creating new attributes assigning meaningful name
train_data['New_Item_Type'] = train_data['Item_Identifier'].apply(lambda c: c[: 2])
# print(train_data['New_Item_Type'])
train_data['New_Item_Type'] = train_data['New_Item_Type'].map({'FD': 'Food', 'NC': 'Non-Consumable', 'DR': 'Drinks'})
# print(train_data['New_Item_Type'].value_counts())


# checking null value filling missing value
# print(train_data.isnull().sum())
item_weight_mean = train_data.pivot_table(values='Item_Weight', index='Item_Identifier')
# print(item_weight_mean)
missing_bool = train_data['Item_Weight'].isnull()
# print(missing_bool)

train_data['Item_Weight'].fillna(train_data['Item_Weight'].mean(), inplace=True)

# print(train_data['Item_Identifier'].isnull().sum())

outlet_size_mode = train_data.pivot_table(values='Outlet_Size', columns='Outlet_Type', aggfunc=(lambda o: o.mode()[0]))
# print(outlet_size_mode)
missing_bol = train_data['Outlet_Size'].isnull()
train_data.loc[missing_bol, 'Outlet_Size'] = train_data.loc[missing_bol, 'Outlet_Type'].apply(
    lambda f: outlet_size_mode[f])

# print(sum(train_data['Item_Visibility'] == 0))
# replacing zeros into mean value
train_data.loc[:, 'Item_Visibility'].replace([0], [train_data['Item_Visibility'].mean()], inplace=True)
# print(sum(train_data['Item_Visibility'] == 0))

# aggregating item fat content / combining
# print(train_data['Item_Fat_Content'].value_counts())
train_data['Item_Fat_Content'] = train_data['Item_Fat_Content'].replace(
    {'LF': 'Low Fat', 'reg': 'Regular', 'low fat': 'Low Fat'})
# print(train_data['Item_Fat_Content'].value_counts())

# creating new attributes
train_data.loc[train_data['New_Item_Type'] == 'Non-Consumable', 'Item_Fat_Content'] = 'Non-Edible'
# print(train_data['Item_Fat_Content'].value_counts())

# creating small values for higher accuracy in prediction
train_data['Outlet_Years'] = 2020 - train_data['Outlet_Establishment_Year']
# print(train_data['Outlet_Years'])

# Exploratory Data Analysis
# print(train_data.head())

# sb.distplot(train_data['Item_Weight'])
# sb.distplot(train_data['Item_Visibility'])
# sb.distplot(train_data['Item_MRP'])
# sb.distplot(train_data['Item_Outlet_Sales'])
# sb.countplot(train_data["Item_Fat_Content"])
# plt.figure(figsize=(20,5))

# li = list(train_data['Item_Type'].unique())
# chart = sb.countplot(train_data["Item_Type"])
# chart.set_xticklabels(labels=li, rotation=90)

# sb.countplot(train_data['Outlet_Establishment_Year'])
# sb.countplot(train_data['Outlet_Size'])
# sb.countplot(train_data['Outlet_Location_Type'])

# li2 = list(train_data['Outlet_Type'].unique())
# chart2 = sb.countplot(train_data["Outlet_Type"])
# chart2.set_xticklabels(labels=li2, rotation=90)
# plt.show()

# Normalization
# log transformation
train_data['Item_Outlet_Sales'] = np.log(1 + train_data['Item_Outlet_Sales'])
# sb.distplot(train_data['Item_Outlet_Sales'])
# plt.show()

# Correlation Matrix

# correlation = train_data.corr()
# sb.heatmap(correlation, annot=True , cmap='coolwarm')
# plt.show()


# Label Encoding
'''
le = LabelEncoder()
train_data['Outlet'] = le.fit_transform(train_data['Outlet_Identifier'])
cat_col = ['Item_Fat_Content', 'Item_Type', 'Outlet_Size', 'Outlet_Location_Type', 'Outlet_Type', 'New_Item_Type']
for col in cat_col:
    train_data[col] = le.fit_transform(train_data[col])
# print(train_data[col])
'''
# Onehot Encoding

train_data = pd.get_dummies(train_data,
                            columns=['Item_Fat_Content', 'Outlet_Size', 'Outlet_Location_Type', 'Outlet_Type',
                                     'New_Item_Type'])

# Train - Test Split

# input Split
x = train_data.drop(
    columns=['Outlet_Establishment_Year', 'Item_Identifier', 'Outlet_Identifier', 'Item_Outlet_Sales', 'Item_Type'])
y = train_data['Item_Outlet_Sales']


# model Training


def train(model, x, y):
    model.fit(x, y)
    pred = model.predict(x)
    cv_score = cross_val_score(model, x, y, scoring='neg_mean_squared_error', cv=5)
    cv_score = np.abs(np.mean(cv_score))
    print('Model Report')
    print('MSE: ', mean_squared_error(y, pred))
    print('CV Score: ', cv_score)


# Finding Accuracy

# LinearRegression
model = LinearRegression(normalize=True)
train(model, x, y)
coef = pd.Series(model.coef_, x.columns).sort_values()
coef.plot(kind='bar', title='Model Coefficients')
# plt.show()



# Ridge
model = Ridge(normalize=True)
train(model, x, y)
coef = pd.Series(model.coef_, x.columns).sort_values()
coef.plot(kind='bar', title='Model Coefficients')
# plt.show()

# Lasso

model = Lasso(normalize=True)
train(model, x, y)
coef = pd.Series(model.coef_, x.columns).sort_values()
coef.plot(kind='bar', title='Model Coefficients')
# plt.show()


# Decision tree
model = DecisionTreeRegressor()
train(model, x, y)
coef = pd.Series(model.feature_importances_, x.columns).sort_values(ascending=False)
coef.plot(kind='bar', title='Feature Importance')
# plt.show()

# Random Forest Regressor
model = RandomForestRegressor()
train(model, x, y)
coef = pd.Series(model.feature_importances_, x.columns).sort_values(ascending=False)
coef.plot(kind='bar', title='Feature Importance')
# plt.show()

# Extra Tree Regressor
model = ExtraTreesRegressor()
train(model, x, y)
coef = pd.Series(model.feature_importances_, x.columns).sort_values(ascending=False)
coef.plot(kind='bar', title='Feature Importance')
plt.show()



# Linear regression has better accuracy than other Algorithm
