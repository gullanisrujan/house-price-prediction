import pandas as pd #for reading csv files
from  sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

df = pd.read_csv("data.csv")
print(df.shape,df.head())
x=df.drop('price',axis=1)
y=df['price']

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=51)

## standardization
sc = StandardScaler()
sc.fit(x_train)

x_train=sc.transform(x_train)
x_test =sc.transform(x_test)

x_train_df=pd.DataFrame(x_train)
print(x_train_df.describe().round(2))

## model building
model = LinearRegression()
model.fit(x_train,y_train)
scr = model.score(x_test,y_test)

print(len(model.coef_),model.intercept_,scr)

print(model.predict(x_test))

