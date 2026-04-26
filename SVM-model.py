import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
data = {  
    "texts_per_date":[1,3,0,4,2,1,3,2,4,5,1,2,3,4],
    "time_you_know_them_month":[0,2,3,1,0,7,1,0,2,1,3,0,2,1],
    "emoji":[1,3,4,2,1,0,3,2,1,0,2,1,3,4],
    "hangouts_time":[0,1,2,1,2,0,4,3,2,1,4,1,2,0,5],
    "average_time_response_days":[0,4,2,1,0,7,2,4,0,1,1,0,6,1,3],
    "conversations":[1,4,6,2,1,1,5,6,1,2,4,5,6,7,8],
    "laugh_at_your_jokes":[2,0,1,3,4,6,5,2,1,4,5],
    "they_like_you":[0,1,0,0,1,0,0,1,0,0,1,0,1,1,0]
}
min_len = min(len(col) for col in data.values())

data_trimmed = {
    key: values[:min_len]
    for key, values in data.items()
}
df = pd.DataFrame(data_trimmed)
X = df.drop("they_like_you",axis=1)
Y = df["they_like_you"]
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2,random_state=31)

#fititng
Model = LogisticRegression()
Model.fit(X_train,Y_train)
#test accuracy
y_predict = Model.predict(X_test)
print(accuracy_score(y_predict,Y_test))

