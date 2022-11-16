# -*- coding: utf-8 -*-
"""
Created on Tue Nov 15 13:37:54 2022

@author: USER
"""


"""




x = 10 
y= 20 

print(df.head(x)) # Prints first x rows

print(df.tail(y)) # Prints last y rows 

print(df.shape) # Numpy array nesnesinin kaç satır ve sütundan oluştuğunu gösteren bir tupple nesnesi döndürür

print(df.Durum.value_counts()) # Durum değeri yani 0 1 olan değeri bize verir.

print(df.info()) # Bize veri seti hakkında bilgi verir.

plt.figure(figsize = (5, 5))
ax = sns.countplot(df['Durum'], palette = 'rocket')
ax.set_xticklabels(ax.get_xticklabels(), rotation = 0, ha = "right")
plt.suptitle("Kanser Durum Sonuçları")
plt.show()



columnName="Durum"
print(df[columnName].value_counts()*100/len(df)) # Veri setinde durum 0 ve 1 yerine dead ve alivedir bunun oranını bize verir.

print(df.nunique()) # Veri setinde bulunan değerlerin farklılık adedini gösterir.

print(df.Durum.unique()) # 0/1 değerlerinin nasıl olduğunu gösterir.

print(df.describe().T) # Sayısal verilere sahip olan stünların max min std gibi istatiksel değerleri döndürür.



k = 9 #number of variables for heatmap
cols = df.corr().nlargest(k, 'Yaş')['Yaş'].index
cm = df[cols].corr()
plt.figure(figsize=(10,6))
sns.heatmap(cm, annot=True, cmap = 'viridis')

f,ax = plt.subplots(figsize=(10, 10))
sns.heatmap(df.corr(), annot=True, linewidths=0.5,linecolor="purple", fmt= '.1f',ax=ax)
plt.show()


df["Yaş"].hist(edgecolor = "red");


top_age = df.Yaş.value_counts().head(15)
plt.figure(figsize=(18,9))
plt.xticks(rotation=75)
plt.title('Diyagonal kişilerde kanserin en büyük yaşları')
sns.barplot(x=top_age.index, y=top_age)



df.plot(kind = 'density', subplots= True, layout = (3,3), sharex =False, figsize = (18, 18))



fig=plt.figure(figsize=(25,7))
for i,col in enumerate(['İncelenen Bölgesel Düğüm']):
 ax=fig.add_subplot(1,2,i+1)
 sns.countplot(df[col])
 
 
 df1 = df.loc[:,["Yaş","Seviye","İncelenen Bölgesel Düğüm"]]
df1.plot()


plt.figure(figsize=(18,9))
sns.swarmplot(x="Durum", y="Yaş",hue="Seviye", data=df)
plt.show()


sns.lmplot(x="Reginol Düğümü Pozitif", y="İncelenen Bölgesel Düğüm", data=df)
plt.show()



plt.figure(figsize=(15,6))
sns.countplot(x='Yaş',data = df, hue = 'Durum',palette='rocket')
plt.show()



sns.catplot(x="Yaş", y="Seviye", hue="Durum", kind="swarm", data= df );


plt.figure(figsize=(14,10))
sns.boxplot(x="Durum", y="İncelenen Bölgesel Düğüm", hue="Durum", data=df, palette="PRGn" )
plt.show()





/##





"""

"""
print(df["Yaş"]) # 
print(df.iloc[9]) # 9. indexin verilerini yazdır
print(df.iloc[:,2]) # ilk 2 sütunu yazdır
print(df.iloc[1:2,2]) # 2. indexin 1'den 2'ye kadar olan sütunları yazdır
columnName="Yaş"
df=df.drop(columnName,axis=1) # columName sütununu siler
print(df)
# line,bar,scatter
df.plot(kind="scatter",x="Tümor Boyutu",y="Hayatta Kalma Ayları")
plt.ylabel("Hayatta Kalma Ayları")

plt.show()

print(df.shape)






X=df.drop("Tümor Boyutu",axis=1)
y=df["Tümor Boyutu"]

donusum = LabelEncoder()

y=donusum.fit_transform(y)


y=np_utils.to_categorical(y)

X_train,X_test,Y_train,Y_test=train_test_split(X,y,test_size=0.33,random_state=123)






islem =MinMaxScaler(feature_range=(0,1)) # Min-Max Normalization

z_scoreNormalization()


X_train=islem.fit_transform(X_train)
X_test=islem.fit_transform(X_test)

model=Sequential()
model.add(Dense(800),input_dim=10,activation="relu")
model.add(Dropout(0.2))
model.add(Dense(800),activation="relu")
model.add(Dropout(0.2))
model.add(Dense(800),activation="relu")
model.add(Dropout(0.2))
model.add(Dense(3,activation="softmax"))

model.summary()
""""
""""""""""""








