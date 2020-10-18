import numpy as np
import pandas as pd
from sklearn import linear_model

# pull the training data
data = pd.read_csv("OFF.csv")

# pull the test data
wl = []
efg = []
tov = []
orb = []
ftp = []

f = open("OFF-test.csv", "r")
f.readline()
for line in f:
    lineAr = line.split(",")
    wl.append(float(lineAr[0]))
    efg.append(float(lineAr[1]))
    tov.append(float(lineAr[2]))
    orb.append(float(lineAr[3]))
    ftp.append(float(lineAr[4]))
f.close()

# get the correlation coefficents
efg_r = np.corrcoef(wl, efg)
tov_r = np.corrcoef(wl, tov)
orb_r = np.corrcoef(wl, orb)
ftp_r = np.corrcoef(wl, ftp)

print(efg_r[0,1])
print(tov_r[0,1])
print(orb_r[0,1])
print(ftp_r[0,1])


#FIRST MODEL - y = wl & x = efg, tov, orb, ftp

# get the independent (4 factors) and dependent (win/loss %) variables
y1 = data.drop(columns=["efg","tov","orb","ftp"])
x1 = data.drop(columns="wl")

# make and fit the linear model
reg1 = linear_model.LinearRegression()
reg1.fit(x1,y1)

#run the test data through model 1 and calculate the MSE
predictions1 = []
diff1 = 0
for w in range(0, len(wl)):
    p = reg1.predict([[efg[w], tov[w], orb[w], ftp[w]]])
    predictions1.append(p)
    er1 = (wl[w] - p) ** 2
    diff1 += er1

mse1 = er1 / len(wl)
print("MSE 1:", mse1)
print("Thetas 1:", reg1.coef_)


#SECOND MODEL - y = wl & x = efg & tov
# get the independent (4 factors) and dependent (win/loss %) variables
y2 = data.drop(columns=["efg","tov","orb","ftp"])
x2 = data.drop(columns=["wl", "orb", "ftp"] )

# make and fit the linear model
reg2 = linear_model.LinearRegression()
reg2.fit(x2,y2)

#run the test data through model 1 and calculate the MSE
predictions2 = []
diff2 = 0
for w in range(0, len(wl)):
    p = reg2.predict([[ efg[w], tov[w] ]])
    predictions2.append(p)
    er2 = (wl[w] - p) ** 2
    diff2 += er2

mse2 = er2 / len(wl)
print("MSE 2:", mse2)
print("Thetas 2:", reg2.coef_)

#THIRD MODEL - y = wl & x = efg

# get the independent (4 factors) and dependent (win/loss %) variables
y3 = data.drop(columns=["efg","tov","orb","ftp"])
x3 = data.drop(columns=["wl","tov","orb","ftp"])

# make and fit the linear model
reg3 = linear_model.LinearRegression()
reg3.fit(x3,y3)

#run the test data through model 1 and calculate the MSE
predictions3 = []
diff3 = 0
for w in range(0, len(wl)):
    p = reg3.predict([[ efg[w] ]])
    predictions3.append(p)
    er3 = (wl[w] - p) ** 2
    diff3 += er3

mse3 = er3 / len(wl)
print("MSE 3:", mse3)
print("Thetas 3:", reg3.coef_)

# print the average win % of all 3 models
avg = []
for v in range(0, len(predictions1)):
     avg.append( (predictions1[v][0] + predictions2[v][0] + predictions3[v][0]) / 3 )

print(avg)





