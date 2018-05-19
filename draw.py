# -*- coding: utf-8 -*-
import sys
reload(sys)
sys.setdefaultencoding("utf-8")

import os
import re
import numpy as np
import sklearn
from sklearn import linear_model
from sklearn.svm import SVR
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt
import csv

# x = np.linspace(0, 10, 1000)
# y = np.sin(x)
# z = np.cos(x**2)

 
def fnmatch_filter_demo(path,pattern):  
    for path,dir,filelist in os.walk(path):  
        for name in fnmatch.filter(filelist,pattern):  
            yield os.path.join(path,name)  

# for name in fnmatch_filter_demo("D:\code\python\urllib\Recount",'*.csv'):
# 	print name

def dir_demo(dir_path,pattern):
	for path,dir,filelist in os.walk(dir_path):
		for dir_name in dir:
			m = re.search(pattern,dir_name)
			if m:
				yield os.path.join(dir_path,dir_name)




def refile_demo(path,pattern):  
    #pattern=fnmatch.translate(pattern)  
    for path,dir,filelist in os.walk(path):  
        for name in filelist:  
            m=re.search(pattern,name)  
            if m:  
                yield os.path.join(path,name)  

def calcDistance(Lat_A, Lng_A, Lat_B, Lng_B):
    try:
        ra = 6378.140  # 赤道半径 (km)
        rb = 6356.755  # 极半径 (km)
        flatten = (ra - rb) / ra  # 地球扁率
        rad_lat_A = radians(Lat_A)
        rad_lng_A = radians(Lng_A)
        rad_lat_B = radians(Lat_B)
        rad_lng_B = radians(Lng_B)
        pA = atan(rb / ra * tan(rad_lat_A))
        pB = atan(rb / ra * tan(rad_lat_B))
        xx = acos(sin(pA) * sin(pB) + cos(pA) * cos(pB) * cos(rad_lng_A - rad_lng_B))
        c1 = (sin(xx) - xx) * (sin(pA) + sin(pB)) ** 2 / cos(xx / 2) ** 2
        c2 = (sin(xx) + xx) * (sin(pA) - sin(pB)) ** 2 / sin(xx / 2) ** 2
        dr = flatten / 8 * (c1 - c2)
        distance = ra * (xx + dr)
        #distance=round(distance,4)
        return distance
    except BaseException:
    	if(Lat_A!=Lat_B or Lng_A!=Lng_B):
        	print Lat_A, Lng_A, Lat_B, Lng_B
        return 0.0


def generateRequiredFileLst(dir_path,dir_pattern,filepattern):
	dir_list=list(dir_demo(dir_path,dir_pattern))
	name_list=[]
	for dir_path in dir_list:
		name=[]
		print "DIR:",dir_path
		name=list(refile_demo(dir_path,filepattern))
		name_list.extend(name)
	return name_list



save_num=0



def paint(gap,rowIndex,distype,name):
	x = np.arange(gap/2,20000,gap,dtype=np.int64)

	y = range(20000/gap)
	for i in range(0,20000/gap):
		y[i] = 0

	#print y

	with open(name,"rb") as ft:
		reader = csv.reader(ft)
		for row in reader:
			if row[rowIndex] == -1:
				continue
			#print float(row[1])
			y[int(float(row[rowIndex])/gap)] += 1
			# print "y[",int(float(row[-2])/gap),"]:",y[int(float(row[-2])/gap)]
	# xx_test = np.linspace(0, 25000, 1000).reshape(1000,1)
	xx=x.reshape(20000/gap,1)
	ynp = np.array(y,dtype=np.int64)
	yy=ynp.reshape(ynp.shape[0],1)
	quadratic_featurizer = PolynomialFeatures(degree=3)   # if we set parameter "interaction_only=True" that means the feature set is like[1,a,b,ab] withot a**2 or b**2
	X_train_quadratic = quadratic_featurizer.fit_transform(xx)  # 形式如[a,b] ,则二阶多项式的特征集如下[1,a,b,a^2,ab,b^2]。
	# XX_test_quadratic = quadratic_featurizer.fit_transform(xx_test)
	regressor_quadratic = linear_model.LinearRegression()
	regressor_quadratic.fit(X_train_quadratic, yy)
	print regressor_quadratic.coef_
	print regressor_quadratic.score(X_train_quadratic,yy)
	global save_num
	np.save("x"+str(save_num),xx)
	np.save("y"+str(save_num),yy)
	save_num+=1


	
        		
	#regr = linear_model.LinearRegression()
	#regr.fit(xx,y)
	

	# plt.figure(figsize=(8,4))
	# plt.plot(x,y,label="$sin(x)$",color="red",linewidth=2)
	# plt.plot(x,z,"b--",label="$cos(x^2)$")
	# plt.xlabel("Time(s)")
	# plt.ylabel("Volt")
	# plt.title("PyPlot First Example")
	# plt.ylim(0,500)
	# plt.legend()
	# plt.show()

	fig = plt.figure()
	axes = fig.add_subplot(111)
	axes.scatter(x,y,label=gap,lw=0.5)
	# plt.plot(xx, y_rbf, color='red',
 #         linewidth=3,label="RBF")
	plt.plot(xx, regressor_quadratic.predict(X_train_quadratic),'r',color='red',
         linewidth=3,label="poly")
	plt.xlabel(distype+' distance(km)')
	plt.xlim(0,25000)
	plt.ylabel('citation number')
	#plt.ylim(0,50)
	Journalname=name.split("\\")[-1].split("disFile")[0]
	print Journalname.strip()+".png"
	plt.title(Journalname)
	#plt.plot(x,y,label='function',color="red",lw=1)
	plt.legend(loc='upper right')
	plt.show()












if __name__=="__main__":
    requiredFilelst = generateRequiredFileLst("E:\DC\urllib","disInfo","disFile.csv")
    for x in requiredFilelst:
    	paint(500,-1,"max",x)
