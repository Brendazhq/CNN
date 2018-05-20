#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/5/19 16:19
# @Author  : WeiXiang
# @Site    :
# @File    : V2v.py
# @Software: PyCharm
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
import random
packet_num=5
packet_size = np.maximum(5.00,np.random.normal(10,1,packet_num))
# packet_size = np.array([  6.76064773 , 8.02054822  ,7.49886275 ,  6.70533496 , 15.0437818])
# request_need_time = np.array([10,7,14,8,9])
request_need_time = np.random.uniform(8.0,14.0,packet_num)
print(packet_size)
# vehicle_speed_list=[70,80,90,100,110,120,130,140,150]
vehicle_speed_list = np.linspace(70,150,9)
# vehicle_speed = 90 # km/h
rsu_interval = 300 # 单位m
vehicle_loc = np.random.uniform(-rsu_interval/2,rsu_interval/2,packet_num)
print(vehicle_loc)

upload_speed = 50 # M/s

def calv2v(hops_j,packet_size,dis):
    dis=dis/1000
    K = 20
    V = 5
    vtran_speed = 30  # M/s
    density=0.3
    if hops_j==0:
        consumption = packet_size * K + dis * packet_size * V/density
        delay = packet_size/upload_speed+dis*packet_size/(vtran_speed)*density
        return consumption,delay
    hops = dis/density
    consumption = packet_size*K+hops*packet_size*V
    delay = packet_size/upload_speed+hops*packet_size/vtran_speed
    # if random.random()>0.9:
    #     consumption +=K*packet_size
    #     delay+=packet_size/upload_speed
    return consumption,delay

def calv2i(hops_j,packet_size,dis):
    dis=dis/1000
    K=20
    V=5
    consumption = (dis)*packet_size*K+hops_j*K*packet_size
    delay = packet_size/(upload_speed/math.exp(dis)) + hops_j*packet_size/upload_speed
    return consumption,delay



density=0.5
D = 555
sum_com=0
sum_i_com=0
sum_i_delay = 0
sum_delay = 0
sum_com_list=[]
sum_i_com_list=[]
sum_i_delay_list =[]
sum_delay_list=[]

for vehicle_speed in vehicle_speed_list:
    drive_dis = []
    for x in request_need_time:
        drive_dis.append(x*vehicle_speed*0.277778)
    drive_dis=np.array(drive_dis)
    vehicle_loc_tmp = vehicle_loc.copy()
    vehicle_loc_tmp+=drive_dis
    print(vehicle_loc_tmp)
    vehicle_loc_tmp=vehicle_loc_tmp//(rsu_interval/2)
    vehicle_loc_tmp=np.maximum(0,vehicle_loc_tmp)
    print(vehicle_loc_tmp)
    sum_com = 0
    sum_i_com = 0
    sum_i_delay = 0
    sum_delay = 0
    for i in range(len(packet_size)):
        psize = packet_size[i]
        # print(psize)
        iconsumption,idelay = calv2i(vehicle_loc_tmp[i],psize,dis=abs(vehicle_loc[i]))
        vconsumption,vdelay = calv2v(vehicle_loc_tmp[i]*1/density,psize,dis=abs(rsu_interval*vehicle_loc_tmp[i]-vehicle_loc[i]))
        # print(iconsumption,idelay)
        # print(vconsumption,vdelay)
        # print("\n")
        sum_i_com+=iconsumption
        sum_i_delay+=idelay
        result=[(iconsumption,idelay),(vconsumption,vdelay)]
        result=sorted(result,key=lambda x:(x[0],x[1]),reverse=False)
        if result[0][1]<D:
            sum_com+=result[0][0]
            sum_delay+=result[0][1]
        elif result[1][1]<D:
            sum_com+=result[1][0]
            sum_delay+=result[1][1]
        else:
            exit(1)
    sum_com_list.append(sum_com/len(packet_size))
    sum_i_com_list.append(sum_i_com/len(packet_size))
    sum_delay_list.append(sum_delay/len(packet_size))
    sum_i_delay_list.append(sum_i_delay/len(packet_size))

data = np.vstack((sum_com_list,sum_delay_list,sum_i_com_list,sum_i_delay_list))
data = np.transpose(data)


dataframe = pd.DataFrame(data,index=vehicle_speed_list,columns=['sum_com','sum_delay','sum_i_com','sum_i_delay'])

print(dataframe)
dataframe.to_csv('D:/毕设实习DATA/实习&&毕设/数据/vi2.csv')
dt = dataframe.loc[:,['sum_com','sum_i_com']]
# dt.cumsum()
dt.plot()
dt_delay= dataframe.loc[:,['sum_delay','sum_i_delay']]
dt_delay.plot()
plt.show()