#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/5/3 15:23
# @Author  : WeiXiang
# @Site    : 
# @File    : cycle.py
# @Software: PyCharm
import numpy as np
from scipy.special import comb

def cal_cloud_delay(y,n,k,f):
    if y>n*f/k:
        return float('inf')
    else:
        return comb(n,y*k/f)/(n*f/k-y) + k/f

def cal_fog_delay(x,k,f):
    if x>f/k:
        return float('inf')
    else:
        return 1/(f/k - x)

def cal_cloud_consumption(f,n):
    return n*(1.2*f**2+1.4*f)
def cal_fog_consumption(f):
    return (1.2*f**2+1.4*f)
servers = 10
requests = 7
frequency_fog=np.array([3.13,2.95,3.06,2.87,3.14,3.22,3.65])
fog_serve  = frequency_fog/0.7
fog_max = fog_serve
fog_last_arrival =np.array([3.5]*7)

frequency_cloud = np.array([4.33,5.25,3.98])
cloud_serve = frequency_cloud/0.7
# cloud_machine_num = np.array([7,4,5])
cloud_machine_num = np.array([30,32,29])
cloud_max = np.multiply(cloud_serve,cloud_machine_num)
cloud_last_arrival = np.array([0.7]*3)
# requests_need = np.random.normal(0.7,0.2,size=7)
# requests_need = np.maximum(requests_need,0.15)
requests_need = np.array([0.60562875,0.70419461,0.70182378,0.38406417,0.83790147,0.59907687,0.88707632])
D = 0.9*7
# for x in requests_need:
#     servers=[]
#     mini_consumption = 0
#     mindelay = 0
#     serverid = 0
#     for serve_num in range(len(fog_serve)):
#         delay = cal_fog_delay(fog_last_arrival[serve_num],x,frequency_fog[serve_num])
#         consumption = delay*cal_fog_consumption(frequency_fog[serve_num])
#         if delay>D:
#             continue
#         if consumption<mini:
#             mini=consumption
#             serverid = serve_num
#             mindelay = delay
#     for serve_num in range(len(cloud_serve)):
#         delay = cal_cloud_delay(cloud_last_arrival[serve_num],cloud_machine_num[serve_num],x,frequency_cloud[serve_num])
#         consumption = delay*cal_cloud_consumption(frequency_cloud[serve_num],cloud_machine_num[serve_num])
#         if delay > D:
#             continue;
#         if consumption<mini:
#             mini = consumption
#             mindelay = mindelay
#             serverid = serve_num+5
#     if serverid>4:
#         cloud_last_arrival[serverid%5]+=1
#     else:
#         fog_last_arrival[serverid%5]+=1

def allocate(x, server_id, arrivate):  # x 代表所需转数，server_id 服务器编号
    if server_id>6:
        delay = cal_cloud_delay(arrivate, cloud_machine_num[server_id % 7], x, frequency_cloud[server_id % 7])
        consum = cal_cloud_consumption(frequency_cloud[server_id % 7], cloud_machine_num[server_id % 7]) * x
    else:
        delay = cal_fog_delay(arrivate, x, frequency_fog[server_id])
        consum = cal_fog_consumption(frequency_fog[server_id]) * x
    return consum,delay



min=float('inf')
for i in range(10):
    for j in range(10):
        for k in range(10):
            for l in range(10):
                for m in range(10):
                    for n in range(10):
                        #for p in range(10):
                            arrival_rate = np.append(fog_last_arrival, cloud_last_arrival)
                            arrival_rate[i] += 0.1
                            arrival_rate[j] += 0.1
                            arrival_rate[k] += 0.1
                            arrival_rate[l] += 0.1
                            arrival_rate[m] += 0.1
                            arrival_rate[n] += 0.1
                           # arrival_rate[p] += 0.5
                            consum1,delay1=allocate(requests_need[0],i,arrival_rate[i])
                            consum2,delay2=allocate(requests_need[1],j,arrival_rate[j])
                            consum3,delay3=allocate(requests_need[2],k,arrival_rate[k])
                            consum4,delay4=allocate(requests_need[3],l,arrival_rate[l])
                            consum5,delay5=allocate(requests_need[4],m,arrival_rate[m])
                            consum6,delay6=allocate(requests_need[5],n,arrival_rate[n])
                           # consum7,delay7=allocate(requests_need[6],p,arrival_rate[p])
                            consum=consum1+consum2+consum3+consum4+consum5+consum6#+consum7
                            delay = delay1+delay2+delay3+delay4+delay5+delay6#+delay7
                            if delay<D and consum<min:
                                min=consum
                                min_delay = delay
                                allocate_seq =[i,j,k,l,m,n]#,p]
                                print(allocate_seq)






print(min)
print(min_delay)
print(allocate_seq)





