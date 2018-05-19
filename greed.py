#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/5/12 16:32
# @Author  : WeiXiang
# @Site    : 
# @File    : greed.py
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
# frequency_fog = np.array([1.5,2.7,2.4,2.5,2.13,1.98,1.74])
fog_serve  = frequency_fog/0.7
fog_max = fog_serve
fog_last_arrival =np.array([4.0]*7,dtype=np.float64)

frequency_cloud = np.array([4.33,5.25,4.98])
cloud_serve = frequency_cloud/0.7
cloud_machine_num = np.array([30,32,29])
cloud_max = np.multiply(cloud_serve,cloud_machine_num)
cloud_last_arrival = np.array([0.7]*3)
# requests_need = np.random.normal(0.7,0.2,size=7)
# requests_need = np.maximum(requests_need,0.15)
# requests_need_cycles = np.array([0.70419461, 0.60562875, 0.70182378, 0.38406417, 0.83790147, 0.59907687])
# requests_need_cycles = np.random.normal(0.9,0.2,6)
requests_need_cycles = np.array([0.60562875,0.70419461,0.70182378,0.38406417,0.83790147,0.59907687,0.88707632])
requests_need_cycles = np.maximum(requests_need_cycles,0.2)

D = 0.9
sum_com=0
sum_delay=0
server_seq=[]
for x in requests_need_cycles:
    mini_consumption = 0
    mindelay = float('inf')
    serverid = 0
    mini = float('inf')
    for serve_num in range(len(fog_serve)):
        delay = cal_fog_delay(fog_last_arrival[serve_num],x,frequency_fog[serve_num])
        consumption = x*cal_fog_consumption(frequency_fog[serve_num])
        # print("original consumption",consumption)
        # if delay>D:
        #     continue
        if consumption<mini and delay<D:
            mini=consumption
            serverid = serve_num
            mindelay = delay
    for serve_num in range(len(cloud_serve)):
        delay = cal_cloud_delay(cloud_last_arrival[serve_num],cloud_machine_num[serve_num],x,frequency_cloud[serve_num])
        consumption = x*cal_cloud_consumption(frequency_cloud[serve_num],cloud_machine_num[serve_num])
        # if delay > D:
        #     continue;
        if consumption<mini and delay < D:
            mini = consumption
            mindelay = delay
            serverid = serve_num+7
    sum_com+=mini
    sum_delay+=mindelay
    server_seq.append(serverid)
    if serverid>6:
        cloud_last_arrival[serverid%7]+=0.5
    else:
        fog_last_arrival[serverid]+=0.5
print("request_num",len(requests_need_cycles))
print("sum_com:",sum_com)
print("sum_delay:",sum_delay)
print("server_seq:",server_seq)
print("fog last arrival:",fog_last_arrival)
print("cloud_last_arrival:",cloud_last_arrival)
print("request_need_cycle:",requests_need_cycles)