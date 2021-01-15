# -*- coding: utf-8 -*-
"""
Created on Thu Sep 10 21:09:14 2020

@author: tang
"""


import math
import gym
from gym import spaces, logger
from gym.utils import seeding
import numpy as np
from scipy.stats import truncnorm
import csv
import matplotlib.pyplot as plt
from itertools import cycle

def measrement_real(self,t):
    step= t/self.dt
    measurement = self.data[int(step)]
    measurement.dtype = 'float64'
    q_t = measurement[0:4]
    omega_obs = measurement[4:7]
    omega_obs = np.vstack(omega_obs)
    acc_m = measurement[7:10]
    acc_m = np.vstack(acc_m)
    mag_m = measurement[10:13]
    mag_m = np.vstack(mag_m)
    return q_t,omega_obs,acc_m,mag_m


# def omega_t_sim_ekfFail(t):
def omega_t_sim(t):
    if (30 < t <= 40):
    # if (31 < t <= 46):
        omega_1 = 51 # the real angular velocity, in q_pred direction
        omega_2 = 0  # the real angular velocity, in y direction
        omega_3 = 0  # the real angular velocity, in z direction
    else:
        omega_1 = 0
        omega_2 = 0
        omega_3 = 0
    omega = np.array([[omega_1], [omega_2], [omega_3]])
    return omega

def omega_t_sim_easy2(t):
# def omega_t_sim(t):
    t=t*50*2
    # input = 0*np.cos(t) * self.dt
    # 额外给一个准确的角速度（带偏移+噪声），然后由此仿真出来acc，gyro，mag的测量值
    # 1. simulate the true angular velocity
    if (0.5 * math.pi / 0.008) < t <= ( 1.5* math.pi / 0.008):
        omega_1 = np.sin((0.008 * t) - 0.5* math.pi)  # the real angular velocity, in q_pred direction
        omega_2 = 0  # the real angular velocity, in y direction
        omega_3 = 0  # the real angular velocity, in z direction
    else:
        omega_1 = 0
        omega_2 = 0
        omega_3 = 0
    omega = np.array([[omega_1], [omega_2], [omega_3]])*0.4
    return omega

def omega_t_sim_medium(t):
# def omega_t_sim(t):
    t=(t- 20)*50*2/7 
    if (1 * math.pi / 0.09) < t <= (9 * math.pi / 0.09):
        omega_1 = np.sin(t * 0.09)
        omega_2 = 0
        omega_3 = 0
    # elif (2 * math.pi / 0.05) <= t <= (3 * math.pi / 0.05):
    #     omega_1 = 0
    #     omega_2 = np.sin(t * 0.05)*1.5
    #     omega_3 = 0
    # elif (3 * math.pi / 0.05) < t < (5 * math.pi / 0.05):
    #     omega_1 = 0
    #     omega_2 = 0
    #     omega_3 = -np.sin(t * 0.05)*2
    else:
        omega_1 = 0
        omega_2 = 0
        omega_3 = 0
    omega = np.array([[omega_1], [omega_2], [omega_3]])*0.7
    return omega

def omega_t_sim_slow(t):
# def omega_t_sim(t):
    t=(t)*50*2
    # input = 0*np.cos(t) * self.dt
    # 额外给一个准确的角速度（带偏移+噪声），然后由此仿真出来acc，gyro，mag的测量值
    # 1. simulate the true angular velocity
    a=1600
    b=1950
    c=5000
    d=6000
    if a<= t < (a+b)/2:
        omega_1 = 2*pow((t-a)/(b-a),2) # the real angular velocity, in q_pred direction
        omega_2 = 0  # the real angular velocity, in y direction
        omega_3 = 0  # the real angular velocity, in z direction
    elif (a+b)/2 <= t < b:
        omega_1 = 1 - 2*pow((t-b)/(b-a),2)
        omega_2 = 0
        omega_3 = 0
    elif b <=t < c:
        omega_1 = 1
        omega_2 = 0
        omega_3 = 0
    elif c<= t <(c+d)/2:
        omega_1 = 1-2*pow((t-c)/(d-c),2)
        omega_2 = 0
        omega_3 = 0
    elif (c+d)/2 <= t < d:
        omega_1 = 2*pow((t-d)/(d-c),2)
        omega_2 = 0
        omega_3 = 0
    else :
        omega_1 = 0
        omega_2 = 0
        omega_3 = 0
    omega = np.array([[omega_1], [omega_2], [omega_3]])*0.2
    return omega

def omega_t_sim_crazy(t):
# def omega_t_sim(t):
    # t=t*50*2
    # input = 0*np.cos(t) * self.dt
    # 额外给一个准确的角速度（带偏移+噪声），然后由此仿真出来acc，gyro，mag的测量值
    # 1. simulate the true angular velocity
    T = 0.7
    # omega_1 = 50  # the real angular velocity, in q_pred direction
    # omega_2 = 0  # the real angular velocity, in y direction
    # omega_3 = 0  # the real angular velocity, in z direction
    if (30 < t <= 40):
        omega_1 = 20 # the real angular velocity, in q_pred direction
        omega_2 = 0  # the real angular velocity, in y direction
        omega_3 = 0  # the real angular velocity, in z direction
    # elif(39.3 < t <= 45):
    #     omega_1 = 0.5 # the real angular velocity, in q_pred direction
    #     omega_2 = 0  # the real angular velocity, in y direction
    #     omega_3 = 0  # the real angular velocity, in z direction
    else:
        omega_1 = 0
        omega_2 = 0
        omega_3 = 0
    omega = np.array([[omega_1], [omega_2], [omega_3]])
    return omega


def omega_t_sim_trainingProfile(t):
# def omega_t_sim(t):
    # t=t*50*2
    # input = 0*np.cos(t) * self.dt
    # 额外给一个准确的角速度（带偏移+噪声），然后由此仿真出来acc，gyro，mag的测量值
    # 1. simulate the true angular velocity
    a=1
    b=3.5
    c=5
    d=7
    if a<= t < (a+b)/2:
        omega_1 = 2*pow((t-a)/(b-a),2) # the real angular velocity, in q_pred direction
        omega_2 = 0  # the real angular velocity, in y direction
        omega_3 = 0  # the real angular velocity, in z direction
    elif (a+b)/2 <= t < b:
        omega_1 = 1 - 2*pow((t-b)/(b-a),2)
        omega_2 = 0
        omega_3 = 0
    elif b <=t < c:
        omega_1 = 1
        omega_2 = 0
        omega_3 = 0
    elif c<= t <(c+d)/2:
        omega_1 = 1-2*pow((t-c)/(d-c),2)
        omega_2 = 0
        omega_3 = 0
    elif (c+d)/2 <= t < d:
        omega_1 = 2*pow((t-d)/(d-c),2)
        omega_2 = 0
        omega_3 = 0
    else :
        omega_1 = 0
        omega_2 = 0
        omega_3 = 0
    omega_ = np.array([[omega_1], [omega_2], [omega_3]])*10
    R = [[-0.1469, 0.3804, -0.9131], [-0.0470, 0.9194, 0.3906], [0.9880, 0.1003, -0.1172]]
    R = np.mod(R, 2)  # normalize rotation matrix
    omega = np.dot(R , omega_)
    return omega

def Log(q):
    qw = q[0]
    qv = q[1:4]
    if(np.linalg.norm(qv)!=0):
        theta = np.arctan2(np.linalg.norm(qv),qw);#a=atan2(y/x), y>0, so 0<a<pi, so 0<theta<pi
        u = qv/np.linalg.norm(qv);
        output = u*theta;
    else:
        output = np.array([0,0,0]);
    return output

def exp(omega):
    theta = np.linalg.norm(omega)
    w_n = np.linalg.norm(omega)
    if (w_n != 0):
        vector = omega / np.linalg.norm(omega)
    else:
        vector = np.array([[0], [0], [0]])
    xyz = vector * np.sin(theta)
    exp_w = np.array([np.cos(theta), xyz[0][0], xyz[1][0], xyz[2][0]])
    return exp_w


def quatRightMulMat(q):
    matrix = np.identity(4) * q[0]
    matrix[0, 0] = q[0]
    matrix[1, 0] = q[1]
    matrix[2, 0] = q[2]
    matrix[3, 0] = q[3]
    matrix[0, 1] = -q[1]
    matrix[0, 2] = -q[2]
    matrix[0, 3] = -q[3]
    matrix[2, 1] = -q[3]
    matrix[1, 2] = q[3]
    matrix[3, 1] = q[2]
    matrix[1, 3] = -q[2]
    matrix[3, 2] = -q[1]
    matrix[2, 3] = q[1]
    return matrix


def quatLeftMulMat(q):
    matrix = np.identity(4) * q[0]
    matrix[0, 0] = q[0]
    matrix[1, 0] = q[1]
    matrix[2, 0] = q[2]
    matrix[3, 0] = q[3]
    matrix[0, 1] = -q[1]
    matrix[0, 2] = -q[2]
    matrix[0, 3] = -q[3]
    matrix[2, 1] = q[3]
    matrix[1, 2] = -q[3]
    matrix[3, 1] = -q[2]
    matrix[1, 3] = q[2]
    matrix[3, 2] = q[1]
    matrix[2, 3] = -q[1]
    return matrix


def quatConj(q):
    p = np.array([[q[0]], [-q[1]], [-q[2]], [-q[3]]])
    return p


def quatPure2Q(v3):
    q = np.array([[0], [v3[0]], [v3[1]], [v3[2]]])
    return q


def vecCross(v3):
    matrix = np.array([[0, -v3[2], v3[1]], [v3[2], 0, -v3[0]], [-v3[1], v3[0], 0]])
    return matrix


def quat2eul(q):
    qw = q[0];
    qx = q[1];
    qy = q[2];
    qz = q[3];
    aSinInput = -2 * (qx * qz - qw * qy)
    if aSinInput > 1:
        aSinInput = 1
    if aSinInput < -1:
        aSinInput = -1
    eul_1 = np.arctan2(2 * (qx * qy + qw * qz), qw * qw + qx * qx - qy * qy - qz * qz)/np.pi*180.0
    eul_2 = np.arcsin(aSinInput)/np.pi*180.0
    eul_3 = np.arctan2(2 * (qy * qz + qw * qx), qw * qw - qx * qx - qy * qy + qz * qz)/np.pi*180.0
    return np.array([eul_1, eul_2, eul_3])

class Ex3_EKF_gyro(gym.Env):

    def __init__(self):

        self.clean =  False
        self.choice = 'otherCase'
        self.realMeasurement = False
        self.returnQ = True

        self.t = 0
        self.dt = 0.01

        self.noise_gyro_bias = np.array(
            [[0.0], [0.0], [0.0]])  # a small changing bias in angular velocity, the initial value is 0.0001
        self.cov_noise_gyro_bias = np.array([[0.00000, 0, 0], [0, 0.00000, 0], [0, 0, 0.00000]])
        # the true value of self.noise_gyro_bias & self.cov_noise_gyro_bias is unknown,
        # and the measurement of acc&mag is used to compensate the uncertain drift caused by noise_gyro_bias

        if not self.clean:
            self.cov_noise_i = np.array([[.0003, 0, 0], [0, .0003, 0], [0, 0, .0003]])*1
            self.cov_a = np.array([[0.0005, 0, 0], [0, 0.0005, 0], [0, 0, 0.0005]])*1
            self.cov_mag = np.array([[0.0003, 0, 0], [0, 0.0003, 0], [0, 0, 0.0003]])*1
        else:
            self.cov_noise_i = np.array([[.00000, 0, 0], [0, .0000, 0], [0, 0, .0000]])
            self.cov_a = np.array([[0.000, 0, 0], [0, 0.000, 0], [0, 0, 0.000]])
            self.cov_mag = np.array([[0.000, 0, 0], [0, 0.000, 0], [0, 0, 0.000]])


        # the true initial pose (from the sensor frame to the world frame)
        self.q_t = np.random.uniform([-1,-1,-1,-1], [1,1,1,1])
        self.q_t = self.q_t /np.linalg.norm(self.q_t)
        self.P = np.eye(4)*0.1

        # displacement limit set to be [-high, high]
        high = np.array([10000, 10000, 10000])

        self.action_space = spaces.Box(low=np.array(
            [-10., -10., -10., -10., -10., -10., -10., -10., -10., -10., -10., -10., -10., -10., -10., -10., -10., -10.])*0.002,
                                       high=np.array(
                                           [10., 10., 10., 10., 10., 10., 10., 10., 10., 10., 10., 10., 10., 10., 10.,
                                            10., 10., 10.])*0.002,
                                       dtype=np.float32)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)

        self.seed()
        self.viewer = None
        self.state = None
        self.output = None

        self.steps_beyond_done = None

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):  # here u1,u2=measurement, which is a result of the action
        train = False
       

        u_11, u_21, u_31, u_41, u_51, u_61, \
        u_12, u_22, u_32, u_42, u_52, u_62, \
        u_13, u_23, u_33, u_43, u_53, u_63 = action

        t = self.t
        
        if self.realMeasurement:
            q_t_lastStep = self.q_t
            self.q_t,omega_obs,acc_m,mag_m=measrement_real(self,t)
            q_t = self.q_t
            omega = 2 / self.dt * Log(np.inner(quatLeftMulMat(quatConj(q_t_lastStep)),np.hstack((q_t))))
        else:        
            omega = omega_t_sim(t) #simulate the trajectory
            # 1. update the true pose
            q_t = self.q_t
            q_t_lastStep = self.q_t
            q_t = np.dot(quatLeftMulMat(q_t), exp(0.5 * self.dt * omega).T)
            q_t = q_t / np.linalg.norm(q_t)
            self.q_t  = q_t
    
            # 2. simulate the sensor measurements
            # wm=omega+noise_gyro_bias+noise_i; d(noise_gyro_bias)/dt=noise_gyro_bias_var
            # noise_i~N(0,cov_w); noise_gyro_bias_var~N(0,cov_noise_gyro_bias)
            noise_gyro_bias_var = np.random.multivariate_normal([0, 0, 0], self.cov_noise_gyro_bias).flatten()
            noise_gyro_bias_t = self.noise_gyro_bias + np.array(
                [[noise_gyro_bias_var[0]], [noise_gyro_bias_var[1]], [noise_gyro_bias_var[2]]])
    
    
            if not self.clean:
                noise_i = np.random.multivariate_normal([0, 0, 0], self.cov_noise_i).flatten()
            else:
                noise_i = np.random.multivariate_normal([0, 0, 0], self.cov_noise_i).flatten()*0
    
            omega_obs = omega + noise_gyro_bias_t + np.array([[noise_i[0]], [noise_i[1]], [noise_i[2]]])
            # omega_obs = omega
    
            # assume the pure acc goes up in proportion to omega
            # change gravity direction
            # acc_m_q = np.dot(np.dot(quatLeftMulMat(quatConj(self.q_t)), quatRightMulMat(self.q_t)), quatPure2Q([0, 0, -1]))
            acc_m_q = np.dot(np.dot(quatLeftMulMat(quatConj(self.q_t)), quatRightMulMat(self.q_t)), quatPure2Q([0, 0, 1]))
            if not self.clean:
                acc_i = np.random.multivariate_normal([0., 0., 0.], self.cov_a).flatten()
            else:
                acc_i = np.random.multivariate_normal([0., 0., 0.], self.cov_a).flatten()*0
    
    
            acc_m = acc_m_q[1:4] + np.array([[acc_i[0]], [acc_i[1]], [acc_i[2]]])
            # acc_m = acc_m_q[1:4]
            acc_m = acc_m / np.linalg.norm(acc_m)
            # assume the dip angle of mag is diata = 30 degree =0.52 rad = (np.pi*30/180)
            mag_m_q = np.dot(np.dot(quatLeftMulMat(quatConj(self.q_t)), quatRightMulMat(self.q_t)),
                             quatPure2Q([np.cos(np.pi * 30 / 180), 0, np.sin(np.pi * 30 / 180)]))
            if not self.clean:
                mag_i = np.random.multivariate_normal([0, 0, 0], self.cov_mag).flatten()
            else:
                mag_i = np.random.multivariate_normal([0, 0, 0], self.cov_mag).flatten()*0
    
            mag_m = mag_m_q[1:4] + np.array([[mag_i[0]], [mag_i[1]], [mag_i[2]]])
            # mag_m = mag_m_q[1:4]
            mag_m = mag_m / np.linalg.norm(mag_m)


        # 3. update the hat_q and q_pred from the last round
        self.q_gyro = np.dot(quatLeftMulMat(self.q_gyro), exp(0.5 * self.dt *np.vstack( omega_obs)).T)
        
        hat_q = self.hat_q
        P = self.P

        b_t_ = np.dot(np.dot(quatLeftMulMat((hat_q)), quatRightMulMat(quatConj(hat_q))),
                          quatPure2Q([mag_m[0][0], mag_m[1][0], mag_m[2][0]]))
        b_t = np.array([[np.linalg.norm(b_t_[1:3])], [0], b_t_[3]])
        
        # 4. dynamic function
        a = self.dt/2.0*omega_obs
        norm_a = np.linalg.norm(a)
        F_t = quatRightMulMat(np.vstack([np.array([np.cos(norm_a)]), np.sin(norm_a)*a/norm_a]))
        G_t = -self.dt/2.0 * np.dot(quatLeftMulMat(hat_q) , np.vstack([np.zeros(3),np.eye(3)]))
        q_pred = np.dot(F_t, hat_q)
        P = np.dot(np.dot(F_t, P),np.transpose(F_t)) + np.dot(np.dot(G_t, self.cov_noise_i*10.),np.transpose(G_t))
        
        # 5. measurement function
        q_w = q_pred[0]
        q_v = q_pred[1:4]
        acc_mv = np.array([acc_m[0][0], acc_m[1][0], acc_m[2][0]])
        H_t_11 = q_w * acc_mv + np.cross(q_v, acc_mv)
        H_t_124 = (q_v[0] * acc_m[0] + q_v[1] * acc_m[1] + q_v[2] * acc_m[2]) * np.identity(3) + np.resize(
            np.kron(q_v, acc_mv.T), (3, 3)) - np.resize(np.kron(acc_mv, q_v.T), (3, 3)) - q_w * vecCross(acc_mv)
        mag_mv = np.array([mag_m[0][0], mag_m[1][0], mag_m[2][0]])
        H_t_21 = q_w * mag_mv + np.cross(q_v, mag_mv)
        H_t_224 = (q_v[0] * mag_mv[0] + q_v[1] * mag_mv[1] + q_v[2] * mag_mv[2]) * np.identity(3) + np.resize(
            np.kron(q_v, mag_mv.T), (3, 3)) - np.resize(np.kron(mag_mv, q_v.T), (3, 3)) - q_w * vecCross(mag_mv)
        H_t_1 = np.hstack((np.array([[H_t_11[0]], [H_t_11[1]], [H_t_11[2]]]), H_t_124))
        H_t_2 = np.hstack((np.array([[H_t_21[0]], [H_t_21[1]], [H_t_21[2]]]), H_t_224))
        H_t = 2 * np.concatenate([H_t_1, H_t_2])  # y = H_t*y_gyro + yi
        
        R = np.vstack([np.hstack([self.cov_a,np.eye(3)*0]), np.hstack([np.eye(3)*0,self.cov_mag])])
        S = np.dot(np.dot(H_t, P),np.transpose(H_t)) + R
        K = np.dot(np.dot(P, np.transpose(H_t)), np.linalg.inv(S))
        
        y =  np.vstack([np.array([[0.0],[0.0],[1.0]]), b_t])
        hat_y = np.dot(H_t, np.vstack(q_pred))
        err = y - hat_y
        hat_q_ekf = q_pred + np.hstack(np.dot(K, err))
        P = np.dot((np.eye(4) - np.dot(K, H_t)), P)

        # 6. normalization
        hat_q_ekf = hat_q_ekf/ np.linalg.norm(hat_q_ekf)
        J1 = np.outer(np.transpose(hat_q_ekf), hat_q_ekf) / np.power( np.linalg.norm(hat_q_ekf) , 3)
        P = np.dot(np.dot(J1, P),np.transpose(J1))
        # y0-y2  重力加速度在世界坐标下的方向
        # y3-y5  磁场强度方向在世界坐标系下的方向
        # q0-q3  四元数，当前传感器相对于世界坐标系的旋转姿态，角度各种耦合
        
        # combine rl action      
        acc_m_q = np.dot(np.dot(quatLeftMulMat((hat_q_ekf)), quatRightMulMat(quatConj(hat_q_ekf))), quatPure2Q(acc_mv))  
        mag_m_q = np.dot(np.dot(quatLeftMulMat((hat_q_ekf)), quatRightMulMat(quatConj(hat_q_ekf))),
                         quatPure2Q(mag_mv))
        hat_y2 = np.vstack([acc_m_q[1:4], mag_m_q[1:4]])
        
        hat_eta = np.array([0.0,0.0,0.0])
        hat_eta[0] = u_11 * (y[0][0] - hat_y2[0]) + u_21 * (y[1][0] - hat_y2[1]) + u_31 * (y[2][0] - hat_y2[2]) \
                    + u_41 * (y[3][0] - hat_y2[3]) + u_51 * (y[4][0] - hat_y2[4]) + u_61 * (y[5][0] - hat_y2[5])
        hat_eta[1] =  u_12 * (y[0][0] - hat_y2[0]) + u_22 * (y[1][0] - hat_y2[1]) + u_32 * (y[2][0] - hat_y2[2]) \
                    + u_42 * (y[3][0] - hat_y2[3]) + u_52 * (y[4][0] - hat_y2[4]) + u_62 * (y[5][0] - hat_y2[5])
        hat_eta[2] = u_13 * (y[0][0] - hat_y2[0]) + u_23 * (y[1][0] - hat_y2[1]) + u_33 * (y[2][0] - hat_y2[2]) \
                    + u_43 * (y[3][0] - hat_y2[3]) + u_53 * (y[4][0] - hat_y2[4]) + u_63 * (y[5][0] - hat_y2[5])
        hat_delta_q_rl = exp(0.5*np.vstack(hat_eta))
        hat_q = np.inner(quatLeftMulMat(hat_delta_q_rl), hat_q_ekf)
             
        # 7. relinearize
        hat_q = hat_q/ np.linalg.norm(hat_q)
        J2 = quatLeftMulMat(hat_delta_q_rl)
        P = np.dot(np.dot(J2, P),np.transpose(J2))

        aaa = 2 * Log(np.inner(quatRightMulMat(quatConj(hat_q)),np.hstack((self.q_t)))) #0<|aaa|<2*pi
        if np.linalg.norm(aaa)<np.pi:
            cost = np.linalg.norm(aaa)**2
        else:
            cost = (np.linalg.norm(aaa)-2*np.pi)**2    

        if cost > (10):
            done = True
        else:
            done = False

        cost_1 = cost
        cost_2 = cost
        cost=-cost

        # 6. update new for next round
        self.hat_q = hat_q
        self.P = P
        q_gyro = self.q_gyro
        # self.state = hat_eta
        self.t = self.t + self.dt

        # eul_hat_q = quat2eul(hat_q)
        # eul_q_t = quat2eul(q_t)
        if self.choice == 'saveData':
            return omega_obs,acc_m,mag_m,q_t,hat_q, cost, done, dict(reference=y[0],
                                        state_of_interest=np.array([hat_q[1], q_t[1],hat_q[2], q_t[2]]))
            # return omega,omega_obs,acc_m,mag_m,q_t,self.q_tt,hat_q, cost, done, dict(reference=y[0],
            #                             state_of_interest=np.array([hat_q[1], q_t[1],hat_q[2], q_t[2]]))
        else:
            # return hat_eta, cost, done, dict(
            #     reference=np.array([q_t[0], q_t[1], q_t[2], q_t[3],self.q_gyro[0],self.q_gyro[1],self.q_gyro[2],self.q_gyro[3]]),
            #     state_of_interest=np.array([hat_q[0], hat_q[1], hat_q[2], hat_q[3]]))
            
            if self.returnQ:
                # return hat_eta, np.array([cost_1, cost_2]), done, dict(
                return hat_eta, cost, done, dict(
                    reference=np.array(
                        [q_t[0], q_t[1], q_t[2], q_t[3], q_gyro[0], q_gyro[1], q_gyro[2], q_gyro[3]]),
                    state_of_interest=np.array([hat_q[0], hat_q[1], hat_q[2], hat_q[3]]))
            else:
                return hat_eta, cost, done, dict(
                    reference=np.array(np.hstack(
                        [quat2eul(q_t), quat2eul(q_gyro)])),
                    state_of_interest=np.array(np.hstack(
                        [quat2eul(hat_q)])))
            # DEBUG
            # return hat_eta, omega, cost, done, dict(
            #     reference=np.array([q_t[0], q_t[1], q_t[2], q_t[3]]),
            #     state_of_interest=np.array([hat_q[0], hat_q[1], hat_q[2], hat_q[3]]))

    def reset(self, eval=False):
        self.t = 0

        if self.realMeasurement:
            p = r'D:\reinforcement learning\learning to SLAM\code\manon data\forWeiPan\ManonData.csv'
            self.data = np.genfromtxt(p, delimiter=',')
            self.q_t,_,_,_=measrement_real(self,self.t)
        else:  
            if not eval:
                self.q_t = np.random.uniform([-1,-1,-1,-1], [1,1,1,1])# the state q, the initial value can be set randomly
                # self.q_t = np.array([0.0, 0.5, 0.1, 0.8])
                # self.q_t = np.array([-0.6, 0.3, 0.35, 0.6])
            else:
                self.q_t = np.array([0.3,0.5,0.1,0.8])#work
                # self.q_t = np.array([0.0, 0.5, 0.1, 0.8])
                # self.q_t = np.array([-0.6,0.3,0.35,0.6])
            self.q_t = self.q_t /np.linalg.norm(self.q_t)
        self.q_gyro = self.q_t
        self.q_t_init = self.q_t
        self.hat_q = self.q_t + np.random.normal([0, 0, 0, 0], [0.1, 0.1, 0.1, 0.1]) *1
        # self.hat_q = np.random.uniform([-1, -1, -1, -1], [1, 1, 1, 1])
        self.hat_q = self.hat_q /np.linalg.norm(self.hat_q)
        self.q_pred_init = self.hat_q
        self.P = np.eye(4)*0.1

        hat_eta = np.random.normal([ 0,0,0], [ 0.1,0.1,0.1])*0.0001
        omega_obs = np.array([[0.],[0.],[0.]])
        self.state = hat_eta

        if self.choice == 'saveData':
            omega_obs = np.array([[0],[0],[0]])
            acc_m = np.array([[0],[0],[0]])
            mag_m= np.array([[0],[0],[0]])
            return omega_obs,acc_m,mag_m,self.q_t,self.q_gyro
        else:
            # return np.hstack([hat_eta,np.hstack(omega_obs)])  # return hat_state
            return hat_eta

    def render(self, mode='human'):

        return


    def saveChoice(self, choiceIn):
        self.choice =choiceIn
        return self.choice



if __name__ == '__main__':
    env = Ex3_EKF_gyro()
    T = 50

    choice = 'saveData'
    # choice = []
    if env.saveChoice(choice) == 'saveData':
        s = env.reset()
        path = []
        steps = []
        omega_obs,acc_m,mag_m,q_t,q_gyro = env.reset()
        measurement = np.vstack([omega_obs,acc_m,mag_m,np.vstack(q_t),np.vstack(q_gyro)])
        measurement = np.hstack(measurement)
        path.append(measurement)
        steps.append(0)
        for i in range(int(T / env.dt)):
            # s, r, info, done = env.step(np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]))
            # path.append(s)
            omega_obs,acc_m,mag_m,q_t,q_gyro, r, info, done = env.step(np.array([ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]))
            measurement = np.vstack([omega_obs,acc_m,mag_m,np.vstack(q_t),np.vstack(q_gyro)])
            measurement = np.hstack(measurement)
            path.append(measurement)
            steps.append(i/100.0)
        np.savetxt('5.csv', path, delimiter = ',')
        colors = "bgrcmk"
        cycol = cycle(colors)
        fig = plt.figure(figsize=(9, 6))
        ax = fig.add_subplot(111)
        color1 = next(cycol)
        ax.plot(steps, np.array(path)[:, 9+4], color=color1, label='x0', linestyle="dashed" )
        ax.plot(steps, np.array(path)[:, 9], color=color1, label='x0')
        color1 = next(cycol)
        ax.plot(steps, np.array(path)[:, 10+4], color=color1, label='x1', linestyle="dashed")
        ax.plot(steps, np.array(path)[:, 10], color=color1, label='x1')
        color1 = next(cycol)
        ax.plot(steps, np.array(path)[:, 11+4], color=color1, label='x2', linestyle="dashed")
        ax.plot(steps, np.array(path)[:, 11], color=color1, label='x2')
        color1 = next(cycol)
        ax.plot(steps, np.array(path)[:, 12+4], color=color1, label='x3', linestyle="dashed")
        ax.plot(steps, np.array(path)[:, 12], color=color1, label='x3')
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles, labels, loc=2, fancybox=False, shadow=False)
        # xlim0 = 38.5
        # xlim1 = 40.1
        # ylim0 = -0.9
        # ylim1 = 0.9
        # ax.set_xlim(xlim0, xlim1)
        # ax.set_ylim(ylim0, ylim1)
        plt.show()
        fig = plt.figure(figsize=(9, 6))
        ax = fig.add_subplot(111)
        ax.plot(steps, np.array(path)[:, 0], color='green', label='$\omega_x$')
        ax.plot(steps, np.array(path)[:, 1], color='yellow', label='$\omega_y$')
        ax.plot(steps, np.array(path)[:, 2], color='blue', label='$\omega_z$')
        handles, labels = ax.get_legend_handles_labels()
        plt.ylabel("Angular Velocity(rad/s)",fontsize=20)
        plt.xlabel("Time(s)",fontsize=20)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20) 
        plt.gcf().subplots_adjust(bottom=0.15,left=0.15)
        ax.legend(handles, labels, loc=2, bbox_to_anchor=(0.75,0.98),fancybox=False, shadow=False,fontsize=20)
        plt.show()
        # plt.savefig('1-.eps',format="eps")
        print('done')
    else:
        path = []
        # path2=[]
        t1 = []
        s = env.reset()
        for i in range(int(T / env.dt)):
            # s, r, info, done = env.step(np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]))
            # path.append(s)
            hat_q, omega, r, info, done = env.step(np.array([ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]))
            path.append(omega)
            t1.append(i * env.dt)
        fig = plt.figure(figsize=(9, 6))
        ax = fig.add_subplot(111)
        ax.plot(t1, np.array(path)[:, 0], color='green', label='x0')
        ax.plot(t1, np.array(path)[:, 1], color='yellow', label='x1')
        ax.plot(t1, np.array(path)[:, 2], color='blue', label='x2')
        # ax.plot(t1, np.array(path)[:, 3], color='red', label='x3')
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles, labels, loc=2, fancybox=False, shadow=False)
        plt.show()
        # plt.savefig('1-.eps',format="eps")
        print('done')
        # fig = plt.figure(figsize=(9, 6))
        # ax = fig.add_subplot(111)
        # ax.plot(t1, np.array(path2)[:, 1], color='green', label='x0')
        # ax.plot(t1, np.array(path2)[:, 1], color='yellow', label='x1')
        # ax.plot(t1, np.array(path2)[:, 2], color='blue', label='x2')
        # ax.plot(t1, np.array(path2)[:, 3], color='red', label='x3')
        # handles, labels = ax.get_legend_handles_labels()
        # ax.legend(handles, labels, loc=2, fancybox=False, shadow=False)
        # plt.show()
        # print('done')
