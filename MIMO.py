import os
import sys
import numpy as np
import json
import multiprocessing
import time
from scipy import linalg
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim

rng = np.random.RandomState(42)

class Net(nn.Module):
    def __init__(self,n,h):
        super(Net, self).__init__()
        self.detector = nn.Sequential(
            nn.Linear(n, h),
            nn.ReLU(),
            nn.Linear(h, h),
            nn.ReLU(),
            nn.Linear(h, n)
        )
    def forward(self, x):
        x = self.detector(x)
        x = torch.tanh(x)
        return x

class DL():
    def __init__(self, nepoch= 10000, mbs = 5, n = 8,  h = 30, adam_lr = 0.001):
        self.nepoch = nepoch
        self.mbs = mbs  # ミニバッチサイズ
        self.n = n  # アンテナ数
        self.h = h  # 隠れ層
        self.adam_lr = adam_lr # Adamの学習率
        self.model = Net(n,h)  # ネットワークインスタンス生成
        self.loss_func = nn.MSELoss()  # 損失関数(二乗損失関数)
        self.optimizer = optim.Adam(self.model.parameters(), lr=adam_lr)  # オプティマイザ

    def gen_minibatch(self, H, mod, N0):
        N, M = H.shape

        # Source bits / alphabet / symbol
        b = torch.randint(0, 2, (M * mod.ml, self.mbs))
        a = np.dot(np.kron(mod.lv, np.eye(M, dtype=int)), b)
        x = np.array(mod.val[a])

        # Channel
        z = (np.random.randn(N, self.mbs) + 1j * np.random.randn(N, self.mbs)) * np.sqrt(N0 / 2)
        y = np.dot(H, x) + z

        x = x * np.sqrt(2.0)
        y = y * np.sqrt(2.0)

        xx = torch.from_numpy(np.concatenate([x.real, x.imag])).transpose(0, 1)
        yy = torch.from_numpy(np.concatenate([y.real, y.imag])).transpose(0, 1)

        return xx.float(), yy.float()

    def trainMIMO(self, H, mod, N0):
        for i in range(self.nepoch):
            x, y = self.gen_minibatch(H, mod, N0)  # ミニバッチの生成
            self.optimizer.zero_grad()  # オプティマイザの勾配情報初期化
            estimate = self.model(y)  # 推論計算
            self.loss = self.loss_func(x, estimate)  # 損失値の計算
            self.loss.backward()  # 誤差逆伝播法
            self.optimizer.step()  # 学習可能パラメータの更新

class CH():
    def __init__(self, M, N, K, Kp):
        self.M = M
        self.N = N
        self.K = K
        self.Kp = Kp

    def fading_gen(self):
        self.H = (np.random.randn(self.N,self.M) + 1j* np.random.randn(self.N,self.M))/np.sqrt(2)

    def awgn_gen(self, N0):
        self.z = (np.random.randn(self.N,self.K+self.Kp) + 1j* np.random.randn(self.N,self.K+self.Kp))*np.sqrt(N0/2)

class MOD():
    ml = 1
    val = []
    w =[]
    def __init__(self, ml):
        self.ml = ml
        self.nsym = 2**ml
        if ml==1:
            self.w = np.array(1)
            self.val = np.array([-1,1])
            self.lv = np.array(1)
        # elif ml == 2:
        #     self.val = np.array([-1-1j, -1+1j, 1-1j, 1+1j])/np.sqrt(2)
        #     self.lv = np.array([2, 1])
        else:
            self.rep_b = np.empty((ml, self.nsym), int)
            for idx in range(0, self.nsym):
                for idx_ in range(0, ml):
                    self.rep_b[idx_, idx] = (idx >> (self.ml - idx_ - 1)) % 2
            self.w = 2**np.arange(ml/2)[::-1]
            self.w = np.concatenate([self.w, 1j * self.w], axis=0)
            self.val = np.dot(self.w, 2 * self.rep_b - 1)
            #self.norm = 1/np.sqrt(np.mean(np.abs(self.val) ** 2))
            self.norm = np.sqrt(3/(2*(self.nsym-1)))
            self.val *= self.norm
            self.Esmax = np.max(np.abs(self.val) ** 2)
            self.lv = 2**np.arange(ml)[::-1]
            self.lay = (np.arange(ml)*2-(ml-1))*self.norm

    def mld_replica(self, M):
        self.sm_nbit = self.ml*M
        self.sm_nrep = 2**(self.ml*M)
        self.rep_b = np.empty((self.sm_nbit,self.sm_nrep), int)
        
        for idx in range (0,self.sm_nrep):
            for idx_ in range (0,self.sm_nbit):
                self.rep_b[idx_,idx] = (idx>>(self.sm_nbit-idx_-1))%2
        self.rep_a = np.dot(np.kron(self.w,np.eye(M, dtype=int)),self.rep_b)
        self.rep_x = np.array(self.val[self.rep_a])

class DET():
    def demod(self,y,mod):
        if mod.ml==1:
            return (y.real>0)*1
        elif mod.ml == 2:
            return np.concatenate([y.real>0, y.imag>0], axis=0)*1
        else:
            b_ = np.empty((mod.ml*y.shape[0], y.shape[1]), int)
            b_tmp = np.empty((y.shape[0],mod.ml), int)
            for idx_k in range(0, y.shape[1]):
                a_ = np.argmin(np.abs(y[:, idx_k] - np.tile(mod.val, (y.shape[1], 1)).transpose()) ** 2, axis=0)
                for idx_m in range(0, y.shape[0]):
                    b_tmp[idx_m,:] = mod.rep_b[:,a_[idx_m]]
                b_[:, idx_k] = b_tmp.transpose().reshape(-1)
            return b_

    def remodel(self, y_, H_):
        y      = np.concatenate([y_.real, y_.imag], axis=0)
        H1     = np.concatenate([H_.real, -H_.imag], axis=1)
        H2     = np.concatenate([H_.imag, H_.real], axis=1)
        H      = np.concatenate([H1, H2], axis=0)
        return y, H
        
    def mld(self ,y, H, rep_b, rep_x, ml):
        N,K = y.shape
        nbit, nvec = rep_b.shape
        y_rep = np.dot(H,rep_x)

        b_ = np.empty((nbit,K), int)
        for idx_k in range (0,K):
            a_ = np.argmin(np.sum(np.abs(y[:,idx_k]-y_rep.T)**2,axis=1))
            b_[:,idx_k] = rep_b[:,a_]
        return b_

    def zf(self ,y, H, mod):
        W = np.linalg.pinv(H)
        y_ = np.dot(W,y)
        return DET.demod(self, y_, mod)

    def mmse(self,y, H, N0, mod):
        N,M = H.shape
        W = np.dot( np.conjugate((H.T)), np.linalg.pinv(np.dot(H, np.conjugate(H.T)) + N0*np.eye(N)))
        y_= np.dot(W,y)
        return DET.demod(self, y_, mod)
    
    def mf(self,y, H, N0, mod):
        W = np.conjugate((H.T))
        y_= np.dot(W,y)
        return DET.demod(self, y_, mod)

    def gabps(self, y_, H_, N0, mod, Niter  = 16):
        N_, M_ = H_.shape; N_, K  = y_.shape
        y, H   = DET.remodel(self, y_, H_)
        N, M   = H.shape
        HH     = (H*H).T

        x_= np.zeros((M,K))
        zeta = np.ones((N,Niter))*0.5
        mu = np.ones(Niter)

        for idx_sym in range (0, K):
            SR_mat = np.zeros((M,N)); ER_mat = np.ones((M, N)) / 2
            uu = np.zeros((M,N)); vv = np.zeros((M,N))

            # Perfect priori
            # SR_mat = np.tile(x, (1, N))
            # ER_mat = SR_mat**2
            for idx_iter in range (0, Niter):
                # SC
                Reconstruct_matrix = H.T*SR_mat
                y_tilde = y[:,idx_sym] - np.sum(Reconstruct_matrix, axis=0) + Reconstruct_matrix
                delta = ER_mat - SR_mat**2

                # BG
                element = HH * delta
                psi = (np.sum(element,axis=0).reshape(1,-1) - element) + N0/2

                u = H.T*y_tilde/psi
                uu = zeta[:,idx_iter] * u + (1-zeta[:,idx_iter])*uu
                s = (np.sum(uu, axis=1) - uu.transpose()).transpose()
                v = HH/psi
                vv = zeta[:, idx_iter] * v + (1 - zeta[:, idx_iter]) * vv
                omega = (np.sum(vv, axis=1) - vv.transpose()).transpose()
                gamma = s/omega

                # RG
                SR_mat = np.zeros((M, N))
                ER_mat = np.zeros((M, N))
                for gamma_ in mod.lay:
                    temp = mu[idx_iter]*(gamma-gamma_)/mod.norm
                    SR_mat += np.tanh(temp)
                    ER_mat += gamma_*np.tanh(temp)
                SR_mat *= mod.norm
                ER_mat *= 2*mod.norm
                ER_mat += mod.Esmax/2
            x_[:,idx_sym] = np.sum(u, axis=1)/np.sum(v, axis=1)
        return DET.demod(self, (x_[:M_,:] + 1j*x_[M_:,:]), mod)

    def gabp(self, y_, H_, N0, mod, Niter=16, weight=1, damp=0, eta=0.5):
        N_, M_ = H_.shape
        N_, K = y_.shape
        y, H = DET.remodel(self, y_, H_)
        N, M = H.shape
        HH = (H * H).T
        llr = np.zeros((M, K, Niter))

        W = 2 * np.sqrt(1 / 2.0) / N0
        for idx_sym in range(0, K):
            Beta_matrix_p = 0
            Beta_ave_p = 0
            SR_mat = np.zeros((M, N))
            # Perfect priori
            # SR_mat = np.tile(x[:,idx_sym], (N, 1)).T

            for idx_iter in range(0, Niter):
                # SC
                Reconstruct_matrix = H.T * SR_mat
                yn = y[:, idx_sym] - np.sum(Reconstruct_matrix, axis=0)

                hynm_matrix = H.T * (yn + Reconstruct_matrix)

                # Weight
                if weight == 1:
                    dnm = 1 / 2.0 - SR_mat * SR_mat
                    element = HH * dnm
                    W = 2 * np.sqrt(1 / 2.0) / (N0 + 2.0 * (np.sum(element, axis=0).reshape(1, -1) - element))

                # alpha matrix
                Alpha_matrix_c = W * hynm_matrix
                Alpha_ave_c = W * HH

                # Create final detection data
                llr[:, idx_sym, idx_iter] = np.sum(Alpha_matrix_c, axis=1)

                # beta matrix
                Beta_matrix_c = np.sum(Alpha_matrix_c, axis=1).reshape(-1, 1) - Alpha_matrix_c
                Beta_ave_c = np.sum(Alpha_ave_c, axis=1).reshape(-1, 1) - Alpha_matrix_c

                # lambda matrix
                if damp == 0 or idx_iter == 0:
                    Damped_matrix = Beta_matrix_c
                    norm_num = Beta_ave_c
                elif damp == 1:
                    Damped_matrix = eta * Beta_matrix_c + (1 - eta) * Beta_matrix_p
                    norm_num = eta * Beta_ave_c + (1 - eta) * Beta_ave_p
                else:
                    Damped_matrix = eta * Beta_matrix_c + (1 - eta) * Damped_matrix
                    norm_num = eta * Beta_ave_c + (1 - eta) * norm_num

                Lambda_matrix = Damped_matrix

                # Update
                Beta_matrix_p = Beta_matrix_c
                Beta_ave_p = Beta_ave_c

                # SG
                SR_mat = np.tanh(Lambda_matrix/2.0) * np.sqrt(1 / 2.0)

        return (llr[:, :, Niter - 1]>0)*1

    # def amp(self, y_, H_, N0, mod, Niter=16):
    #     N_, M_ = H_.shape; N_, K  = y_.shape
    #     y, H   = DET.remodel(self, y_, H_)
    #     N, M   = H.shape
    #     HH     = (H*H).T
    #
    #     for idx_sym in range (0, K):
    #         SR_matrix = np.zeros((M,N))
    #         for idx_iter in range (0, Niter):
    #             # SC
    #             Reconstruct_matrix = H.T*SR_matrix
    #             yn = y[:,idx_sym] - np.sum(Reconstruct_matrix, axis=0)
    #
    #             dnm = 1/2.0 - SR_matrix*SR_matrix
    #             element = HH*dnm
    #             W = 2*np.sqrt(1/2.0) / (N0/2.0 + np.sum(element,axis=0).reshape(1,-1) - element)
    #
    #     W = np.conjugate((H.T))
    #     x_= np.dot(W,y)
    #     return DET.demod(self, (x_[:M_,:] + 1j*x_[M_:,:]), mod)

    def dld(self,y, H, N0, mod):
        N, M = H.shape
        dl = DL(10000,5, 2*N,  30, 0.001)
        dl.trainMIMO(H, mod, N0)
        y = y * np.sqrt(2.0)
        yy = torch.from_numpy(np.concatenate([y.real, y.imag])).transpose(0, 1)
        yy = yy.float()
        x_ = dl.model(yy)
        x_ = x_.transpose(0, 1)
        return DET.demod(self, x_[:M, :] + 1j * x_[M:, :], mod)

def main_task(params):
    method = params[1][1]
    EsN0 = range(int(params[1][9]),int(params[1][11])+int(params[1][10]),int(params[1][10]))

    ch = CH(int(params[1][2]),int(params[1][3]),int(params[1][4]),int(params[1][5]))
    det = DET()

    mod = MOD(int(params[1][6]))
    if method =='MLD':
        mod.mld_replica(ch.M)

    wloop = int(params[1][7])
    nproc = int(params[1][8])
    nloop = int(np.ceil((10**wloop)/nproc))
    noe   = np.zeros((2,len(EsN0)),dtype = int)

    # Pilot
    p = linalg.hadamard(ch.Kp,dtype=complex)
    p = p[:ch.M,:]
            
    for idx_En in range (0, len(EsN0)):
        N0 = 10.0 ** (-EsN0[idx_En] / 10.0)
        # N0 = 10.0 ** (-EsN0[idx_En] / 10.0) * ch.M

        for idx_loop in range (0, nloop):
            # Fading generation
            ch.fading_gen()
    
            # Source bits / alphabet / symbol
            b = np.random.randint(0, 2, (ch.M*mod.ml, ch.K))
            a = np.dot(np.kron(mod.lv,np.eye(ch.M, dtype=int)),b)
            x = np.array(mod.val[a])
            
            # TX symbol
            s = np.concatenate([p, x], axis=1)
            
            # Channel
            ch.awgn_gen(N0)
            y = np.dot(ch.H,s) + ch.z
    
            # RX detector
            #H_ = np.dot(y[:,:ch.Kp],p.T)/ch.Kp
            H_ = ch.H

            if method =='MLD':
                b_ = det.mld(y[:,ch.Kp:], H_, mod.rep_b, mod.rep_x, mod)
            elif method =='ZF':
                b_ = det.zf(y[:,ch.Kp:], H_, mod)
            elif method =='MMSE':
                b_ = det.mmse(y[:,ch.Kp:], H_, N0, mod)
            elif method =='MF':
                b_ = det.mf(y[:,ch.Kp:], H_, N0, mod)
            elif method =='GaBP':
                if(mod.ml<4):
                    b_ = det.gabp(y[:,ch.Kp:], H_, N0, mod)
                else:
                    b_ = det.gabps(y[:, ch.Kp:], H_, N0, mod)
            elif method =='AMP':
                b_ = det.amp(y[:,ch.Kp:], H_, N0, mod, 16)
            elif method == 'DL':
                b_ = det.dld(y[:,ch.Kp:], H_, N0, mod)

            tmp = (b != b_).sum(axis=1)
            noe[0,idx_En] += tmp.sum()
            noe[1,idx_En] += (mod.ml*ch.M*ch.K)
            if noe[0,idx_En] > (mod.ml*ch.M*ch.K*nloop)*0.01:
                break
    return noe

def resut2f(argvs,BER):
    EsN0 = range(int(argvs[9]),int(argvs[11])+int(argvs[10]),int(argvs[10]))
    for idx_En in range (0, len(EsN0)):
        SIM_dict = {'Method':argvs[1], 'M':argvs[2], 'N':argvs[3], 'K':argvs[4],'Kp':argvs[5], 'ml':argvs[6], 'wloop':argvs[7],'EsN0':0.0,'BER':0.0}
        SIM_dict['EsN0'] = EsN0[idx_En]
        SIM_dict['BER'] = BER[idx_En]

        fn = 'DATA/'
        fn += SIM_dict['Method']
        fn += '.json'
        f_out = open(fn, 'a')
        json.dump(SIM_dict, f_out)
        f_out.write("\n")
        f_out.close()
    
if __name__ == '__main__':
    argvs = sys.argv  # コマンドライン引数を格納したリストの取得
    argc = len(argvs) # 引数の個数
    if (argc != 12):  # 引数が足りない場合
        print('Usage: # MIMO.py DET M N K Kp ml wloop nproc En1 delta En2')
        quit()        # プログラムの終了
    
    fn = argvs[1];    fn += '.json'
    if os.path.exists(fn):
        os.remove(fn)

    start = time.time()
    params = [(i, argvs) for i in range (0, int(argvs[8]))]

    if int(argvs[8])==1:
        res = main_task((0,argvs))
    else:
        pool = multiprocessing.Pool(processes=int(argvs[8]))
        res_ = pool.map(main_task, params)
        pool.close()
        res = sum(res_)
    BER = res[0]/res[1]

    elapsed_time = time.time() - start
    print ("elapsed_time:{0}".format(elapsed_time) + "[sec]")
    
    resut2f(argvs,BER)
    
    EsN0 = range(int(argvs[9]),int(argvs[11])+int(argvs[10]),int(argvs[10]))
    fig = plt.plot(EsN0, BER, 'bo', EsN0, BER, 'k')
    plt.axis([int(argvs[9]), int(argvs[11]), 1e-5, 1])
    plt.xscale('linear')
    plt.yscale('log')
    plt.xlabel(r'$E_s/N_0$ [dB]')
    plt.ylabel('BER')
    plt.title(argvs[1]+' (M='+ str(argvs[2]) + ', N=' + str(argvs[3])+ ', K=' + str(argvs[4])+ ', Q=' + str(2**int(argvs[6]))+')')
    plt.grid(True)
    plt.savefig('FIG/' + argvs[1]+'_'+ str(argvs[2]) + '_' + str(argvs[3])+ '_' + str(argvs[4])+ '_' + str(argvs[6])+'.eps', bbox_inches="tight", pad_inches=0.05)
    plt.show()