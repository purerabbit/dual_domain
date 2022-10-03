import torch.nn as nn
#3 type of loss
'''
Lpdc=||Ypred1-y||1 + ||Ypred2-y||1 +||Ypredu-y||1

Lac=lamda_img*L_img + lamda_grad*L_grad

'''
'''
input:Y(pred_1,pred_2,pred_u),X_p1,X_p2,X_u

'''
#input(B,C,H,W)
def cal_gradv(x):
    v_grad =x[:,:,1:,:]-x[:,:,:-1,:]
    return v_grad
def cal_gradh(x):
    h_grad =x[:,:,:,1:]-x[:,:,:,:-1]
    return h_grad
def cal_grad_lossv(a,b):
    av=cal_gradv(a)
    bv=cal_gradv(b)
    lossl1=nn.L1Loss()
    return lossl1(av,bv)
def cal_grad_lossh(a,b):
    ah=cal_gradh(a)
    bh=cal_gradh(b)
    lossl1=nn.L1Loss()
    return lossl1(ah,bh)

def cal_loss(y,y1,y2,yu,x,x1,x2,xu):
    lossl1=nn.L1Loss()
    #k_space loss
    L_pdc=lossl1(y1,y)+lossl1(y2,y)+lossl1(yu,y)

    #image loss
    L_img=lossl1(x,x1)+lossl1(x,x2)+lossl1(x1,x2)
    L_grad=cal_grad_lossv(xu,x1)+cal_grad_lossv(xu,x2)+cal_grad_lossv(x1,x2)+cal_grad_lossh(xu,x1)+cal_grad_lossh(xu,x2)+cal_grad_lossh(x1,x2)
    L_ac=2*L_img+L_grad

    #total loss
    L_tot=10*L_pdc+L_ac
    print('----------------------L_tot:',L_tot)
    return L_tot
