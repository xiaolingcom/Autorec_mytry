import torch as t
import torch.nn as nn
import pickle
from autoRec_Params import *
import numpy as np
from ToolScripts.TimeLogger import log
import sys

device = t.device('cuda' if t.cuda.is_available() else 'cpu')
print('device: '+str(device))
resultFile='result.txt'

class autoEncoder():
    def __init__(self,trainMat,testMat,LR,LR_decay,V_regularWeight,W_regularWeight,is_loadModel=False):
        self.trainMat = trainMat
        self.testMat = testMat
        self.LR = LR
        self.LR_decay = LR_decay
        self.V_regularWeight = V_regularWeight
        self.W_regularWeight = W_regularWeight
        self.is_loadModel = is_loadModel
        
        self.trainMask = (trainMat != 0)
        self.testMask = (testMat != 0)
        self.loss_mse = nn.MSELoss(reduction = 'sum') #没有求mean=(x-y)^2   

        self.curEpoch = 0
        #画图所要用到的数据
        self.train_losses=[]
        self.train_RMSEs=[]
        self.test_losses=[]
        self.test_RMSEs=[]
        self.step_losses=[]

    def prePareModel(self):
        #参数初始化
        if MOVIE_BASED:
            D_in=USER_NUM
            D_out=USER_NUM
        else:
            D_in=MOVIE_NUM
            D_out=MOVIE_NUM
        self.V = t.empty(D_in,LATENT_DIM,dtype=t.float,device=device,requires_grad=True) #需要求梯度,需要更新
        self.W = t.empty(LATENT_DIM,D_out,dtype=t.float,device=device,requires_grad=True)
        self.b1=t.zeros(LATENT_DIM,dtype=t.float,device=device,requires_grad=True)
        self.b2=t.zeros(D_out,dtype=t.float,device=device,requires_grad=True)
        nn.init.xavier_uniform_(self.V) 
        nn.init.xavier_uniform_(self.W)
        self.optimizer=t.optim.Adam([self.V,self.W,self.b1,self.b2],lr=self.LR)

    def model(self,train):
        #model要做的事情是将初始化好的数据串起来
        self.encoder=t.sigmoid(t.mm(train,self.V)+self.b1)
        self.decoder=t.mm(self.encoder,self.W)+self.b2
        return self.decoder

    def trainModel(self,trainMat,trainMask,op):
        #一个epoch
        num=trainMat.shape[0]
        shuffledIds=np.random.permutation(num)
        steps=int(np.ceil(num/BATCH_SIZE)) #没有整除，取上值
        epoch_loss=0
        epoch_rmse_loss=0
        epoch_rmse_num=0
        for i in range(steps):
            ed=min((i+1)*BATCH_SIZE,num) 
            batch_Ids=shuffledIds[i*BATCH_SIZE:ed]
            batch_len=len(batch_Ids) #后面计算batch_rmse需要用到

            #准备第i步训练所用的数据
            ##转换为numpy格式
            tmpTrain=trainMat[batch_Ids].toarray()
            tmpMask=trainMask[batch_Ids].toarray()
            ##转换为tensor格式
            train=t.from_numpy(tmpTrain).float().to(device)
            mask=t.from_numpy(1*tmpMask).float().to(device)

            y_pred=self.model(train) #第i步所用数据的预测值
            #目标函数
            pred_loss=self.loss_mse(y_pred*mask,train)/batch_len #为什么这里要除一个batchsize呢，原文的损失函数好像没有除
            v_loss=t.sum(self.V*self.V) #t.sum而不是t.mm
            w_loss=t.sum(self.W*self.W) #*是element-wise操作
            batch_loss=pred_loss + self.V_regularWeight*v_loss +self.W_regularWeight*w_loss
            #计算误差rmse
            epoch_loss+=batch_loss.item()
            epoch_rmse_loss+=self.RMSE(y_pred.cpu().detach().numpy(),tmpTrain,tmpMask)#//////////////////②?????/////////////
            epoch_rmse_num+=t.sum(mask).item()

            log('step %d/%d, step_loss=%f'%(i,steps,batch_loss.item()),save=False,oneline=True)
            op.zero_grad() #清空梯度，不累加
            batch_loss.backward()
            op.step()
        epoch_rmse=np.sqrt(epoch_rmse_loss/epoch_rmse_num)
        return epoch_loss,epoch_rmse

    def testModel(self,trainMat,testMat,testMask,div):
        num=trainMat.shape[0]
        shuffledIds=np.random.permutation(num)
        steps=int(np.ceil(num/(BATCH_SIZE*div)))
        epoch_loss=0
        epoch_rmse_loss=0
        epoch_rmse_num=0
        for i in range(steps):
            ed=min((i+1)*BATCH_SIZE,num)
            batch_Ids=shuffledIds[i*BATCH_SIZE:ed]
            batch_len=len(batch_Ids)
            #取一个batch数据
            tmpTrain=trainMat[batch_Ids].toarray()
            tmpTest=testMat[batch_Ids].toarray()
            tmptestMask=testMask[batch_Ids].toarray()

            #转换为tensor格式
            train=t.from_numpy(tmpTrain).float().to(device)
            test=t.from_numpy(tmpTest).float().to(device)
            mask=t.from_numpy(1*tmptestMask).float().to(device) #bool转换为int

            y_pred=self.model(train)
            #用测试集的误差去衡量泛化误差,这个时候的参数应该训练好了
            epoch_rmse_loss+=self.RMSE(y_pred.cpu().detach().numpy(),tmpTest,tmptestMask)
            epoch_loss+=epoch_rmse_loss #/////////////???/////////////
            epoch_rmse_num+=t.sum(mask).item()
        epoch_rmse=np.sqrt(epoch_rmse_loss/epoch_rmse_num)
        return epoch_loss,epoch_rmse

    def run(self):
        self.prePareModel()
        # if self.isLoadModel:   #先不管这个
        #     self.loadModel(LOAD_MODEL_PATH)
        #训练
        for e in range(self.curEpoch,EPOCH+1):
            epoch_loss,epoch_rmse=self.trainModel(self.trainMat,self.trainMask,self.optimizer)
            log("epoch %d/%d, epoch_loss=%.2f, epoch_rmse=%.4f"%(e,EPOCH,epoch_loss,epoch_rmse))
            self.train_losses.append(epoch_loss)
            self.train_RMSEs.append(epoch_rmse)
            #打印正则损失
            log('V_Loss = %.2f, W_Loss = %.2f'%(t.sum(self.V * self.V), t.sum(self.W * self.W)))

            #交叉验证
            cv_epoch_loss,cv_epoch_rmse=self.testModel(self.trainMat,self.testMat,self.testMask,5)#//////////①???///////////
            log("epoch %d/%d, cv_epoch_loss=%.2f,cv_epoch_rmse=%.4f"%(e, EPOCH, cv_epoch_loss,cv_epoch_rmse))
            log("\n")
            self.test_losses.append(cv_epoch_loss)
            self.test_RMSEs.append(cv_epoch_rmse)

            #调整学习率并保存
            self.curLr = self.adjust_learning_rate(self.optimizer, e)
            self.curEpoch=e #记录当前的EPOCH，用于保存model
            # if e%10==0 and e!=0:
            #     self.saveModel()
            #     test_epoch_loss=self.testModel(self.trainMat,self.testMat,self.testMask,1)
            #     log("epoch %d/%d, test_epoch_loss=%.2f"%(e, EPOCH, test_epoch_loss))
            #     self.step_losses.append(test_epoch_loss)
            #     # for i in range(len(self.step_losses)):
            #     #     print("***************************")
            #     #     print("rmse = %.4f"%(self.step_rmse[i]))
            #     #     print("***************************")
        #测试
        _,test_rmse=self.testModel(self.trainMat,self.testMat,self.testMask,1)
        self.writeResult(test_rmse)
        log("\n")
        log("test_rmse=%.4f"%(test_rmse))
        self.getModelName()

    #根据epoch数调整学习率
    def adjust_learning_rate(self, optimizer, epoch):
        LR = self.LR * (LR_DECAY**epoch)
        for param_group in optimizer.param_groups:
            param_group['lr'] = LR #保存////////////3//////////////////
        return LR

    def saveModel(self):
        history=dict()
        history['losses']=self.train_losses
        history['val_losses']=self.test_losses #验证
        ModelName=self.getModelName()
        
        savePath = r'./' + dataset + r'/Model/' + ModelName + r'.pth'
        # t.save({
        #     'epoch': self.curEpoch,
        #     'lr': self.learning_rate,
        #     'decay': self.decay,
        #     'V' : self.V,
        #     'W' : self.W,
        #     'b1': self.b1,
        #     'b2' :self.b2,
        #     'v_weight':self.V_regularWeight,
        #     'w_weight':self.W_regularWeight,
        #     'history': history
        #     }, savePath)
        print("save model : " + ModelName)
        
        with open('./' + dataset + r'/History/' + ModelName + '.his', 'wb') as fs:
            pickle.dump(history, fs)

    def getModelName(self):
        ModelName = "autorec_" + dataset + modelUTCStr + \
        "_CV" + str(cv) + \
        "_V-Weight_" + str(self.V_regularWeight) + \
        "_W-Weight_" + str(self.W_regularWeight) + \
        "_BATCH_" + str(BATCH_SIZE) + \
        "_LATENT_DIM" + str(LATENT_DIM) + \
        "_LR" + str(LR) + \
        "_LR_DACAY" + str(self.LR_decay)
        return ModelName
        
    def writeResult(self, result):
        with open(resultFile, mode='a') as f:
            modelName = self.getModelName()
            f.write('\r\n')
            f.write(dataset + '\r\n')
            f.write(modelName + '\r\n')
            f.write(str(result) + '\r\n')

    def RMSE(self, decoder, label, mask): #预测数据和输入数据的square error
            add_avg = decoder + 3 - 3 * (decoder != 0) #没有评分的统一给3分
            return np.sum(np.square((add_avg - label) * mask))
#####################################################################################
if __name__=='__main__':
    params = sys.argv
    if len(params) == 2:
        cv = params[1]
    else:
        cv = 1
    print(params)
    #数据已经划分好了，直接加载拿来用,90%训练，10%调参（但都是压缩存储格式）
    with open('bak_1_0.821\sparseMat_0.9_train.csv', 'rb') as f:
        trainMat = pickle.load(f)
    with open('bak_1_0.821\sparseMat_0.9_test.csv', 'rb') as f:
        testMat = pickle.load(f)
    hope=autoEncoder(trainMat,testMat,LR,LR_DECAY,V_regularWeight,W_regularWeight,is_loadModel)
    modelName=hope.getModelName()
    print('model name='+modelName)
    hope.run()