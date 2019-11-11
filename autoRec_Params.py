import time

modelUTCStr=str(int(time.time()))

dataset = 'ml-1m'
USER_NUM=6040
MOVIE_NUM=3706
MOVIE_BASED=True
is_loadModel=False
BATCH_SIZE=8
EPOCH=130
LATENT_DIM=500

#需要调的超参
LR=0.001
LR_DECAY=0.97
V_regularWeight=0.05
W_regularWeight=0.05
