DEVICE=0
DATASET=mnist
TARGET=blank
SEEDS="0"
EPS=10.0
LR=0.05

CUDA_VISIBLE_DEVICES=$DEVICE python3 audit.py --dataset $DATASET --target $TARGET --seeds $SEEDS --eps $EPS --lr $LR