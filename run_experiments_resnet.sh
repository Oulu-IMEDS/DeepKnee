#!/usr/bin/env bash


# Activating the environment
#source /media/lext/FAST/pytorch_3_6_env/bin/activate

# Training parameters
export DROPOUT=0.5
export WD=1e-4
export BATCH_SIZE=64
export BATCH_SIZE_VAL=64
export BASE_WIDTH=64
export LR=1e-3

export LOGS_FLD="training_logs"
export N_EPOCH=100
export N_BATCHES=300
export BOOTRSTRAP=15


mkdir -p $LOGS_FLD
cd resnet_codes


export SEED=42
export SUFFIX="${SEED}_${BATCH_SIZE}_resnet"
python -u train.py --use_visdom True --experiment "[ResNet-34]" --n_epoch $N_EPOCH --drop $DROPOUT --wd $WD --lr $LR --lr_drop 101 --bs $BATCH_SIZE  --val_bs $BATCH_SIZE_VAL --lr_min 1e-6 --base_width $BASE_WIDTH --seed $SEED  --n_batches $N_BATCHES --bootstrap $BOOTRSTRAP #> ../$LOGS_FLD/$SUFFIX.txt 2>&1 
echo "Started training..."
sleep 3s

