#!/bin/bash

# make the script stop when error (non-true exit code) is occured
set -e

if [ $1 == 'computers' ] && [ $2 == 'GCN' ];
then
    DATA='computers'
    NET='GCN'
    N_LAYER=2
    FEAT_DIM=256
    MSize=3
    MDROP=0.5
    EMB='SSGC'
    DEPTH=16
    ALPHA=0.20
    S_SCALE=2.0
    U_WARM=0
    U_DECAY=1000
    U_SCALE=1.0
    KM_DEPTH=2
    KM_ALPHA=0.15
    KM_METRIC='euclidean'
    KM_METHOD='pam'
elif [ $1 == 'computers' ] && [ $2 == 'GAT' ];
then
    DATA='computers'
    NET='GAT'
    N_LAYER=2
    FEAT_DIM=256
    MSize=3
    MDROP=0.5
    EMB='SSGC'
    DEPTH=16
    ALPHA=0.20
    S_SCALE=0.5
    U_WARM=0
    U_DECAY=1000
    U_SCALE=0.5
    KM_DEPTH=2
    KM_ALPHA=0.05
    KM_METRIC='euclidean'
    KM_METHOD='pam'
elif [ $1 == 'computers' ] && [ $2 == 'SAGE' ];
then
    DATA='computers'
    NET='SAGE'
    N_LAYER=2
    FEAT_DIM=256
    MSize=6
    MDROP=0.5
    EMB='APPNP'
    DEPTH=2
    ALPHA=0.05
    S_SCALE=0.5
    U_WARM=0
    U_DECAY=500
    U_SCALE=1.0
    KM_DEPTH=4
    KM_ALPHA=0.10
    KM_METRIC='manhattan'
    KM_METHOD='pam'
elif [ $1 == 'photo' ] && [ $2 == 'GCN' ];
then
    DATA='photo'
    NET='GCN'
    N_LAYER=2
    FEAT_DIM=256
    MSize=2
    MDROP=0.5
    EMB='SSGC'
    DEPTH=4
    ALPHA=0.05
    S_SCALE=0.5
    U_WARM=0
    U_DECAY=1000
    U_SCALE=1.0
    KM_DEPTH=4
    KM_ALPHA=0.10
    KM_METRIC='manhattan'
    KM_METHOD='pam'
elif [ $1 == 'photo' ] && [ $2 == 'GAT' ];
then
    DATA='photo'
    NET='GAT'
    N_LAYER=2
    FEAT_DIM=256
    MSize=4
    MDROP=0.5
    EMB='APPNP'
    DEPTH=2
    ALPHA=0.20
    S_SCALE=0.5
    U_WARM=0
    U_DECAY=500
    U_SCALE=0.5
    KM_DEPTH=4
    KM_ALPHA=0.15
    KM_METRIC='manhattan'
    KM_METHOD='pam'
elif [ $1 == 'photo' ] && [ $2 == 'SAGE' ];
then
    DATA='photo'
    NET='SAGE'
    N_LAYER=2
    FEAT_DIM=256
    MSize=4
    MDROP=0.0
    EMB='SSGC'
    DEPTH=8
    ALPHA=0.05
    S_SCALE=1.0
    U_SCALE=1.0
    U_DECAY=1000
    KM_DEPTH=8
    KM_ALPHA=0.15
    KM_METRIC='manhattan'
elif [ $1 == 'Cora' ] && [ $2 == 'GCN' ];
then
    DATA='Cora'
    NET='GCN'
    N_LAYER=2
    FEAT_DIM=256
    MSize=5
    MDROP=0.5
    EMB='APPNP'
    DEPTH=8
    ALPHA=0.10
    S_SCALE=0.5
    U_WARM=0
    U_DECAY=1000
    U_SCALE=0.5
    KM_DEPTH=4
    KM_ALPHA=0.10
    KM_METRIC='manhattan'
    KM_METHOD='pam'
elif [ $1 == 'Cora' ] && [ $2 == 'GAT' ];
then
    DATA='Cora'
    NET='GAT'
    N_LAYER=2
    FEAT_DIM=256
    MSize=4
    MDROP=0.5
    EMB='SSGC'
    DEPTH=4
    ALPHA=0.05
    S_SCALE=1.0
    U_WARM=0
    U_DECAY=1000
    U_SCALE=1.0
    KM_DEPTH=4
    KM_ALPHA=0.05
    KM_METRIC='manhattan'
    KM_METHOD='pam'
elif [ $1 == 'Cora' ] && [ $2 == 'SAGE' ];
then
    DATA='Cora'
    NET='SAGE'
    N_LAYER=2
    FEAT_DIM=256
    MSize=3
    MDROP=0.5
    EMB='SSGC'
    DEPTH=16
    ALPHA=0.2
    S_SCALE=4.0
    U_WARM=0
    U_DECAY=500
    U_SCALE=2.0
    KM_DEPTH=12
    KM_ALPHA=0.10
    KM_METRIC='manhattan'
    KM_METHOD='pam'
elif [ $1 == 'CiteSeer' ] && [ $2 == 'GCN' ];
then
    DATA='CiteSeer'
    NET='GCN'
    N_LAYER=2
    FEAT_DIM=256
    MSize=4
    MDROP=0.5
    EMB='APPNP'
    DEPTH=4
    ALPHA=0.05
    S_SCALE=1.0
    U_WARM=0
    U_DECAY=2000
    U_SCALE=1.0
    KM_DEPTH=8
    KM_ALPHA=0.2
    KM_METRIC='manhattan'
    KM_METHOD='pam'
elif [ $1 == 'CiteSeer' ] && [ $2 == 'GAT' ];
then
    DATA='CiteSeer'
    NET='GAT'
    N_LAYER=2
    FEAT_DIM=256
    MSize=4
    MDROP=0.5
    EMB='APPNP'
    DEPTH=8
    ALPHA=0.05
    S_SCALE=0.5
    U_WARM=0
    U_DECAY=1000
    U_SCALE=1.0
    KM_DEPTH=12
    KM_ALPHA=0.10
    KM_METRIC='euclidean'
    KM_METHOD='pam'
elif [ $1 == 'CiteSeer' ] && [ $2 == 'SAGE' ];
then
    DATA='CiteSeer'
    NET='SAGE'
    N_LAYER=2
    FEAT_DIM=256
    MSize=4
    MDROP=0.0
    EMB='APPNP'
    DEPTH=2
    ALPHA=0.15
    S_SCALE=1.0
    U_WARM=0
    U_DECAY=1000
    U_SCALE=0.5
    KM_DEPTH=16
    KM_ALPHA=0.15
    KM_METRIC='manhattan'
    KM_METHOD='pam'
elif [ $1 == 'PubMed' ] && [ $2 == 'GCN' ];
then
    DATA='PubMed'
    NET='GCN'
    N_LAYER=2
    FEAT_DIM=256
    MSize=15
    MDROP=0.5
    EMB='SSGC'
    DEPTH=4
    ALPHA=0.15
    S_SCALE=0.5
    U_WARM=0
    U_DECAY=2000
    U_SCALE=0.5
    KM_DEPTH=16
    KM_ALPHA=0.10
    KM_METRIC='euclidean'
    KM_METHOD='pam'
elif [ $1 == 'PubMed' ] && [ $2 == 'GAT' ];
then
    DATA='PubMed'
    NET='GAT'
    N_LAYER=2
    FEAT_DIM=256
    MSize=10
    MDROP=0.5
    EMB='APPNP'
    DEPTH=8
    ALPHA=0.20
    S_SCALE=0.5
    U_WARM=0
    U_DECAY=2000
    U_SCALE=1.0
    KM_DEPTH=4
    KM_ALPHA=0.10
    KM_METRIC='manhattan'
    KM_METHOD='pam'
elif [ $1 == 'PubMed' ] && [ $2 == 'SAGE' ];
then
    DATA='PubMed'
    NET='SAGE'
    N_LAYER=2
    FEAT_DIM=256
    MSize=15
    MDROP=0.5
    EMB='APPNP'
    DEPTH=4
    ALPHA=0.15
    S_SCALE=0.5
    U_WARM=0
    U_DECAY=1500
    U_SCALE=1.0
    KM_DEPTH=12
    KM_ALPHA=0.05
    KM_METRIC='euclidean'
    KM_METHOD='pam'
else
    echo 'ERROR'
fi

python main.py --dataset $DATA \
            --net $NET \
            --n-layer $N_LAYER \
            --feat-dim $FEAT_DIM \
            --imb-ratio $3 \
            --running-time $4 \
            --meta-size $MSize \
            --context-emb $EMB \
            --depth $DEPTH \
            --alpha $ALPHA \
            --meta-drop $MDROP \
            --syn-scale $S_SCALE \
            --ul-warmup $U_WARM \
            --ul-decay-epoch $U_DECAY \
            --ul-scale $U_SCALE \
            --km-depth $KM_DEPTH \
            --km-alpha $KM_ALPHA \
            --km-metric $KM_METRIC \
            --km-method $KM_METHOD \
            --device $5
