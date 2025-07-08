CUDA_VISIBLE_DEVICES=0 pyth1on train.py --cuda -d  dataset \
    -n 128 --lambda 0.025 --epochs 150 --lr_epoch 200 240 --batch-size 6 \
    --save_path checkpoint/kodak --save --train-dataname flickr30k  --test-dataname Kodak \
