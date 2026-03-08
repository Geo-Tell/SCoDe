#python3 -m torch.distributed.launch --nproc_per_node=2 train.py --config_path configs/baseline/oetr_config.py --validation --epoch 35 --batch_size 4
# CUDA_VISIBLE_DEVICES=2,3 nohup python -m torch.distributed.launch \
# --nproc_per_node 2 train_scode.py --num_workers 0 --validation --epoch 40 --batch_size 2 > outputs/nohups/SCoDe_1216.log 2>&1 &

# python -m torch.distributed.launch \
# --nproc_per_node 4 \
# train_scode.py \
# --num_workers 0 \
# --validation \
# --epoch 40 \
# --batch_size 2


# echo 'for 1024x1024'
# nohup python -m torch.distributed.launch --nproc_per_node 4 --master_port=29501 train_scode.py --num_workers 0 --epoch 15 --batch_size 4 --validation --learning_rate 5e-4 > outputs/nohups/SCoDe_1024_pt3.log 2>&1 &

echo 'post-training'
nohup python -m torch.distributed.launch --nproc_per_node 4 --master_port=29501 train_scode.py --num_workers 4 --epoch 15 --batch_size 4 --validation --learning_rate 1e-5 > outputs/nohups/SCoDe_1024_pt4.log 2>&1 &
# python -m torch.distributed.launch --nproc_per_node 4 --master_port=29500 train_scode.py --num_workers 4 --epoch 15 --batch_size 4 --validation --learning_rate 1e-5
