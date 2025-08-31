mode=$1
run_name=$2
DEVICE=$3

export CUDA_VISIBLE_DEVICES=${DEVICE}
python train.py \
  --run_name "${run_name}" \
  --mode "${mode}" \
  --batch_size 64 \
  --num_classes 10 \
  --imb_ratio 200
