base_dir="./eval_dir_4_5"
mkdir -p ${base_dir}

CUDA_VISIBLE_DEVICES=0,1,2,3 \
torchrun  \
    --standalone    \
    --nnodes=1     \
    --nproc_per_node=4 \
./test.py \
    --model MyModel \
    --world_size 1 \
    --test_data_json /mnt/data0/xiaochen/workspace/IMDLBenCo_pure/test_metrics/test_datasets.json \
    --checkpoint_path "./output_dir/" \
    --test_batch_size 5 \
    --image_size 512 \
    --if_resizing \
    --output_dir ${base_dir}/ \
    --log_dir ${base_dir}/ \
2> ${base_dir}/error.log 1>${base_dir}/logs.log