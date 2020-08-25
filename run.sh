# data process

python data_process.py \
--dataset 'imdb' \
--data_path $DATA_DIR \
--embd_path 'imdb' \
--similarity_threshold 0.8 \
--perturbation_constraint 100 \

# training

python train.py \
--model_name_or_path bert-base-uncased \
--task_name $TASK_NAME \
--do_train \
--data_dir $DATA_DIR \
--output_dir ./cygong_amazon_textcnn_output

# evaluation

python evaluate.py \
--similarity_threshold 0.8 \
--perturbation_constraint 100 \
--skip 20 \
--num_random_sample 100000 \
--mc_error 0.01 \
--task_name imdb \
--do_lower_case \
--data_dir imdb/ \
--result_dir imdb/result \
--model_type bert \
--model_name_or_path bert-base-uncased \
--max_seq_length 256 \
--checkpoint_dir imdb_bert_checkpoint \
--overwrite_output_dir

