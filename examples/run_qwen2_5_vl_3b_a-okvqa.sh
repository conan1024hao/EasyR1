set -x

export VLLM_ATTENTION_BACKEND=XFORMERS
export WANDB_RESUME="allow"
export WANDB_API_KEY="5b8322df11b04a8895325a5bf6ef8cbba0dd64a2"
export WANDB_ENTITY=conan1024hao

export NCCL_P2P_DISABLE=1
export CUDA_VISIBLE_DEVICES=1,2,3,4
MODEL_PATH=Qwen/Qwen2.5-VL-3B-Instruct

# python3 -m verl.trainer.main \
#     config=examples/grpo_a-okvqa.yaml \
#     data.train_files=Amasia-MLLM/A-OKVQA@train \
#     data.val_files=Amasia-MLLM/KnowRecall@zh \
#     worker.actor.model.model_path=${MODEL_PATH} \
#     worker.rollout.tensor_parallel_size=4 \
#     worker.rollout.enable_chunked_prefill=false \
#     trainer.experiment_name=qwen2_5_vl_3b_a-okvqa \
#     trainer.n_gpus_per_node=4

# for debug
python3 -m verl.trainer.main \
    config=examples/grpo_a-okvqa.yaml \
    data.train_files=Amasia-MLLM/A-OKVQA@train \
    data.val_files=Amasia-MLLM/A-OKVQA@train \
    data.rollout_batch_size=16 \
    worker.actor.model.model_path=${MODEL_PATH} \
    worker.actor.global_batch_size=4 \
    worker.rollout.tensor_parallel_size=4 \
    worker.rollout.enable_chunked_prefill=false \
    worker.rollout.n=2 \
    trainer.experiment_name=qwen2_5_vl_3b_a-okvqa \
    trainer.n_gpus_per_node=4
