set -x

export VLLM_ATTENTION_BACKEND=XFORMERS
export NCCL_P2P_DISABLE=1
export WANDB_RESUME="allow"
export WANDB_API_KEY="5b8322df11b04a8895325a5bf6ef8cbba0dd64a2"
export WANDB_ENTITY=conan1024hao

MODEL_PATH=Qwen/Qwen2.5-VL-3B-Instruct
RUN_NAME=qwen2_5_vl_3b_a-okvqa_he

python3 -m verl.trainer.main \
    config=examples/grpo_a-okvqa.yaml \
    data.train_files=Amasia-MLLM/A-OKVQA@train \
    data.val_files=Amasia-MLLM/KnowRecall@he \
    worker.actor.model.model_path=${MODEL_PATH} \
    worker.rollout.tensor_parallel_size=1 \
    worker.rollout.gpu_memory_utilization=0.3 \
    worker.rollout.enable_chunked_prefill=false \
    trainer.experiment_name=${RUN_NAME} \
    trainer.n_gpus_per_node=8

# for debug
# python3 -m verl.trainer.main \
#     config=examples/grpo_a-okvqa.yaml \
#     data.train_files=Amasia-MLLM/A-OKVQA@debug \
#     data.val_files=Amasia-MLLM/KnowRecall@he \
#     data.rollout_batch_size=8 \
#     worker.actor.global_batch_size=8 \
#     worker.actor.model.model_path=${MODEL_PATH} \
#     worker.rollout.tensor_parallel_size=1 \
#     worker.rollout.enable_chunked_prefill=false \
#     trainer.experiment_name=${RUN_NAME} \
#     trainer.n_gpus_per_node=8
