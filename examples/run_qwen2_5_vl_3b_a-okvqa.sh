set -x

export VLLM_ATTENTION_BACKEND=XFORMERS
export CUDA_VISIBLE_DEVICES=0,1

MODEL_PATH=Qwen/Qwen2.5-VL-3B-Instruct

python3 -m verl.trainer.main \
    config=examples/grpo_a-okvqa.yaml \
    data.train_files=HuggingFaceM4/A-OKVQAk@train \
    data.val_files=Amasia-MLLM/KnowRecall@zh \
    worker.actor.model.model_path=${MODEL_PATH} \
    worker.rollout.tensor_parallel_size=1 \
    worker.rollout.enable_chunked_prefill=false \
    trainer.experiment_name=qwen2_5_vl_3b_a-okvqa \
    trainer.n_gpus_per_node=2
