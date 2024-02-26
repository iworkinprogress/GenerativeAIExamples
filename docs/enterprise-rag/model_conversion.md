# Generating model repositories for LLM models

This guide explains the steps developers need to follow in order to convert huggingface checkpoints to TRT-LLM format which can be consumed and accelerated using [Nemo Inference Microservice](https://registry.ngc.nvidia.com/orgs/ohlfw0olaadg/teams/ea-participants/containers/nemollm-inference-ms). You can skip these steps, if you are using an A100 GPU based system. You can use the [prebuilt model repositories available in NGC to deploy the pipeline.](../../RetrievalAugmentedGeneration/README.md#getting-started).

## llama-2 chat models
1. Download llama2 model from [huggingface](https://huggingface.co/meta-llama/Llama-2-13b-chat-hf).

2. Create a config file named `model_config.yaml` with the content below in working directory
```
model_repo_path: "/model-store/"
use_ensemble: false
model_type: "LLAMA"
backend: "trt_llm"
base_model_id: "ensemble"
prompt_timer: 60
gateway_ip: "gateway-api"
server_port_internal: 9009
path_to_fastertransformer_ini_config:
customization_cache_capacity: 10000
logging_level: "INFO"
enable_chat: true
preprocessor:
  enable_customization_fetching: true
  chat_cfg:
    roles:
      system:
        prefix: "[INST] <<SYS>>\n"
        suffix: "\n<</SYS>>\n\n"
      user:
        prefix: ""
        suffix: " [/INST] "
      assistant:
        prefix: ""
        suffix: " </s><s>[INST] "
    stop_words: ["</s>", "<s>", "[INST]", "[/INST]", "<<SYS>>", "<</SYS>>"]
    rstrip_turn: true
    turn_suffix: "\n"
pipeline:
  model_name: "ensemble"
  num_instances: 1
trt_llm:
  use: true
  ckpt_type: "hf"
  model_name: "trt_llm"
  backend: "python"
  num_gpus: 1
  model_path: /engine_dir
  max_queue_delay_microseconds: 10000
  model_type: "llama"
  max_batch_size: 8
  max_input_len: 4096
  max_output_len: 4096
  max_beam_width: 1
  tensor_para_size: 1
  pipeline_para_size: 1
  data_type: "float16"
  int8_mode: 0
  enable_custom_all_reduce: 0
  per_column_scaling: false
```

3. Create a `model-store` directory keep it empty. Engine files will be generated here. You can create it in pwd.
    ```
    mkdir model-store
    ```
4. Run the conversion script to generate engine
    ```
    docker run --rm -ti --gpus=1 -v $PWD/model-store:/model-store -v $PWD/Llama-2-13b-chat-hf:/engine_dir -v $PWD/model_config.yaml:/model_config.yaml nvcr.io/ohlfw0olaadg/ea-participants/nemollm-inference-ms:24.01 bash -c "model_repo_generator llm --verbose --yaml_config_file=/model_config.yaml"
    ```

    Note: If you've downloaded model to any other path change `$PWD/Llama-2-13b-chat-hf` to absolute path of your hf model repo.

5. After the model-store directory is generated. You need to update `compose.env` with absolute path of `model-store` directory e.g given below
    ```
    # This is example value, you'll need to populate absolute path of your `model-store` directory
    export MODEL_DIRECTORY="/home/nvidia/model-store"
    ```

6. After you generate the model repository, [you can now follow these steps to deploy the end to end pipeline.](../../RetrievalAugmentedGeneration/README.md#install-guide)

## Deploying other model architectures
For deploying other models like llama completion you will need to follow similar steps as the llama-chat model. Detailed instruction for deploying other models can be found by following these steps:
1. Download the required model like [llama-7b](https://huggingface.co/meta-llama/Llama-2-7b) completion model checkpoint from huggingface.

2. Once you have the checkpoints downloaded, follow these steps from [Nemo Inference Microservice](https://developer.nvidia.com/docs/nemo-microservices/model-repo-generator.html) to access required documentation:


3. After you generate the model repository, [You can now follow these steps to deploy the end to end pipeline.](../../RetrievalAugmentedGeneration/README.md#install-guide)

**Note:**

    Nemotron based checkpoints in huggingface are not supported in this flow. It can be deloyed only in A100 based systems using prebuilt model repositories in NGC.