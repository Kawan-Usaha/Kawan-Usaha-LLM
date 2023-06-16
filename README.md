# Kawan-Usaha-LLM
This AI model serves as the foundation for our chatbot and article generation. It is built upon the LLAMA-13b base model and enhanced with the Vicuna-13b-v1.1 delta weight. Furthermore, the architecture of the model has been modified by replacing the activation function in the last layer of the Multi Layer Perceptron from SiLU to GELU.

## Server Inference Specification
To cater to the resource demands of our project, we have leveraged a Virtual Machine (VM) equipped with 24 virtual CPUs, 96 GB of RAM, and 2 x Nvidia L4 GPUs. Among these specifications, the presence of 2 x Nvidia L4 GPUs is of utmost significance. This is primarily due to the fact that the Language Model (LLM) we are working with necessitates a substantial GPU memory allocation, specifically 28 GB.

By capitalizing on the computational capabilities of this hardware configuration, we can ensure smooth execution of complex tasks and achieve optimal performance throughout our project. The ample GPU memory provided by the Nvidia L4 GPUs significantly contributes to the model's capacity to process large-scale language data, enabling us to attain accurate and reliable results.

## Update regarding the LLM's repository
The repository for the LLM can be accessed at the following location: https://huggingface.co/KawanUsaha/Kawan-Usaha-13b/tree/main. We have chosen to host it on HuggingFace due to the platform's ability to accommodate unlimited file sizes for storage purposes. Note: "Gopalatius" is **M181DSX1496**

## What the ML division did
Throughout the course of this project, we have encountered numerous trial and error scenarios. Nevertheless, in hindsight, we can now outline the step-by-step approach we would have taken, armed with our present knowledge.

### 1. Clone the LLAMA-13b repository
To incorporate the foundational LLAMA-13b model into our project, it is crucial to clone its repository. Follow these steps to initiate the cloning process:

1. Ensure that Git LFS (Large File Storage) is installed:
```bash
git lfs install
```

2. Proceed with cloning the repository using the following command:
```bash
git clone https://huggingface.co/huggyllama/llama-13b
```

Please note that the cloning process may take a significant amount of time due to the repository's size, which exceeds 100 GBs.
## 2. Install Fastchat
FastChat is a comprehensive open platform designed to facilitate the training, deployment, and assessment of chatbots that are built upon large language models. Key highlights of this platform include:

- Access to cutting-edge models such as Vicuna and FastChat-T5, along with their respective weights, training code, and evaluation code.
- A distributed multi-model serving system that boasts a user-friendly web interface and RESTful APIs, ensuring compatibility with OpenAI standards.

To install FastChat, please follow the step-by-step instructions below:

1. Begin by cloning the FastChat repository from GitHub using the following command:
```bash
git clone https://github.com/lm-sys/FastChat.git
```

2. Move into the cloned FastChat directory:
```bash
cd FastChat
```

3. Next, ensure that your pip package manager is up to date by executing the following command:
```bash
pip3 install --upgrade pip
```
This step is necessary to enable support for PEP 660.

4. Finally, install FastChat by running the following command:
```bash
pip3 install -e .
```
This command will install FastChat in editable mode, allowing you to make modifications if needed. It basically install all the library that is needed, including Pytorch, Huggingface's Transformer, etc.

By following these instructions, you will successfully install FastChat on your system.

### 3. Apply the Vicuna-13b weight
The Vicuna-13b model demonstrates superior performance in terms of GPT-4 evaluation scores compared to the standard LLAMA-13b models. Vicuna-13b is an optimized version derived from the LLAMA-13b models, incorporating specific weight modifications to enhance its capabilities.

Please note that the conversion process necessitates approximately 60 GB of CPU RAM. To execute the conversion, utilize the following command:

```bash!
python3 -m fastchat.model.apply_delta \
    --base-model-path /path/to/llama-13b \
    --target-model-path /path/to/output/vicuna-13b \
    --delta-path lmsys/vicuna-13b-delta-v1.1
```

It is important to highlight that there is no requirement to manually clone the delta-path, as it will be automatically downloaded by executing the provided command.

### 4. Customizing the Model Architecture

Customizing the model architecture is a crucial step to ensure compliance with Bangkit's guideline of not doing plug and play using prebuilt models.  Prior to proceeding with this customization, we sought permission from the Capstone Admin by sending an email outlining our intentions. We received approval to base our capstone project on PyTorch (given the involvement of transformers) and restrict our modifications solely to the model architecture.

You can find the Python code required for editing the model architecture [here](https://huggingface.co/KawanUsaha/Kawan-Usaha-13b/blob/main/old-model/edit_model.py). This code serves as a reference for making the necessary adjustments to the model architecture as per the approved guidelines.
```python!
# import io
from torch.nn import GELU
from transformers import AutoModelForCausalLM
import torch

# Load the model from the buffer
# Use the model that has been applied with vicuna-13b delta weight
print('LLM models before:')
model = AutoModelForCausalLM.from_pretrained('/mnt/old-model', device_map="auto",torch_dtype=torch.float16) 
print(model)
# Access the 39th layer of the model
layer = model.model.layers[39]

# Access the llamaMLP of the layer
mlp = layer.mlp

# Replace the act_fn with GELU
mlp.act_fn = GELU()

print('LLM models after:')
print(model)
model.save_pretrained('/mnt/kawan-usaha-13b')
```
The command is
```bash!
sudo python3 edit_model.py
```
Here is the result
```bash!
[m181dsx1496@kawan-usaha-llm-lowpower old-model]$ sudo python3 edit_model.py
Loading checkpoint shards: 100%|██████████████████| 3/3 [01:38<00:00, 32.86s/it]
LLM models before:
LlamaForCausalLM(
  (model): LlamaModel(
    (embed_tokens): Embedding(32000, 5120, padding_idx=0)
    (layers): ModuleList(
      (0-39): 40 x LlamaDecoderLayer(
        (self_attn): LlamaAttention(
          (q_proj): Linear(in_features=5120, out_features=5120, bias=False)
          (k_proj): Linear(in_features=5120, out_features=5120, bias=False)
          (v_proj): Linear(in_features=5120, out_features=5120, bias=False)
          (o_proj): Linear(in_features=5120, out_features=5120, bias=False)
          (rotary_emb): LlamaRotaryEmbedding()
        )
        (mlp): LlamaMLP(
          (gate_proj): Linear(in_features=5120, out_features=13824, bias=False)
          (down_proj): Linear(in_features=13824, out_features=5120, bias=False)
          (up_proj): Linear(in_features=5120, out_features=13824, bias=False)
          (act_fn): SiLUActivation()
        )
        (input_layernorm): LlamaRMSNorm()
        (post_attention_layernorm): LlamaRMSNorm()
      )
    )
    (norm): LlamaRMSNorm()
  )
  (lm_head): Linear(in_features=5120, out_features=32000, bias=False)
)
LLM models after:
LlamaForCausalLM(
  (model): LlamaModel(
    (embed_tokens): Embedding(32000, 5120, padding_idx=0)
    (layers): ModuleList(
      (0-38): 39 x LlamaDecoderLayer(
        (self_attn): LlamaAttention(
          (q_proj): Linear(in_features=5120, out_features=5120, bias=False)
          (k_proj): Linear(in_features=5120, out_features=5120, bias=False)
          (v_proj): Linear(in_features=5120, out_features=5120, bias=False)
          (o_proj): Linear(in_features=5120, out_features=5120, bias=False)
          (rotary_emb): LlamaRotaryEmbedding()
        )
        (mlp): LlamaMLP(
          (gate_proj): Linear(in_features=5120, out_features=13824, bias=False)
          (down_proj): Linear(in_features=13824, out_features=5120, bias=False)
          (up_proj): Linear(in_features=5120, out_features=13824, bias=False)
          (act_fn): SiLUActivation()
        )
        (input_layernorm): LlamaRMSNorm()
        (post_attention_layernorm): LlamaRMSNorm()
      )
      (39): LlamaDecoderLayer(
        (self_attn): LlamaAttention(
          (q_proj): Linear(in_features=5120, out_features=5120, bias=False)
          (k_proj): Linear(in_features=5120, out_features=5120, bias=False)
          (v_proj): Linear(in_features=5120, out_features=5120, bias=False)
          (o_proj): Linear(in_features=5120, out_features=5120, bias=False)
          (rotary_emb): LlamaRotaryEmbedding()
        )
        (mlp): LlamaMLP(
          (gate_proj): Linear(in_features=5120, out_features=13824, bias=False)
          (down_proj): Linear(in_features=13824, out_features=5120, bias=False)
          (up_proj): Linear(in_features=5120, out_features=13824, bias=False)
          (act_fn): GELU(approximate='none')
        )
        (input_layernorm): LlamaRMSNorm()
        (post_attention_layernorm): LlamaRMSNorm()
      )
    )
    (norm): LlamaRMSNorm()
  )
  (lm_head): Linear(in_features=5120, out_features=32000, bias=False)
)
[m181dsx1496@kawan-usaha-llm-lowpower old-model]$
```
![](https://hackmd.io/_uploads/HkjLIpFPh.png)
![](https://hackmd.io/_uploads/HJ4DIaKvn.png)

### 5. Launching the Model for Inference

FastChat provides comprehensive guidelines for performing Large Language Model (LLM) inference using FastAPI and OpenAI API templates. To initiate the inference process, follow the instructions below:

1. Open three separate terminal windows.
2. Change the directory to the FastChat directory:
```bash
cd FastChat
```
3. Execute the following commands sequentially in each terminal window:

**Terminal 1**:
```bash
sudo python3 -m fastchat.serve.controller
```

**Terminal 2**:
```bash
sudo python3 -m fastchat.serve.model_worker --model-name 'Kawan-Usaha' --model-path /mnt/kawan-usaha-13b --num-gpus 1 --load-8bit --cpu-offloading --controller http://localhost:21001 --port 31000 --worker http://localhost:31000
```
Please note that it is observed, although the exact reason is unclear, that specifying `--num-gpus 1` inadvertently creates two processes on each of our two GPUs. This behavior is potentially attributed to memory load balancing. Furthermore, the configuration supports 8-bit quantization and CPU offloading, enhancing the efficiency and performance of the model worker.

**Terminal 3**:
```bash
sudo python3 -m fastchat.serve.openai_api_server --host 0.0.0.0 --port 8000
```

By executing these commands, you will launch the FastChat model for inference. Terminal 1 starts the controller, Terminal 2 initiates the model worker with specified configurations, and Terminal 3 runs the OpenAI API server with a specified host and port.

Ensure that you follow these instructions precisely to establish the necessary environment for performing LLM inference using FastChat.

## Additional Notes

### No Fine-tuning (training) on the LLM?

1. In our research, we encountered significant challenges when attempting to train a 13 billion parameter Language Model (LLM). Despite implementing the Low Rank Adapter technique to optimize GPU memory usage, we found that the computational demands were still excessively high. Consequently, the current allocation of Google Cloud Platform (GCP) credits was projected to be insufficient to support the training process. As a result, we reached out to Bangkit, explaining our predicament and requesting additional GCP credits. Fortunately, they graciously granted our request, enabling us to proceed with the project, even without retraining the model.
2. To ensure that our modifications align with the project guidelines, we consulted the Capstone Admin. We proposed changing the model architecture while refraining from fine-tuning the LLM. The Admin approved our proposed modification, allowing us to proceed with the alternative approach.
3. To facilitate the training process, we meticulously prepared a well-structured dataset, which can be accessed through the following link: [dataset link](https://huggingface.co/KawanUsaha/Kawan-Usaha-13b/blob/main/old-model/conversation.json). For those interested in replicating the training, we have provided the necessary command in the [deepspeed_config.json](https://huggingface.co/KawanUsaha/Kawan-Usaha-13b/blob/main/old-model/training%20lora.bash) file within our repository. Executing this command will initiate the training process, utilizing the appropriate configurations for the Low Rank Adapter technique.
```bash!
sudo deepspeed --num_nodes 1 --num_gpus 2 fastchat/train/train_lora.py --deepspeed /mnt/vicuna-13b/deepspeed_config.json \
    --model_name_or_path /path/to/vicuna-13b \
    --data_path /path/to/dataset.json \
    --output_dir /path/to/trained-model \
    --num_train_epochs 3 \
	--bf16 True \
    --per_device_train_batch_size 2 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 1200 \
    --save_total_limit 10 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --model_max_length 2048 \
    --gradient_checkpointing False \
    --lazy_preprocess True \
    --gradient_accumulation_steps 16 \
    --learning_rate 2e-05 \
```
We tried fine-tuning using the current compute instance. However, the memory is still not enough and we encounter ```CUDA out of memory``` error, even after implementing LoRa.
Note:
* When employing distributed computing methods, such as CPU offloading and training with multiple GPUs, it is possible to encounter an error related to tensors being located on different devices. To address this issue, it becomes necessary to make modifications to both the HuggingFace's Transformer library and the FastChat training library.
* To ensure seamless tensor calculations, a crucial step involves moving the tensors to the same device before performing any computations. By implementing this modification, we can avoid the "tensors on different devices" error and ensure consistent and efficient processing across the distributed system.
* The modifications required for achieving tensor device consistency involve adapting the codebase of both the HuggingFace's Transformer library and the FastChat training library. By applying these changes, we can enhance the interoperability and compatibility of these libraries within a distributed computing environment.
* It is worth noting that making these modifications necessitates a deep understanding of the underlying frameworks and libraries involved, as well as the intricacies of distributed computing. Careful consideration and rigorous testing should be carried out to ensure the correctness and stability of the modified libraries, thus enabling smooth tensor operations and optimal performance in distributed training scenarios.

4. Given that our capstone project has secured funding as the top-ranking project in Bangkit 2023, we are now empowered to undertake the fine-tuning process for our model. With this financial support, we can confidently invest resources into training the model using extensive datasets specific to Small and Medium Enterprises (SMEs) in Indonesia. 

    While the current iteration of the model has been trained on general conversational data, it still demonstrates an impressive level of quality, achieving approximately 90% of the performance seen in ChatGPT (changing SiLU to GELU don't have significant impact on the performance. we tested it). This indicates that our model possesses a strong foundation and the potential to excel further with targeted fine-tuning.

    By utilizing [specialized datasets](https://huggingface.co/KawanUsaha/Kawan-Usaha-13b/blob/main/old-model/conversation.json) focused on Indonesian SMEs, we aim to enhance the model's proficiency in addressing the unique challenges and intricacies of this domain. This fine-tuning process will allow us to adapt the model to the specific requirements and nuances of SME-related conversations, enabling it to provide more accurate and relevant responses to users.

    Through this fine-tuning endeavor, we anticipate that our model will surpass its current performance and achieve a higher degree of accuracy and contextual understanding when engaging with SME-related queries. This, in turn, will enhance the overall user experience and value proposition of our capstone project.
