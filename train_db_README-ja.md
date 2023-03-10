这是DreamBooth的指南。

请同时查看[有关学习的共同文档](./train_README-ja.md)。

# 概要

DreamBooth是一种技术，可以将特定主题添加到图像生成模型中进行额外学习，并使用特定的标识符在生成图像中呈现出来。[论文在此](https://arxiv.org/abs/2208.12242)。

具体来说，可以使用Stable Diffusion模型来学习角色和绘画风格等内容，并使用类似于“shs”这样的特定单词来调用（呈现在生成的图像中）。

脚本基于[Diffusers的DreamBooth](https://github.com/huggingface/diffusers/tree/main/examples/dreambooth)，但增加了以下功能（某些功能在原始脚本中也得到了支持）。

脚本的主要功能如下：

- 通过8位Adam优化器和latents缓存实现内存省略（与[Shivam Shrirao版](https://github.com/ShivamShrirao/diffusers/tree/main/examples/dreambooth)相同）。
- 通过xformers实现内存省略。
- 不仅在512x512上进行学习，而且可以在任意尺寸上进行学习。
- 通过数据扩充提高质量。
- 支持DreamBooth以外的Text Encoder+U-Net微调。
- Stable Diffusion格式的模型读写。
- 纵横比bucketing。
- Stable Diffusion v2.0兼容。

# 学习流程

请参考此存储库的README，并准备好环境。

## 准备数据

请参阅[有关准备训练数据的文档](./train_README-ja.md)。

## 进行训练

执行脚本。以下是最大限度地节省内存的命令（实际上在一行中输入）。根据需要更改每行。它似乎可以在约12GB的VRAM上运行。

```
accelerate launch --num_cpu_threads_per_process 1 train_db.py 
    --pretrained_model_name_or_path=<.ckpt or .safetensord or Diffusers版模型的目录>
    --dataset_config=<准备数据创建的.toml文件>
    --output_dir=<训练模型的输出文件夹>
    --output_name=<训练模型输出时的文件名>
    --save_model_as=safetensors 
    --prior_loss_weight=1.0 
    --max_train_steps=1600 
    --learning_rate=1e-6 
    --optimizer_type="AdamW8bit" 
    --xformers 
    --mixed_precision="fp16" 
    --cache_latents 
    --gradient_checkpointing
```

`num_cpu_threads_per_process`通常应指定为1。

`pretrained_model_name_or_path`指定要进行额外学习的原始模型。可以指定Stable Diffusion的checkpoint文件（.ckpt或.safetensors）、Diffusers本地磁盘上的模型目录、Diffusers模型ID（例如“stabilityai/stable-diffusion-2”）。

`output_dir`指定保存训练后模型的文件夹。使用`output_name`指定模型文件的名称，不包括扩展名。通过`save_model_as`指定保存为safetensors格式。

指定 `dataset_config` 文件为 `.toml` 文件。在文件中，将 batch_size 设置为 `1`，以减少内存消耗。

`prior_loss_weight` 是正则化图像的损失权重。通常将其设置为 1.0。

将训练步数 `max_train_steps` 设置为 1600。学习率 `learning_rate` 在这里设置为 1e-6。

为了节省内存，指定 `mixed_precision="fp16"`（在 RTX30 系列及以上版本中，还可以指定为 `bf16`。请与加速设置相匹配）。同时，指定 `gradient_checkpointing`。

为了使用内存占用更少的 8 位 AdamW 优化器（将模型优化为适合于训练数据的类），指定 `optimizer_type="AdamW8bit"`。

指定 `xformers` 选项并使用 xformers 的 CrossAttention。如果未安装 xformers 或出现错误（取决于环境，例如 `mixed_precision="no"` 的情况），则可以指定 `mem_eff_attn` 选项，以使用省内存版的 CrossAttention（速度会变慢）。

为了节省内存，指定 `cache_latents` 选项，以缓存 VAE 的输出。

如果有足够的内存，请编辑 `.toml` 文件，将 batch_size 增加到大约 `4` 左右（可能会加快速度并提高精度）。另外，取消 `cache_latents` 选项可以进行数据增强。

### 关于常用选项

请参阅 [train_README-ja.md](./train_README-ja.md) 中的 "常用选项"，以学习如何在以下情况下使用：

- 学习 Stable Diffusion 2.x 或其派生模型
- 学习需要 clip skip 大于等于 2 的模型
- 使用超过 75 个令牌的字幕进行训练

### 关于 DreamBooth 的步数

为了节省内存，每个步骤的训练次数减半（将目标图像和正则化图像分别分成不同的批次进行训练）。

如果要进行与原始 Diffusers 版本或 XavierXiao 先生的 Stable Diffusion 版本几乎相同的训练，请将步数加倍。

（由于将训练图像和正则化图像分组后进行 shuffle，因此严格来说，数据的顺序会发生变化，但我认为这对训练没有太大影响。）

### 关于 DreamBooth 的批次大小

与像 LoRA 这样的训练相比，为了学习整个模型，内存消耗量会更大（与 fine tuning 相同）。

### 关于学习率

在 Diffusers 版本中，学习率为 5e-6，而 Stable Diffusion 版本中学习率为 1e-6，因此在上面的示例中，将学习率指定为 1e-6。

### 如果指定了旧格式的数据集配置

使用选项指定分辨率和批次大小。以下是命令行示例。

```
accelerate launch --num_cpu_threads_per_process 1 train_db.py 
    --pretrained_model_name_or_path=<.ckpt 文件、.safetensor 文件或 Diffusers 版本模型的目录> 
    --train_data_dir=<训练数据目录> 
    --reg_data_dir=<正则化图像目录> 
    --output_dir=<训练模型输出目录> 
    --output_name=<训练模型输出文件名> 
    --prior_loss_weight=1.0 
    --resolution=512 
    --train_batch_size=1 
    --learning_rate=1e-6 
    --max_train_steps=1600 
    --use_8bit_adam 
    --xformers 
    --mixed_precision="bf16" 
    --cache_latents
    --gradient_checkpointing
```

## 使用训练模型生成图像

训练完成后，safetensors 文件将按指定的名称输出到指定的文件夹中。

对于 v1.4/1.5 和其他派生模型，您可以在此模型中使用 Automatic1111 先生的 WebUI 进行推理。请将其放置在 models\Stable-diffusion 文件夹中。

对于 v2.x 模型，在 WebUI 中生成图像时，需要单独的 .yaml 文件来描述模型规格。对于 v2.x base，请使用 v2-inference.yaml，对于 768/v，请使用 v2-inference-v.yaml，并将其放置在相同的文件夹中，并将扩展名前面的部分命名为与模型相同的名称。

![image](https://user-images.githubusercontent.com/52813779/210776915-061d79c3-6582-42c2-8884-8b91d2f07313.png)

每个 .yaml 文件都可以在 [Stability AI 的 SD2.0 存储库](https://github.com/Stability-AI/stablediffusion/tree/main/configs/stable-diffusion) 中找到。

# DreamBooth 的其他主要选项

有关所有选项的详细信息，请参见其他文档。

## 从中间停止学习文本编码器 --stop_text_encoder_training

将 stop_text_encoder_training 选项指定为数字，可以在该步骤后停止学习文本编码器，并仅学习 U-Net。在某些情况下，可能会期望提高精度。

（我猜测文本编码器可能会先过拟合，因此可以防止它发生，但详细影响尚不清楚。）

## 不进行 Tokenizer 的填充 --no_token_padding

如果指定 no_token_padding 选项，则不会对 Tokenizer 的输出进行填充（与旧版 Diffusers 的 DreamBooth 相同）。


<!-- 
如果使用 bucketing（如下所述）并使用数据增强，命令行示例如下。

```
accelerate launch --num_cpu_threads_per_process 8 train_db.py 
    --pretrained_model_name_or_path=<.ckpt 文件、.safetensor 文件或 Diffusers 版本模型的目录>
    --train_data_dir=<训练数据目录>
    --reg_data_dir=<正则化图像目录>
    --output_dir=<训练模型输出目录>
    --resolution=768,512 
    --train_batch_size=20 --learning_rate=5e-6 --max_train_steps=800 
    --use_8bit_adam --xformers --mixed_precision="bf16" 
    --save_every_n_epochs=1 --save_state --save_precision="bf16" 
    --logging_dir=logs 
    --enable_bucket --min_bucket_reso=384 --max_bucket_reso=1280 
    --color_aug --flip_aug --gradient_checkpointing --seed 42
```


-->
