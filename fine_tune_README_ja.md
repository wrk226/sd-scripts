这是一种 fine tuning 方法，支持 NovelAI 提出的学习方法、自动描述、标签化以及 Windows + VRAM 12GB（适用于 SD v1.x）环境等。在此，fine tuning 指的是使用图像和描述来训练模型（不包括 LoRA、Textual Inversion 和 Hypernetworks）。

请参阅 [共同培训文档](./train_README-ja.md) 以获取有关培训的常见信息。

# 概述

使用 Diffusers 对 Stable Diffusion 的 U-Net 进行微调。它支持以下改进，这些改进在 NovelAI 的文章中被提出（关于纵横比存储桶，参考了 NovelAI 的代码，但最终代码均为原创）。

* 使用 CLIP（文本编码器）倒数第二层的输出而不是最后一层的输出。
* 在非方形分辨率下进行学习（纵横比存储桶）。
* 将令牌长度从 75 扩展到 225。
* 使用 BLIP 进行描述生成，使用 DeepDanbooru 或 WD14Tagger 进行自动标签化。
* 支持 Hypernetwork 的训练。
* 适用于 Stable Diffusion v2.0（基础和 768/v）。
* 通过预先获取 VAE 的输出并将其保存到磁盘上，实现训练的省内存和快速化。

默认情况下，不进行 Text Encoder 的训练。在整个模型的 fine tuning 中，只学习 U-Net 是常见的做法（NovelAI 也是如此）。可以使用选项指定 Text Encoder 作为训练目标。

# 关于额外功能

## 更改 CLIP 的输出

为了将提示反映在图像中，需要将文本特征转换为图像，CLIP（文本编码器）用于此。在 Stable Diffusion 中，使用 CLIP 的最后一层的输出，但可以更改为使用倒数第二层的输出。据 NovelAI 表示，这可以更准确地反映提示。仍然可以使用原始的最后一层输出。

※ 在 Stable Diffusion 2.0 中，默认使用倒数第二层。请不要指定 clip_skip 选项。

## 在正方形以外的分辨率下进行训练

Stable Diffusion被训练在512*512的分辨率上，但也可以在256*1024或384*640等其他分辨率上进行训练。这将减少被裁剪的部分，期望能更准确地学习到提示和图像之间的关系。
训练分辨率是根据给定的分辨率参数来创建的，不能超过该分辨率面积（即内存使用量）范围，以64像素为单位在水平和垂直方向上进行调整。

在机器学习中，通常将所有输入大小统一，但实际上只要在同一批次内统一即可，没有特殊限制。NovelAI所说的bucketing是指预先将训练数据根据宽高比分别分类到相应的学习分辨率中。然后通过使用各个bucket内的图像来创建批次，从而统一批次的图像大小。

## 将标记长度从75扩展到225

Stable Diffusion最大支持75个标记（包括起始和结束标记为77个），但现在将其扩展到225个标记。
但是，CLIP接受的最大长度为75个标记，因此在225个标记的情况下，将其简单地分成三个部分调用CLIP，然后将结果连接起来。

※我不确定这是否是期望的实现方式。但它似乎能够运行。特别是在2.0中没有任何有用的实现，因此我已经独自实现了它。

※Automatic1111先生的Web UI似乎会意识到逗号并进行分割，但我的实现比较简单，只是简单地分割。

# 训练过程

请先参考本存储库的README，进行环境配置。

## 数据准备

请参阅[有关准备训练数据的说明](./train_README-ja.md)。对于fine tuning，只支持使用元数据进行fine tuning方式。

## 执行训练
例如，可以按以下方式执行训练。以下是为省略内存的设置。请根据需要更改每行内容。

```
accelerate launch --num_cpu_threads_per_process 1 fine_tune.py 
    --pretrained_model_name_or_path=<.ckpt或.safetensord或Diffusers版模型的目录> 
    --output_dir=<训练后模型的输出目录> 
    --output_name=<训练后模型的文件名，不包括扩展名> 
    --dataset_config=<在数据准备过程中创建的.toml文件> 
    --save_model_as=safetensors 
    --learning_rate=5e-6 --max_train_steps=10000 
    --use_8bit_adam --xformers --gradient_checkpointing
    --mixed_precision=fp16
```

通常情况下，应该将`num_cpu_threads_per_process`设置为1。

使用`pretrained_model_name_or_path`指定要进行追加训练的基础模型。可以指定Stable Diffusion的checkpoint文件（.ckpt或.safetensors）、Diffusers本地磁盘上的模型目录或Diffusers模型ID（如"stabilityai/stable-diffusion-2"）。

使用`output_dir`指定保存训练后模型的文件夹，使用`output_name`指定模型文件的名称（不包括扩展名）。使用`save_model_as`指定以safetensors格式保存模型。

使用`dataset_config`指定`.toml`文件。在文件中，批次大小设置为`1`，以抑制内存使用。

将训练步骤数`max_train_steps`设置为10000。学习率`learning_rate`在此处设置为5e-6。

使用`mixed_precision="fp16"`减少内存占用（在RTX30系列及以上版本中，也可以使用`bf16`）。另外，指定`gradient_checkpointing`。

使用内存消耗较少的8位AdamW优化器（将模型优化以适合训练数据），使用`optimizer_type="AdamW8bit"`指定。

使用`xformers`选项，并使用xformers的CrossAttention。如果未安装xformers或出现错误（取决于环境，如`mixed_precision="no"`的情况），可以使用`mem_eff_attn`选项代替，使用省内存版CrossAttention（速度会变慢）。

如果有足够的内存，请编辑`.toml`文件，将批次大小增加到例如`4`（可能会提高速度和精度）。

### 关于常用选项

请参考以下情况的选项文档：

- 学习Stable Diffusion 2.x或派生模型
- 学习假设clip skip大于2的模型
- 使用超过75个标记的标题进行训练

### 关于批量大小

与学习LoRA等整个模型相比，内存消耗量会增加（与DreamBooth相同）。

### 关于学习率

通常在1e-6到5e-6之间。请参考其他fine tuning的示例等。

### 当指定旧格式的数据集时的命令行

使用选项指定分辨率和批量大小。以下是命令行示例。

```
accelerate launch --num_cpu_threads_per_process 1 fine_tune.py 
    --pretrained_model_name_or_path=model.ckpt 
    --in_json meta_lat.json 
    --train_data_dir=train_data 
    --output_dir=fine_tuned 
    --shuffle_caption 
    --train_batch_size=1 --learning_rate=5e-6 --max_train_steps=10000 
    --use_8bit_adam --xformers --gradient_checkpointing
    --mixed_precision=bf16
    --save_every_n_epochs=4
```

<!--
### 使用fp16进行训练（实验性功能）
当指定full_fp16选项时，将梯度从常规float32更改为float16（fp16）进行训练（似乎不是混合精度而是完全的fp16训练）。这样，在SD1.x的512 * 512大小下，似乎可以在少于8GB的VRAM使用量下进行训练，在SD2.x的512 * 512大小下，似乎可以在少于12GB的VRAM使用量下进行训练。

为了预先指定加速配置中的fp16，请使用选项mixed_precision =“fp16”（不适用于bf16）。

为了最小化内存使用量，请指定各选项xformers、use_8bit_adam、gradient_checkpointing，并将train_batch_size设置为1。
（如果有余地，请逐步增加train_batch_size，这样可以稍微提高准确性。）

我们在PyTorch源代码中应用了补丁，强行实现了这一点（在PyTorch 1.12.1和1.13.0中进行了确认）。精度会大大降低，并且学习过程中失败的概率也会变高。学习率和步数的设置似乎也很敏感。请在认识到这些情况后自行承担责任使用。

-->

# fine tuning其他主要选项

请参阅单独的文档以了解所有选项。

## `train_text_encoder`
将Text Encoder作为训练对象。内存使用量会略微增加。

通常的fine tuning不会将Text Encoder作为训练对象（可能是为了使U-Net遵循Text Encoder的输出），但如果训练数据量较少，则在Text Encoder侧进行训练也是有效的，例如像DreamBooth这样的情况。

## `diffusers_xformers`
使用Diffusers的xformers功能而不是脚本独有的xformers替换功能。无法训练Hypernetwork。 

