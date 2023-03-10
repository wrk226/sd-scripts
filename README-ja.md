## 关于本repo
这是一个包含了Stable Diffusion训练、图像生成和其他脚本的代码库。

[README in English](./README.md) ←最新更新在此处

为了更加方便易用，bmaltais的代码库提供了GUI和PowerShell脚本等功能，请在[bmaltais的repo](https://github.com/bmaltais/kohya_ss)上查看（英文）。感谢bmaltais。

以下是可用的脚本：

* 支持DreamBooth、U-Net和Text Encoder的训练
* fine-tuning、同上
* 画像生成
* 模型转换（Stable Diffision ckpt/safetensors与Diffusers相互转换）

## 关于使用方法

在本代码库中和note.com中都有文章，请查看（将来可能全部移至此处）。

* [关于训练的共同部分](./train_README-ja.md) : 包括数据准备和选项等
    * [数据集设置](./config_README-ja.md)
* [DreamBooth训练指南](./train_db_README-ja.md)
* [fine-tuning指南](./fine_tune_README_ja.md):
* [LoRA训练指南](./train_network_README-ja.md)
* [Textual Inversionの训练指南](./train_ti_README-ja.md)
* note.com [图像生成脚本](https://note.com/kohya_ss/n/n2693183a798e)
* note.com [模型转换脚本](https://note.com/kohya_ss/n/n374f316fe4ad)

## 在Windows上运行需要的程序

需要Python 3.10.6和Git。

- Python 3.10.6: https://www.python.org/ftp/python/3.10.6/python-3.10.6-amd64.exe
- git: https://git-scm.com/download/win

如果使用PowerShell，请按照以下步骤更改安全设置以使用venv。
（请注意，这不仅适用于venv，还适用于脚本的运行权限。）

- 以管理员身份打开PowerShell。
- 输入"Set-ExecutionPolicy Unrestricted"，并回答Y。
- 关闭管理员PowerShell。

## Windows环境下的安装

以下示例安装PyTorch 1.12.1/CUDA 11.6版。如果使用CUDA 11.3版或PyTorch 1.13，请相应地更改。

（如果“python”只显示在python -m venv〜行中，请将python更改为py。）

在常规（非管理员）PowerShell中，按以下顺序执行：

```powershell
git clone https://github.com/kohya-ss/sd-scripts.git
cd sd-scripts

python -m venv venv
.\venv\Scripts\activate

pip install torch==1.12.1+cu116 torchvision==0.13.1+cu116 --extra-index-url https://download.pytorch.org/whl/cu116
pip install --upgrade -r requirements.txt
pip install -U -I --no-deps https://github.com/C43H66N12O12S2/stable-diffusion-webui/releases/download/f/xformers-0.0.14.dev0-cp310-cp310-win_amd64.whl

cp .\bitsandbytes_windows\*.dll .\venv\Lib\site-packages\bitsandbytes\
cp .\bitsandbytes_windows\cextension.py .\venv\Lib\site-packages\bitsandbytes\cextension.py
cp .\bitsandbytes_windows\main.py .\venv\Lib\site-packages\bitsandbytes\cuda_setup\main.py

accelerate config
```

<!-- 
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 --extra-index-url https://download.pytorch.org/whl/cu117
pip install --use-pep517 --upgrade -r requirements.txt
pip install -U -I --no-deps xformers==0.0.16
-->

使用命令提示符时则如下。


```bat
git clone https://github.com/kohya-ss/sd-scripts.git
cd sd-scripts

python -m venv venv
.\venv\Scripts\activate

pip install torch==1.12.1+cu116 torchvision==0.13.1+cu116 --extra-index-url https://download.pytorch.org/whl/cu116
pip install --upgrade -r requirements.txt
pip install -U -I --no-deps https://github.com/C43H66N12O12S2/stable-diffusion-webui/releases/download/f/xformers-0.0.14.dev0-cp310-cp310-win_amd64.whl

copy /y .\bitsandbytes_windows\*.dll .\venv\Lib\site-packages\bitsandbytes\
copy /y .\bitsandbytes_windows\cextension.py .\venv\Lib\site-packages\bitsandbytes\cextension.py
copy /y .\bitsandbytes_windows\main.py .\venv\Lib\site-packages\bitsandbytes\cuda_setup\main.py

accelerate config
```

（注意：因为使用 ``python -m venv --system-site-packages venv`` 可能会引起全局 Python 环境的问题，所以我将其改写为 ``python -m venv venv``，这样更加安全。）

请回答以下有关 accelerate config 的问题。（如果使用 bf16 进行训练，请在最后一个问题中回答 bf16。）

※从0.15.0开始，当在日语环境下按光标键进行选择时，程序会崩溃。请使用数字键 0、1、2 等进行选择。

```txt
- This machine
- No distributed training
- NO
- NO
- NO
- all
- fp16
```

※ 根据情况，可能会出现 "ValueError：fp16混合精度需要GPU" 错误。 在这种情况下，请在第6个问题中回答：“在这台机器上应该使用哪个GPU（按id）作为逗号分隔的列表进行训练？[all]：”，并回答“0”。 （将使用id `0`的GPU。）

### 关于PyTorch和xformers的版本

在其他版本中，可能会出现训练不成功的情况。如果没有特殊原因，请使用指定的版本。

## 升级

如果有新的版本发布，可以使用以下命令进行更新。

```powershell
cd sd-scripts
git pull
.\venv\Scripts\activate
pip install --use-pep517 --upgrade -r requirements.txt
```

如果命令成功执行，您将可以使用新版本。

## 致谢

LoRA的实现基于[cloneofsimo的存储库](https://github.com/cloneofsimo/lora)。我们表示感谢。

Conv2d 3x3的扩展是由[cloneofsimo](https://github.com/cloneofsimo/lora)最初发布的，KohakuBlueleaf在[LoCon](https://github.com/KohakuBlueleaf/LoCon)中证明了其有效性。我们深深地感谢KohakuBlueleaf。

## 许可证

脚本的许可证为ASL 2.0，但包含部分其他许可证的代码（包括Diffusers和cloneofsimo存储库的代码）。

[Memory Efficient Attention Pytorch](https://github.com/lucidrains/memory-efficient-attention-pytorch): MIT

[bitsandbytes](https://github.com/TimDettmers/bitsandbytes): MIT

[BLIP](https://github.com/salesforce/BLIP): BSD-3-Clause


