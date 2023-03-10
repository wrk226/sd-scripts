This repository contains training, generation and utility scripts for Stable Diffusion.

[__Change History__](#change-history) is moved to the bottom of the page.
更新历史已移至[页面底部](#change-history)。

[中文版README](./README-ch.md)

For easier use (GUI and PowerShell scripts etc...), please visit [the repository maintained by bmaltais](https://github.com/bmaltais/kohya_ss). Thanks to @bmaltais!

This repository contains the scripts for:

* DreamBooth training, including U-Net and Text Encoder
* Fine-tuning (native training), including U-Net and Text Encoder
* LoRA training
* Texutl Inversion training
* Image generation
* Model conversion (supports 1.x and 2.x, Stable Diffision ckpt/safetensors and Diffusers)

__Stable Diffusion web UI now seems to support LoRA trained by ``sd-scripts``.__ (SD 1.x based only) Thank you for great work!!! 

## About requirements.txt

These files do not contain requirements for PyTorch. Because the versions of them depend on your environment. Please install PyTorch at first (see installation guide below.) 

The scripts are tested with PyTorch 1.12.1 and 1.13.0, Diffusers 0.10.2.

## Links to how-to-use documents

All documents are in Japanese currently.

* [Training guide - common](./train_README-ja.md) : data preparation, options etc...
    * [Dataset config](./config_README-ja.md)
* [DreamBooth training guide](./train_db_README-ja.md)
* [Step by Step fine-tuning guide](./fine_tune_README_ja.md):
* [training LoRA](./train_network_README-ja.md)
* [training Textual Inversion](./train_ti_README-ja.md)
* note.com [Image generation](https://note.com/kohya_ss/n/n2693183a798e)
* note.com [Model conversion](https://note.com/kohya_ss/n/n374f316fe4ad)

## Windows Required Dependencies

Python 3.10.6 and Git:

- Python 3.10.6: https://www.python.org/ftp/python/3.10.6/python-3.10.6-amd64.exe
- git: https://git-scm.com/download/win

Give unrestricted script access to powershell so venv can work:

- Open an administrator powershell window
- Type `Set-ExecutionPolicy Unrestricted` and answer A
- Close admin powershell window

## Windows Installation

Open a regular Powershell terminal and type the following inside:

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

update: ``python -m venv venv`` is seemed to be safer than ``python -m venv --system-site-packages venv`` (some user have packages in global python).

Answers to accelerate config:

```txt
- This machine
- No distributed training
- NO
- NO
- NO
- all
- fp16
```

note: Some user reports ``ValueError: fp16 mixed precision requires a GPU`` is occurred in training. In this case, answer `0` for the 6th question: 
``What GPU(s) (by id) should be used for training on this machine as a comma-separated list? [all]:`` 

(Single GPU with id `0` will be used.)

### about PyTorch and xformers

Other versions of PyTorch and xformers seem to have problems with training.
If there is no other reason, please install the specified version.

## Upgrade

When a new release comes out you can upgrade your repo with the following command:

```powershell
cd sd-scripts
git pull
.\venv\Scripts\activate
pip install --use-pep517 --upgrade -r requirements.txt
```

Once the commands have completed successfully you should be ready to use the new version.

## Credits

The implementation for LoRA is based on [cloneofsimo's repo](https://github.com/cloneofsimo/lora). Thank you for great work!

The LoRA expansion to Conv2d 3x3 was initially released by cloneofsimo and its effectiveness was demonstrated at [LoCon](https://github.com/KohakuBlueleaf/LoCon) by KohakuBlueleaf. Thank you so much KohakuBlueleaf!

## License

The majority of scripts is licensed under ASL 2.0 (including codes from Diffusers, cloneofsimo's and LoCon), however portions of the project are available under separate license terms:

[Memory Efficient Attention Pytorch](https://github.com/lucidrains/memory-efficient-attention-pytorch): MIT

[bitsandbytes](https://github.com/TimDettmers/bitsandbytes): MIT

[BLIP](https://github.com/salesforce/BLIP): BSD-3-Clause

## Change History

- 9 Mar. 2023, 2023/3/9:
  - There may be problems due to major changes. If you cannot revert back to the previous version when problems occur, please do not update for a while.
  - Minimum metadata (module name, dim, alpha and network_args) is recorded even with `--no_metadata`, issue https://github.com/kohya-ss/sd-scripts/issues/254
  - `train_network.py` supports LoRA for Conv2d-3x3 (extended to conv2d with a kernel size not 1x1).
    - Same as a current version of [LoCon](https://github.com/KohakuBlueleaf/LoCon). __Thank you very much KohakuBlueleaf for your help!__
      - LoCon will be enhanced in the future. Compatibility for future versions is not guaranteed.
    - Specify `--network_args` option like: `--network_args "conv_dim=4" "conv_alpha=1"`
    - [Additional Networks extension](https://github.com/kohya-ss/sd-webui-additional-networks) version 0.5.0 or later is required to use 'LoRA for Conv2d-3x3' in Stable Diffusion web UI.
    - __Stable Diffusion web UI built-in LoRA does not support 'LoRA for Conv2d-3x3' now. Consider carefully whether or not to use it.__
  - Merging/extracting scripts also support LoRA for Conv2d-3x3.
  - Free CUDA memory after sample generation to reduce VRAM usage, issue https://github.com/kohya-ss/sd-scripts/issues/260 
  - Empty caption doesn't cause error now, issue https://github.com/kohya-ss/sd-scripts/issues/258
  - Fix sample generation is crashing in Textual Inversion training when using templates, or if height/width is not divisible by 8.
  - Update documents (Japanese only).

- 由于进行了大规模更改，可能存在缺陷。如果在执行脚本后无法将其回滚到之前的版本，请暂时避免更新。
- 即使使用 `--no_metadata` 选项，也将记录最少的元数据（模块名称、dim、alpha 和 network_args） issue https://github.com/kohya-ss/sd-scripts/issues/254
- `train_network.py` 现在支持 LoRA 的 Conv2d-3x3 扩展（也将范围扩展到 Conv2d 的大小不是 1x1 的情况）。
    - 这与当前版本的[LoCon](https://github.com/KohakuBlueleaf/LoCon)的规格相同。__我们深深地感谢 KohakuBlueleaf 的支持。__
      - 如果 LoCon 将来被扩展，无法保证与这些版本的兼容性。
    - 请使用 `--network_args` 选项指定类似 `--network_args "conv_dim=4" "conv_alpha=1"` 的参数。
    - Stable Diffusion web UI 使用时需要 [Additional Networks 扩展](https://github.com/kohya-ss/sd-webui-additional-networks)的 version 0.5.0 或更高版本。
    - __稳定的 Diffusion web UI 中的 LoRA 功能似乎不支持 LoRA 的 Conv2d-3x3 扩展。请谨慎考虑是否使用。__
  - 合并和提取脚本也支持 LoRA 的 Conv2d-3x3 扩展。
  - 生成样本图像后释放CUDA内存以减少VRAM使用量。 issue https://github.com/kohya-ss/sd-scripts/issues/260 
  - 现在可以使用空标题。 issue https://github.com/kohya-ss/sd-scripts/issues/258
  - 修正了使用 Textual Inversion 学习并使用模板时，当高度/宽度不能被8整除时生成样本图像会崩溃的问题。
  - 更新了文档。

  - Sample image generation:
    A prompt file might look like this, for example

    ```
    # prompt 1
    masterpiece, best quality, 1girl, in white shirts, upper body, looking at viewer, simple background --n low quality, worst quality, bad anatomy,bad composition, poor, low effort --w 768 --h 768 --d 1 --l 7.5 --s 28

    # prompt 2
    masterpiece, best quality, 1boy, in business suit, standing at street, looking back --n low quality, worst quality, bad anatomy,bad composition, poor, low effort --w 576 --h 832 --d 2 --l 5.5 --s 40
    ```

    Lines beginning with `#` are comments. You can specify options for the generated image with options like `--n` after the prompt. The following can be used.

    * `--n` Negative prompt up to the next option.
    * `--w` Specifies the width of the generated image.
    * `--h` Specifies the height of the generated image.
    * `--d` Specifies the seed of the generated image.
    * `--l` Specifies the CFG scale of the generated image.
    * `--s` Specifies the number of steps in the generation.

    The prompt weighting such as `( )` and `[ ]` are not working.

  - 生成示例图像：
    提示文件示例如下：

    ```
    # prompt 1
    masterpiece, best quality, 1girl, in white shirts, upper body, looking at viewer, simple background --n low quality, worst quality, bad anatomy,bad composition, poor, low effort --w 768 --h 768 --d 1 --l 7.5 --s 28

    # prompt 2
    masterpiece, best quality, 1boy, in business suit, standing at street, looking back --n low quality, worst quality, bad anatomy,bad composition, poor, low effort --w 576 --h 832 --d 2 --l 5.5 --s 40
    ```

    以 `#` 开头的行是注释。您可以使用“破折号两个+小写英文字母”的形式来指定选项，例如 `--n`。以下选项可用：

    * `--n` Negative prompt up to the next option.
    * `--w` Specifies the width of the generated image.
    * `--h` Specifies the height of the generated image.
    * `--d` Specifies the seed of the generated image.
    * `--l` Specifies the CFG scale of the generated image.
    * `--s` Specifies the number of steps in the generation.

    加权（如 `( )` 或 `[ ]`）不起作用。

Please read [Releases](https://github.com/kohya-ss/sd-scripts/releases) for recent updates.
请查看[Releases](https://github.com/kohya-ss/sd-scripts/releases)获取最新更新信息。
