<!--由于文档正在更新，可能存在错误。-->

#关于训练，共同部分

本存储库支持模型的fine tuning、DreamBooth、LoRA和Textual Inversion的学习。本文档介绍了它们共同的学习数据准备方法和选项等。

#概述

请预先参考本存储库的README文件进行环境设置。


以下内容进行说明。

1.有关准备学习数据的内容（使用设置文件的新格式）
2.简要介绍用于训练的术语
3.以前的指定格式（不使用设置文件，而是从命令行指定）
4.生成训练中样本图像
5.各脚本中常用的常规选项
6.fine tuning方式元数据准备：如标题生成等

只需执行1，就可以暂时开始训练（有关训练，请参阅各脚本的文档）。如需了解2及其后内容，请根据需要参考。 


#学习数据的准备

在任何文件夹中（可以是多个文件夹），准备好学习数据的图像文件。支持`.png`、`.jpg`、`.jpeg`、`.webp`和`.bmp`格式的文件。通常不需要进行任何预处理，如调整大小等。

但是请勿使用比学习分辨率（稍后将进行说明）极小的图像，或者建议先在超分辨率AI等上进行放大处理。另外，似乎会出现错误，如果使用极端大的图像（大约3000x3000像素），请事先缩小它们。

在学习时，需要整理要向模型学习的图像数据，并将其指定给脚本。根据学习数据的数量、学习对象和是否准备好标题（即图像描述），可以使用多种方法指定学习数据。以下是可用的方式（这些名称不是一般的名称，而是本存储库独有的定义）。关于规则化图像的说明稍后进行。

1. DreamBooth，class+identifier方式（可使用规则化图像）

    学习将特定的单词（标识符）与学习目标相关联。无需准备标题。例如，如果要学习特定的角色，则无需准备标题，但由于学习数据的所有要素都与标识符相关联并进行了学习（例如，发型、服装、背景等），因此在生成时，可能会发生无法更改衣服等情况。

1. DreamBooth，标题方式（可使用规则化图像）

    准备记录了每个图像标题的文本文件，并进行学习。例如，如果要学习特定的角色，则通过在标题中记录图像详细信息（如穿着白色衣服的角色A、穿着红色衣服的角色A等）来将角色和其他元素分离，可以更严格地让模型只学习角色。

1. fine tuning方式（不支持规则化图像）

    预先将标题汇总到元数据文件中。支持将标签和标题分开管理，以及预先缓存潜在变量以加快学习速度等功能（这些功能在单独的文档中进行了说明）。（尽管名称为fine tuning方式，但也可以在fine tuning之外使用。）

#学习目标及可用的指定方法如下表所示。

| 学习目标或方法 | 脚本 | DB / class+identifier | DB / 标题方式 | fine tuning |
| ----- | ----- | ----- | ----- | ----- |
| 对模型进行微调 | `fine_tune.py` | x | x | o |
| DreamBooth 模型 | `train_db.py` | o | o | x |
| LoRA | `train_network.py` | o | o | o |
| 文本反转 | `train_textual_inversion.py` | o | o | o |

## 选择哪个方法

对于 LoRA 和 Textual Inversion，如果您想轻松地进行学习而无需准备标题文件，则DreamBooth class+identifier可能是最好的选择；如果您可以准备好标题文件，则DreamBooth 标题方式更好。如果学习数据数量很大并且不使用规则化图像，则还应考虑使用 fine tuning 方式。

对于 DreamBooth 也是同样的情况，但是fine tuning 方式不能使用。如果使用fine tuning，则只能使用fine tuning方式。

# 各种方式的指定方法

这里仅介绍每种指定方法的典型模式。有关更详细的指定方法，请参见 [数据集设置](./config_README-ja.md)。

# DreamBooth、class+identifier方式（可使用规则化图像）

使用此方法，每个图像都将通过类标识符（例如`shs dog`）进行学习。

## 第一步. 确定identifier和class

需要将要学习的对象与识别单词identifier以及对象所属的class进行绑定。

(class是指学习对象的一般类型，例如学习特定犬种时，class会被设定为dog。对于动漫角色，由于模型不同，class可能会被设定为boy、girl、1boy、1girl等。)

identifier是为了识别学习对象并进行学习而设定的单词。虽然可以任意使用单词，但根据原论文建议，选择在tokenizer中为1个token且长度小于3个字符的稀有单词会更好。

通过使用identifier和class，例如使用“shs dog”等来训练模型，可以从class中识别学习想要的对象并进行学习。

在图像生成时，如果使用“shs dog”，则可以生成所学习的犬种的图像。

（至于我最近使用的identifier，例如“shs sts scs cpc coc cic msm usu ici lvl cic dii muk ori hru rik koo yos wny”等，仅供参考。更理想的情况是选择不包含在Danbooru标签中的单词。）

## 第二步. 确定是否使用正则化图像，如果使用，则生成正则化图像。

正则化图像是为了防止学习对象发生“语言漂移”而生成的图像，其中class被保持不变。如果不使用正则化图像，则使用“1girl”等提示生成的图像将与特定角色相似，因为“1girl”在训练时的标题中包含了该角色。

通过同时学习学习对象的图像和正则化图像，class保持不变，只有在给出带有identifier的提示时，才会生成学习对象。

在LoRA或DreamBooth等只需要特定角色的情况下，可以不使用正则化图像。

在Textual Inversion中也不需要使用正则化图像（如果要学习的token字符串不包含在标题中，则不会学习任何内容）。

通常，使用学习对象模型生成仅包含class名称的图像作为正则化图像（例如“1girl”）。但是，如果生成图像的质量很差，则可以设计提示或使用从网上下载的图像。

（由于正则化图像也会被学习，因此其质量会影响模型。）

通常情况下，准备数百张图像是最好的做法（如果图像数量很少，则class图像将被泛化并学习它们的特征）。

如果使用生成的图像，请确保其大小与训练分辨率（更准确地说是bucket分辨率，稍后会解释）相适应。

## 第二步. 配置文件的编写

创建一个文本文件，将扩展名设置为 `.toml`。例如，可以按以下方式进行编写。

（以 `#` 开头的部分是注释，因此可以直接复制粘贴或删除它们。）

```toml
[general]
enable_bucket = true                        # 是否使用Aspect Ratio Bucketing

[[datasets]]
resolution = 512                            # 学习分辨率
batch_size = 4                              # 批量大小

  [[datasets.subsets]]
  image_dir = 'C:\hoge'                     # 指定用于学习的图像文件夹
  class_tokens = 'hoge girl'                # 指定identifier和class
  num_repeats = 10                          # 学习图像的重复次数

  # 只有在使用正则化图像时才需要编写以下内容。如果不使用，请删除。
  [[datasets.subsets]]
  is_reg = true
  image_dir = 'C:\reg'                      # 指定正则化图像的文件夹
  class_tokens = 'girl'                     # 指定class
  num_repeats = 1                           # 正则化图像的重复次数，通常设置为1即可。
```

基本上只需要更改以下位置就可以进行训练。

1. 学习分辨率

   如果指定一个数字，则将成为正方形（如果是512，则为512x512）。如果使用括号和逗号指定两个数字，则将成为横向和纵向的尺寸（例如，`[512,768]`将成为512x768）。在SD1.x系列中，原始的学习分辨率为512。指定较大的分辨率（例如`[512,768]`）可能会减少在生成纵向或横向图像时的破绽。在SD2.x 768系列中，分辨率为768。

2. 批量大小

   指定同时学习的数据量。这取决于GPU的VRAM大小和学习分辨率。有关详细信息，请参见后面的说明。还要注意，对于fine tuning/DreamBooth/LoRA等任务，批量大小可能会有所不同，因此请参阅各个脚本的说明。

3. 文件夹指定

   指定用于学习的图像文件夹和正则化图像（仅在使用时）。指定包含图像数据的文件夹本身。

4. 指定identifier和class

   如前所述的示例。

5. 重复次数

   请参见后面的说明。

### 关于重复次数

重复次数用于调整正则化图像数量和学习用图像数量。由于正则化图像数量多于学习用图像数量，因此通过重复学习用图像以匹配数量，从而可以以1:1的比率进行学习。

请指定重复次数为“__学习图像的重复次数×学习图像数量≥正则化图像的重复次数×正则化图像数量__”，以进行学习。

（数据数目为1个epoch（数据一周）时，“学习图像的重复次数×学习图像数量”将被设定。如果正则化图像数量超过这个数目，那么剩余的正则化图像将不会被使用。）

## 步骤3：学习

请参考各自的文档进行学习。

＃DreamBooth、字幕方式（可以使用规范化图像）

在这种方法中，每个图像都将通过字幕进行训练。

## 步骤1：准备字幕文件

在学习用图像文件夹中，以与图像相同的文件名放置扩展名为`.caption`的文件（可以在设置中更改）。请将每个文件限制为一行。编码为`UTF-8`。

## 步骤2：决定是否使用规范化图像，并在使用时生成规范化图像

与class + identifier形式相同。注意，可以为规范化图像添加字幕，但通常不需要。

## 步骤2：编写设置文件

创建一个文本文件，将扩展名设置为`.toml`。例如，您可以编写以下内容：```

```toml
[general]
enable_bucket = true                       # 是否使用Aspect Ratio Bucketing

[[datasets]]
resolution = 512                           # 学习分辨率
batch_size = 4                             # Batch大小

  [[datasets.subsets]]
  image_dir = 'C:\hoge'                    # 指定包含学习图像的文件夹
  caption_extension = '.caption'           # 标题文件的扩展名，如果使用 .txt，则更改为 .txt
  num_repeats = 10                         # 学习图像的重复次数

  # 以下仅在使用规范化图像时编写。如果不使用，请删除
  [[datasets.subsets]]
  is_reg = true
  image_dir = 'C:\reg'                     # 指定包含规范化图像的文件夹
  class_tokens = 'girl'                    # 指定类别
  num_repeats = 1                          # 规范化图像的重复次数，基本上为1即可
```

基本上只需更改位置即可进行学习。没有特别说明的部分与class+identifier方式相同。

1. 学习分辨率
2. Batch大小
3. 指定文件夹
4. 标题文件的扩展名

    可以指定任何扩展名。
5. 重复次数

## 步骤3：学习

请参考各自的文档进行学习。

＃微调方法

## 步骤1：准备元数据

包含标题和标签的管理文件称为元数据。其拓展名为`.json`，格式为json。由于创建方法很长，因此在本文档末尾进行了说明。

## 步骤2：编写设置文件

创建一个文本文件，将扩展名设置为`.toml`。例如，您可以编写以下内容：```

```toml
[general]
shuffle_caption = true
keep_tokens = 1

[[datasets]]
resolution = 512                                    # 学习分辨率
batch_size = 4                                      # Batch大小

  [[datasets.subsets]]
  image_dir = 'C:\piyo'                             # 指定包含学习图像的文件夹
  metadata_file = 'C:\piyo\piyo_md.json'            # 元数据文件名
```

基本上只需更改位置即可进行学习。没有特别说明的部分与DreamBooth、class+identifier方式相同。

1. 学习分辨率
2. Batch大小
3. 指定包含学习图像的文件夹
4. 指定元数据文件名

    指定使用以下方法创建的元数据文件。

## 步骤3：学习

请参考各自的文档进行学习。

# 用于学习的术语简单解释

由于细节被省略了，而我也没有完全理解，因此请各位自行查阅详细资料。

## 微调（fine tuning）

指的是通过学习来微调模型。根据使用方式不同，其含义也有所不同。在Stable Diffusion中，狭义的微调是指通过图像和标题来学习模型。DreamBooth是微调的一种特殊方式。广义的微调包括LoRA、Textual Inversion、Hypernetworks等，包括学习模型的所有方法。

## 步骤

简单来说，每次计算学习数据的次数就是一步。每一步中，模型会尝试将学习数据的标题流入当前模型，生成一张图片并将其与学习数据中的图像进行比较，然后略微修改模型使其更接近学习数据。

## Batch大小

Batch大小指定一次计算多少个数据进行学习。一次计算多个数据可以提高速度，通常也会提高准确度。

`Batch大小×步数` 将是用于学习的数据量。因此，如果增加Batch大小，则应该减少步数。

（但是，“Batch大小为1，步数为1600”和“Batch大小为4，步数为400”不会产生相同的结果。在相同的学习率下，后者通常会导致欠拟合。可以稍微增加学习率（例如`2e-6`），将步数减少到500等。）

增加Batch大小会占用更多的GPU内存。如果内存不足，将出现错误，并且在接近出错的边缘时，学习速度会下降。您可以通过任务管理器或`nvidia-smi`命令来检查内存使用情况。

此外，Batch是指“一组数据”。

## 学习率

大致意思是每一步更改的大小。如果指定一个较大的值，学习速度会更快，但是可能会改变太多而导致模型损坏或无法达到最佳状态。指定较小的值会减慢学习速度，并且可能无法达到最佳状态。

在fine tuning、DreamBooth和LoRA中，学习率的值大相径庭，而且还会因学习数据、学习模型、Batch大小和步数而有所不同。从常见的值开始，并观察学习状态，逐步调整。

默认情况下，整个学习过程中学习率保持不变。通过指定调度程序来确定如何更改学习率，因此结果也会发生变化。

## エポック（epoch）

当学習データ完成一轮（数据完整周转）后，称为1个epoch。若设定了重复次数，则在完成该次重复后，数据周转一次即为1个epoch。

1个epoch的步骤数基本上是`数据数量÷批次大小`，但使用Aspect Ratio Bucketing会微微增加（不同bucket的数据不能放在同一批次中，因此步骤数会增加）。

## Aspect Ratio Bucketing

Stable Diffusion v1是在512x512的分辨率下进行训练的，但它也在256x1024、384x640等分辨率下进行训练。这样可以减少被裁剪掉的部分，期望能更准确地学习图像和标题之间的关系。

此外，由于可以在任何分辨率下进行训练，因此不需要预先统一图像数据的纵横比。

虽然它可以在设置中启用并切换，但在此之前的设置文件示例中它是启用的（设置为`true`）。

学习分辨率将在给定分辨率的参数下，不超过该面积（=内存使用量）范围内，以64像素为单位（默认可更改）在垂直和水平方向上进行调整和创建。

在机器学习中，通常会统一所有输入大小，但实际上只要在同一批次内统一即可，没有特别的限制。NovelAI所说的bucketing是指预先将教师数据按照纵横比分类到相应的学习分辨率中。然后通过使用每个bucket内的图像创建批次，以统一批次图像大小。

# 先前的指定格式（不使用.toml文件，而是从命令行指定）

这是通过命名方式指定重复次数的方法。此外，使用`train_data_dir`选项和`reg_data_dir`选项。

## DreamBooth、class+identifier方式

使用文件夹名称指定重复次数。此外，使用`train_data_dir`选项和`reg_data_dir`选项。

### 步骤1. 准备训练用图像

创建一个用于存储训练图像的文件夹。在该文件夹中，使用以下名称创建目录。


```
<重复次数>_<identifier> <class>
```

请不要忘记``_``之间的空格。

例如，如果使用“sls frog”提示，在数据上重复20次，则应该是“20_sls frog”。如下所示。

![image](https://user-images.githubusercontent.com/52813779/210770636-1c851377-5936-4c15-90b7-8ac8ad6c2074.png)

### 学习多个class、多个identifier的方法

方法很简单，在存储训练图像的文件夹中，分别使用``<重复次数>_<identifier> <class>``的文件夹来为多个类和目标准备训练图像。同样，在规范化图像文件夹中也应该创建类似的文件夹，但不包含identifier。

例如，要同时学习“sls frog”和“cpc rabbit”的情况如下所示。

![image](https://user-images.githubusercontent.com/52813779/210777933-a22229db-b219-4cd8-83ca-e87320fc4192.png)

如果只有一个类别，但有多个目标，则规范化图像文件夹只需一个即可。例如，在1girl中有角色A和角色B的情况如下。

- train_girls
  - 10_sls 1girl
  - 10_cpc 1girl
- reg_girls
  - 1_1girl

### 步骤2. 准备规范化图像

这是使用规范化图像的步骤。

创建一个用于存储规范化图像的文件夹。在该文件夹中，使用``<重复次数>_<class>``的名称创建目录。

例如，如果使用“frog”提示，只需进行一次数据操作，如下所示。

![image](https://user-images.githubusercontent.com/52813779/210770897-329758e5-3675-49f1-b345-c135f1725832.png)


### 步骤3. 运行训练

运行各个训练脚本。使用`--train_data_dir`选项指定先前训练数据的文件夹（__不是包含图像的文件夹，而是其父文件夹__），并使用`--reg_data_dir`选项指定规范化图像的文件夹（__不是包含图像的文件夹，而是其父文件夹__）。

## DreamBooth、caption方式

在存储训练图像和规范化图像的文件夹中，放置与图像文件同名的文件，并使用扩展名.caption（可以使用选项更改）的文件来读取说明，并将其用作训练的提示。

※在这些图像的学习中，文件夹名称（identifier class）将不再使用。

默认情况下，说明文件的扩展名为.caption。您可以使用学习脚本的`--caption_extension`选项更改它。使用`--shuffle_caption`选项，可以在学习过程中对说明进行洗牌。

## fine tuning方式

与使用设置文件创建元数据的过程类似。使用`in_json`选项指定元数据文件。

# 中间训练样本输出

通过在正在训练的模型上生成图像样本，可以确认训练的进度。将以下选项指定为学习脚本。

- `--sample_every_n_steps` / `--sample_every_n_epochs`

指定采样输出的步数或周期数。每经过此步数或周期数时，就会进行采样输出。如果同时指定了两者，则以周期数为准。

- `--sample_prompts`

指定用于采样输出的提示文件。

- `--sample_sampler`

指定用于采样输出的采样器。可选的选项包括`'ddim', 'pndm', 'heun', 'dpmsolver', 'dpmsolver++', 'dpmsingle', 'k_lms', 'k_euler', 'k_euler_a', 'k_dpm_2', 'k_dpm_2_a'`。

要进行采样输出，需要事先准备好包含提示文本的文本文件。每行写一个提示。


```txt
# prompt 1
masterpiece, best quality, 1girl, in white shirts, upper body, looking at viewer, simple background --n low quality, worst quality, bad anatomy,bad composition, poor, low effort --w 768 --h 768 --d 1 --l 7.5 --s 28

# prompt 2
masterpiece, best quality, 1boy, in business suit, standing at street, looking back --n low quality, worst quality, bad anatomy,bad composition, poor, low effort --w 576 --h 832 --d 2 --l 5.5 --s 40
```

以#开头的行是注释行。可以使用 "--" + 英文小写字母 来指定生成图像的选项。以下选项可用：

- `--n` 将下一个选项设为负面提示。
- `--w` 指定生成图像的宽度。
- `--h` 指定生成图像的高度。
- `--d` 指定生成图像的种子。
- `--l` 指定生成图像的 CFG scale。
- `--s` 指定生成时的步数。

# 各脚本共用的常用选项

如果脚本更新后，文档还没有更新，可以使用 `--help` 选项查看可用选项。

## 学习模型的指定

- `--v2` / `--v_parameterization`

如果要使用 Hugging Face 的 stable-diffusion-2-base 或从那里细调模型（在推断时使用 `v2-inference.yaml` 的模型），请使用 `--v2` 选项。如果要使用 stable-diffusion-2 或 768-v-ema.ckpt，或从中细调模型（在推断时使用 `v2-inference-v.yaml` 的模型），请使用 `--v2` 和 `--v_parameterization` 两个选项。

    Stable Diffusion 2.0 有以下几个重要的变化：

    1. 使用的分词器
    2. 使用的文本编码器以及使用的输出层（2.0使用倒数第二层）
    3. 文本编码器的输出维度（从768到1024）
    4. U-Net的结构（CrossAttention的头数等）
    5. v-parameterization（采样方法似乎已经改变）

    在base模型中，使用了1-4，而在没有base的情况下（768-v），则使用了1-5。启用1-4选项的是 `--v2` 选项，启用5选项的是 `--v_parameterization` 选项。

- `--pretrained_model_name_or_path` 
    
    指定用于进行追加学习的模型。可以指定 Stable Diffusion 的checkpoint文件（.ckpt或.safetensors）、Diffusers 的本地模型目录、Diffusers 的模型ID（例如 "stabilityai/stable-diffusion-2"）。

## 关于学习的设置

- `--output_dir` 

    指定用于保存训练后模型的文件夹。
    
- `--output_name` 
    
    指定模型的文件名，不包括扩展名。
    
- `--dataset_config` 

    指定包含数据集设置的 `.toml` 文件。

- `--max_train_steps` / `--max_train_epochs`

    指定训练的步数或者 epoch 数。如果两个都指定了，以 epoch 数为准。

- `--mixed_precision`

    为了节省内存，使用混合精度（mixed precision）进行训练。可以指定 `--mixed_precision="fp16"` 等。与默认值相比，可能会降低精度，但是可以大大减少所需的GPU内存。
    
    （在 RTX30 系列及更高版本中，也可以指定 `bf16`。请与加速时的设置相一致）。
    
- `--gradient_checkpointing`

    通过逐步计算重量而不是一次性计算来减少需要的GPU内存量。打开或关闭不会影响精度，但是如果打开，可以增加批量大小，从而会对其产生影响。
    
    通常，如果打开，则速度会变慢，但是可以增加批量大小，因此总体的训练时间可能会更快。指定 `xformers` 选项以使用 `xformers` 的 CrossAttention。如果未安装 `xformers` 或出现错误（具体取决于环境，例如 `mixed_precision="no"` 的情况），则可以指定 `mem_eff_attn` 选项以使用省内存版 CrossAttention（速度比 `xformers` 慢）。

- `--xformers` / `--mem_eff_attn`

    指定 `xformers` 选项以使用 `xformers` 的 CrossAttention。如果未安装 `xformers` 或出现错误（具体取决于环境，例如 `mixed_precision="no"` 的情况），则可以指定 `mem_eff_attn` 选项以使用省内存版 CrossAttention（速度比 `xformers` 慢）。

- `--save_precision`

    指定保存时的数据精度。可以将 `save_precision` 选项指定为 `float`、`fp16` 或 `bf16` 中的任何一种格式来保存模型（在 DreamBooth 和 fine tuning 中使用 Diffusers 格式保存模型时无效）。如果想要缩小模型的大小等，请使用该选项。

- `--save_every_n_epochs` / `--save_state` / `--resume`

    如果指定了 `save_every_n_epochs` 选项并指定了数字，则会在每个 epoch 结束时保存训练中的模型。

    如果同时指定 `save_state` 选项，则还将保存学习状态，包括优化器等状态（虽然可以从保存的模型中恢复学习，但相比之下可以期待更高的精度和缩短的训练时间）。保存位置为文件夹。

    学习状态将在保存目标文件夹中的名为 `<output_name>-??????-state`（?????? 是 epoch 数）的文件夹中输出。请在长时间的学习期间使用它。

    要从保存的学习状态重新开始学习，请使用 `resume` 选项。请指定包含学习状态的文件夹（而不是 `output_dir` 中的状态文件夹）。

    另外，请注意，由于 Accelerator 的规格，epoch 数和 global step 没有保存，当您恢复时，它们将从1开始，请谅解。

- `--save_model_as` （仅适用于 DreamBooth 和 fine tuning）

    可以从`ckpt，safetensors，diffusers，diffusers_safetensors` 中选择保存模型的格式。

    请指定 `--save_model_as=safetensors` 等。如果要将 Stable Diffusion 格式（ckpt 或 safetensors）加载并保存为 Diffusers 格式，则缺少的信息将从 Hugging Face 的 v1.5 或 v2.1 中获取并补充。

- `--clip_skip`

    如果指定 `2`，则使用 Text Encoder (CLIP) 的倒数第二层输出。如果省略选项或指定为 `1`，则使用最后一层输出。

    ※由于 SD2.0 默认使用倒数第二层，因此请不要在 SD2.0 学习中指定。

    如果要使用原始模型中已经训练好使用倒数第二层的模型，请指定 `2`。
如果使用的是最后一层进行学习，那么整个模型都是基于这个前提进行学习的。因此，如果要再次使用第二层进行学习，可能需要一定数量的教师数据和较长的学习时间才能获得期望的学习结果。

- `--max_token_length`

    默认值为75。通过指定150或225，可以扩展令牌长度以进行学习。请在使用长标题进行学习时指定。

    但是，由于学习时令牌扩展的规格略有不同（例如分割规格），因此如果不需要，请建议在75上进行学习。

    与clip_skip类似，为了以与模型的学习状态不同的长度进行学习，可能需要一定数量的教师数据和较长的学习时间。

- `--logging_dir` / `--log_prefix`

    这是有关保存训练日志的选项。请使用logging_dir选项指定日志保存文件夹。日志以TensorBoard格式保存。

    例如，指定--logging_dir=logs，然后工作文件夹中会创建一个名为logs的文件夹，其中包含按日期命名的文件夹，其中包含日志。
    另外，如果指定--log_prefix选项，则在日期之前添加指定的字符串。例如，使用“--logging_dir=logs --log_prefix=db_style1_”进行标识。

    要在TensorBoard中查看日志，请在工作文件夹中打开另一个命令提示符，并输入以下内容： 

    ```
    tensorboard --logdir=logs
    ```

    在准备环境时应安装TensorBoard，但如果没有安装，可以使用“pip install tensorboard”命令安装。

    然后打开浏览器，访问http://localhost:6006/即可显示。

- `--noise_offset`

    这是实现此文章的选项: https://www.crosslabs.org//blog/diffusion-with-offset-noise

    整体上，生成的暗色和明亮图像的结果可能会更好。这对于LoRA学习也很有效。建议指定约为`0.1`的值。

- `--debug_dataset`

    通过指定此选项，可以在进行学习之前确认要使用的图像数据和标题的内容。按Esc键可以退出并返回到命令行。

    ※ 在Linux环境（包括Colab）中，图像不会显示。

- `--vae`

    如果将vae选项指定为Stable Diffusion的检查点、VAE的检查点文件、Diffuses模型或VAE（都可以指定本地或Hugging Face的模型ID），则将使用该VAE进行学习（在获取latents的缓存或学习latents时）。

    在DreamBooth和微调中，保存的模型将包含该VAE。


## 优化器相关

- `--optimizer_type`
  --指定优化器的类型。以下可选：
  - AdamW：[torch.optim.AdamW](https://pytorch.org/docs/stable/generated/torch.optim.AdamW.html)
  - 未指定旧版本选项时与旧版本相同
  - AdamW8bit：参数同上
  - 指定--use_8bit_adam选项的旧版本相同
  - Lion：https://github.com/lucidrains/lion-pytorch
  - 指定--use_lion_optimizer选项的旧版本相同
  - SGDNesterov：[torch.optim.SGD](https://pytorch.org/docs/stable/generated/torch.optim.SGD.html)，nesterov=True
  - SGDNesterov8bit：参数同上
  - DAdaptation：https://github.com/facebookresearch/dadaptation
  - AdaFactor：[Transformers AdaFactor](https://huggingface.co/docs/transformers/main_classes/optimizer_schedules)
  - 任意优化器

- `--learning_rate`

  指定学习率。由于适当的学习率因学习脚本而异，请参阅各自的说明。

- `--lr_scheduler` / `--lr_warmup_steps` / `--lr_scheduler_num_cycles` / `--lr_scheduler_power`
  
  与学习率调度器相关的指定。

  在lr_scheduler选项中，可以从linear、cosine、cosine_with_restarts、polynomial、constant、constant_with_warmup中选择学习率调度器。默认值为constant。
    
  lr_warmup_steps指定调度器的预热（逐渐改变学习率）步数。
    
  lr_scheduler_num_cycles是cosine with restarts调度器中的重启次数，lr_scheduler_power是polynomial调度器中的polynomial power。

  有关详细信息，请自行查阅相关文档。

### 指定优化器

优化器的选项参数可以使用--optimizer_args选项进行指定。以key=value的形式，可以指定多个值。此外，value可以使用逗号分隔指定多个值。例如，如果要指定AdamW优化器的参数，则应该是``--optimizer_args weight_decay=0.01 betas=.9,.999``。

如果要指定选项参数，请查阅各个优化器的规格说明。

某些优化器需要必填的参数，如果省略，则会自动添加（例如SGDNesterov的动量参数）。请检查控制台输出。

D-Adaptation优化器会自动调整学习率。在学习率选项中指定的值不是学习率本身，而是应用D-Adaptation确定的学习率的适用率，通常应指定为1.0。如果要将U-Net的一半学习率指定为Text Encoder，请指定``--text_encoder_lr=0.5 --unet_lr=1.0``。

如果在AdaFactor优化器中指定relative_step=True，则可以自动调整学习率（默认情况下将添加）。如果要自动调整学习率，则必须使用adafactor_scheduler作为学习率调度器。此外，指定scale_parameter和warmup_init也是有用的。

指定自动调整选项的示例是 ``--optimizer_args "relative_step=True" "scale_parameter=True" "warmup_init=True"``。

如果不希望自动调整学习率，请添加选项参数``relative_step=False``。在这种情况下，将使用constant_with_warmup作为学习率调度程序，并且建议不要对梯度进行clip norm。因此，参数应为 ``--optimizer_type=adafactor --optimizer_args "relative_step=False" --lr_scheduler="constant_with_warmup" --max_grad_norm=0.0``。

### 使用任意优化器

如果要使用``torch.optim``的优化器，请仅使用类名（例如``--optimizer_type=RMSprop``），如果要使用其他模块的优化器，则指定“模块名.类名”（例如``--optimizer_type=bitsandbytes.optim.lamb.LAMB``）。

（这只是在内部使用importlib，尚未经过验证。如果需要，请安装相应的软件包。）


<!-- 
## 任意サイズ的图像训练 --resolution
可以在非正方形的图像上进行训练。请使用“宽度，高度”指定resolution，例如“448,640”。宽度和高度必须能被64整除。请将训练图像和规范化图像的尺寸调整为相同的大小。

个人经常生成纵向图像，因此也会使用“448,640”等进行训练。

## 比例桶分组 --enable_bucket / --min_bucket_reso / --max_bucket_reso
指定enable_bucket选项后将启用此功能。Stable Diffusion是在512x512上训练的，但也在256x768或384x640等分辨率下进行训练。

如果指定此选项，则无需将训练图像和规范化图像统一到特定的分辨率。从几个分辨率（宽高比）中选择最佳分辨率进行训练。分辨率以64像素为单位，因此可能与原始图像的宽高比不完全匹配，但在这种情况下，超出部分会略微裁剪。

可以使用min_bucket_reso选项指定最小分辨率，使用max_bucket_reso选项指定最大分辨率。默认值分别为256和1024。
例如，如果将最小大小设置为384，则将不再使用256x1024或320x768等分辨率。如果将分辨率增大到768x768之类的尺寸，则可以指定1280作为最大大小。

另外，请注意，在启用Aspect Ratio Bucketing时，最好为规范化图像准备与训练图像类似的各种分辨率。

（这样一来，在一个批次内的图像就不会偏向于训练图像或规范化图像。我认为影响不会太大......）

## 数据增强 --color_aug / --flip_aug
数据增强是通过在训练时动态改变数据来提高模型性能的技术。使用color_aug微调色彩并使用flip_aug进行左右翻转进行训练。

由于要动态更改数据，因此无法与cache_latents选项同时指定。


## 使用FP16梯度的学习（实验性功能）--full_fp16
指定full_fp16选项后，将使用float16（fp16）而不是常规的float32来训练梯度（似乎不是混合精度而是完全的fp16训练）。
这使得在512x512大小的SD1.x上，可以在少于8GB的VRAM使用量下进行训练，在512x512大小的SD2.x上可以在少于12GB的VRAM使用量下进行训练。

请提前在accelerate config中指定fp16，并将选项设置为“mixed_precision =”fp16“”（不适用于bf16）。

为了最小化内存使用量，请指定xformers、use_8bit_adam、cache_latents和gradient_checkpointing选项，并将train_batch_size设置为1。

（如果有余地，可以逐步增加train_batch_size，这样精度可能会略有提高。）

我们在PyTorch源代码中打了补丁来强行实现这一点（已在PyTorch 1.12.1和1.13.0中确认）。精度会大幅下降，并且学习失败的概率也会增加。
学习率和步数的设置也非常敏感。请自行承担责任并在认识到这些限制的情况下使用此功能。

-->

# 创建元数据文件

## 准备教师数据

按照前面所述的方法，准备要训练的图像数据，并将其放入任意文件夹中。

例如，您可以按以下方式存储图像。

![教師データフォルダのスクショ](https://user-images.githubusercontent.com/52813779/208907739-8e89d5fa-6ca8-4b60-8927-f484d2a9ae04.png)

## 自动描述（自动添加标签）

如果您不想使用描述来进行训练而只想使用标签，则可以跳过此步骤。

如果您希望手动添加描述，则应将描述与教师数据图像放在同一目录下，并使用相同的文件名和扩展名.caption进行命名。每个文件应该是仅有一行的文本文件。

### 使用BLIP自动描述

在最新版本中，不需要下载BLIP、下载权重或添加虚拟环境。它会直接运行。

运行finetune文件夹中的make_captions.py文件以自动添加描述。

```
python finetune\make_captions.py --batch_size <批处理大小> <教师数据文件夹>
```

如果使用批处理大小为8，将教师数据图像和描述文件放在父文件夹 train_data 中，则使用以下命令：
```
python finetune\make_captions.py --batch_size 8 ..\train_data
```

描述文件将与教师数据图像放在同一目录下，使用相同的文件名和扩展名.caption进行命名。

根据GPU的VRAM容量，可以增加或减少batch_size。大的batch_size将更快地训练模型（即使在12GB的VRAM上也可以稍微增加batch_size）。
可以使用max_length选项指定描述的最大长度。默认值为75。如果要使用长度为225的标记来训练模型，则可以将最大长度设为更长。
使用caption_extension选项可以更改描述文件的扩展名。默认情况下，扩展名为.caption（将其更改为.txt会与DeepDanbooru发生冲突）。

如果有多个教师数据文件夹，则对每个文件夹分别运行该命令。

请注意，由于推断中存在随机性，因此每次运行的结果都会有所不同。如果要使结果固定，请使用--seed选项指定一个随机种子，例如 `--seed 42`。

有关其他选项，请使用 `--help` 查看帮助（似乎没有关于参数含义的文档，所以只能查看源代码）。

默认情况下，描述文件将使用扩展名.caption生成。

![描述文件所在的文件夹](https://user-images.githubusercontent.com/52813779/208908845-48a9d36c-f6ee-4dae-af71-9ab462d1459e.png)

例如，图像将如下所示带有描述：

![描述和图像](https://user-images.githubusercontent.com/52813779/208908947-af936957-5d73-4339-b6c8-945a52857373.png)

## 使用DeepDanbooru进行标签化

如果不进行danbooru标签化本身，请继续进行“标题和标签信息的前处理”。

可以使用DeepDanbooru或WD14Tagger进行标记。WD14Tagger似乎更准确。如果要使用WD14Tagger进行标记，请继续下一章。

### 环境设置

将DeepDanbooru https://github.com/KichangKim/DeepDanbooru 克隆到工作文件夹中，或者下载zip并解压缩。我用zip解压缩了。
此外，请从DeepDanbooru的Releases页面https://github.com/KichangKim/DeepDanbooru/releases 的“DeepDanbooru Pretrained Model v3-20211112-sgd-e28”的Assets中下载deepdanbooru-v3-20211112-sgd-e28.zip，并将其解压缩到DeepDanbooru文件夹中。

从下面下载。点击“Assets”打开，然后从那里下载。

![DeepDanbooru下载页面](https://user-images.githubusercontent.com/52813779/208909417-10e597df-7085-41ee-bd06-3e856a1339df.png)

请将其组织成以下目录结构

![DeepDanbooru的目录结构](https://user-images.githubusercontent.com/52813779/208909486-38935d8b-8dc6-43f1-84d3-fef99bc471aa.png)

安装Diffusers所需的库。转到DeepDanbooru文件夹并进行安装（实际上只添加了tensorflow-io）。

```
pip install -r requirements.txt
```

接下来安装DeepDanbooru本身。

```
pip install .
```

上述步骤完成后，标记环境设置已完成。

### 进行标记

转到DeepDanbooru文件夹并运行deepdanbooru进行标记。

```
deepdanbooru evaluate <教师数据文件夹> --project-path deepdanbooru-v3-20211112-sgd-e28 --allow-folder --save-txt
```

如果将教师数据放在父文件夹train_data中，则如下所示。

```
deepdanbooru evaluate ../train_data --project-path deepdanbooru-v3-20211112-sgd-e28 --allow-folder --save-txt
```

标签文件与教师数据图像位于同一目录中，具有相同的文件名和扩展名.txt。由于每个文件都要逐个处理，因此速度相对较慢。

如果有多个教师数据文件夹，请对每个文件夹执行此操作。

生成的内容如下所示。

![DeepDanbooru生成的文件](https://user-images.githubusercontent.com/52813779/208909855-d21b9c98-f2d3-4283-8238-5b0e5aad6691.png)

标签会像这样附加在图像上（非常信息丰富......）。

![DeepDanbooru标签和图像](https://user-images.githubusercontent.com/52813779/208909908-a7920174-266e-48d5-aaef-940aba709519.png)

## 使用WD14Tagger进行标签化

这是使用WD14Tagger而不是DeepDanbooru的步骤。

使用Automatic1111的WebUI中使用的标记器。我参考了这个github页面（https://github.com/toriato/stable-diffusion-webui-wd14-tagger#mrsmilingwolfs-model-aka-waifu-diffusion-14-tagger）。

在初始的环境设置中，所需的模块已经安装。此外，权重将自动从Hugging Face下载。

### 执行标签化

执行脚本进行标记。
```
python tag_images_by_wd14_tagger.py --batch_size <批量大小> <教师数据文件夹>
```

如果将教师数据放在父文件夹train_data中，则如下所示。
```
python tag_images_by_wd14_tagger.py --batch_size 4 ..\train_data
```

在首次启动时，模型文件将自动下载到wd14_tagger_model文件夹中（文件夹可以使用选项更改）。生成的内容如下所示。

![下载的文件](https://user-images.githubusercontent.com/52813779/208910447-f7eb0582-90d6-49d3-a666-2b508c7d1842.png)

标签文件与教师数据图像位于同一目录中，具有相同的文件名和扩展名.txt。

![生成的标签文件](https://user-images.githubusercontent.com/52813779/208910534-ea514373-1185-4b7d-9ae3-61eb50bc294e.png)

![标签和图像](https://user-images.githubusercontent.com/52813779/208910599-29070c15-7639-474f-b3e4-06bd5a3df29e.png)

可以使用thresh选项指定confidence（置信度）达到多少以上才添加标签。默认值与WD14Tagger示例相同为0.35。降低值会添加更多标签，但精度会降低。

batch_size应根据GPU的VRAM容量进行调整。较大的值速度较快（即使VRAM为12GB，也可以再增加一些）。可以使用caption_extension选项更改标签文件的扩展名。默认值为.txt。

可以使用model_dir选项指定模型保存的目录。

还可以使用force_download选项重新下载模型，即使目标目录已经存在。

如果有多个教师数据文件夹，请对每个文件夹执行此操作。

## 处理标题和标签信息

将标题和标签作为元数据放入一个文件中以便于脚本处理。

### 处理标题

要将标题放入元数据中，请在工作文件夹中运行以下命令（如果不使用标题进行训练，则不需要执行此操作）（实际上，可在一行中编写，以下类似）。使用`--full_path`选项将图像文件的位置存储为完整路径以保存在元数据中。如果省略此选项，则使用相对路径记录，但在.toml文件中需要另外指定文件夹。

```
python merge_captions_to_metadata.py --full_path <教师数据文件夹>
　 --in_json <要加载的元数据文件名> <元数据文件名>
```

元数据文件名是任意的名称。
如果教师数据是train_data，没有要加载的元数据文件，元数据文件名为meta_cap.json，则如下所示。

```
python merge_captions_to_metadata.py --full_path train_data meta_cap.json
```

可以使用caption_extension选项指定标题的扩展名。

如果有多个教师数据文件夹，请指定full_path参数并对每个文件夹执行此操作。

```
python merge_captions_to_metadata.py --full_path 
    train_data1 meta_cap1.json
python merge_captions_to_metadata.py --full_path --in_json meta_cap1.json 
    train_data2 meta_cap2.json
```

如果省略in_json，则将从现有的元数据文件中读取并覆盖写入该文件。

__※最好每次更改in_json选项和写入目标以将其更改为不同的元数据文件。__

### 处理标签

同样，将标签合并到元数据中（如果不使用标签进行训练，则不需要执行此操作）。
```
python merge_dd_tags_to_metadata.py --full_path <教师数据文件夹>
--in_json <要加载的元数据文件名> <要写入的元数据文件名>
```

如果要按与前面相同的目录结构从meta_cap.json读取并写入meta_cap_dd.json，则为：
```
python merge_dd_tags_to_metadata.py --full_path train_data --in_json meta_cap.json meta_cap_dd.json
```

如果有多个教师数据文件夹，请指定full_path参数并对每个文件夹执行此操作。

```
python merge_dd_tags_to_metadata.py --full_path --in_json meta_cap2.json
    train_data1 meta_cap_dd1.json
python merge_dd_tags_to_metadata.py --full_path --in_json meta_cap_dd1.json 
    train_data2 meta_cap_dd2.json
```

如果省略in_json选项，则将从存在的元数据文件中读取并覆盖该文件。 
__*使用更改in_json选项和写入目标，以便每次将其更改为不同的元数据文件是更安全的。*__

###清理标题和标签

到目前为止，标题和DeepDanbooru标签已汇总到元数据文件中。但是，由于自动标题中存在写作差异等问题，标签可能包含下划线或评级（DeepDanbooru的情况），因此最好使用编辑器的替换功能等对标题和标签进行清理。

例如，如果要学习动画图像的少女，则标题中可能包含girl / girls / woman / women等。 另外，“anime girl”也可能更适合简单地称为“girl”。

有一个用于清理的脚本可用，因此请根据情况编辑脚本内容并使用它。

（不需要指定教师数据文件夹。将清理元数据中的所有数据。）

```
python clean_captions_and_tags.py <读取的元数据文件名> <写入的元数据文件名>
```

请注意，不需要使用--in_json选项。 例如：

```
python clean_captions_and_tags.py meta_cap_dd.json meta_clean.json
```

现在，标题和标签的预处理已完成。
## latents的预先获取

* * 这一步骤并非必需。您可以在训练过程中获取latents而无需省略它。
* * 另外，如果您在训练期间执行'random_crop'或'color_aug'等操作，则无法预先获取latents（因为您需要每次更改图像以进行训练）。如果不进行预处理，则可以使用到目前为止的元数据进行学习。

提前获取图像的潜在表达式并保存到磁盘中。这将使学习过程更快。同时进行bucketing（根据纵横比对教师数据进行分类）。

请在工作文件夹中输入以下内容。
```
python prepare_buckets_latents.py --full_path <教师数据文件夹>
    <要加载的元数据文件名> <要写入的元数据文件名>
    <要微调的模型名称或checkpoint>
    --batch_size <批大小>
    --max_resolution <分辨率 宽,高>
    --mixed_precision <精度>
```

如果模型为model.ckpt，批大小为4，学习分辨率为512*512，精度为no（float32），从meta_clean.json加载元数据并将其写入meta_lat.json，则应为以下内容：

```
python prepare_buckets_latents.py --full_path 
    train_data meta_clean.json meta_lat.json model.ckpt 
    --batch_size 4 --max_resolution 512,512 --mixed_precision no
```

latents将以numpy的npz格式保存在教师数据文件夹中。

您可以使用--min_bucket_reso选项指定最小分辨率大小，并使用--max_bucket_reso指定最大分辨率大小。默认值分别为256和1024。例如，如果将最小分辨率设置为384，则不再使用分辨率为256 * 1024或320 * 768的图像。
如果将分辨率增加到768 * 768等较大的值，则最好将最大分辨率设置为1280。

如果指定--flip_aug选项，则进行左右翻转的数据扩充。这可以将数据量伪装增加到两倍，但如果数据不是左右对称的（例如角色外观，发型等），则学习将不会进行得很好。

（这是一个简单的实现，它可以获取翻转后的图像的latents并保存为\_flip.npz文件。在fline_tune.py中，不需要特定的选项。如果有带有\_flip的文件，则会随机读取带有和不带有flip的文件。）

即使VRAM为12GB，批量大小也可以稍微增加一些。
分辨率应为64的倍数，并以“宽度，高度”指定。分辨率直接关系到微调时的内存大小。在12GB VRAM中，512,512似乎是极限（*）。如果有16GB，则可以将其提高到512,704或512,768。即使将其设置为256,256等，也似乎在VRAM 8GB中也很困难（参数，优化器等不涉及分辨率，但需要一定的内存）。

* * 据报道，在批量大小为1，VRAM为12GB，分辨率为640,640时，仍然可以运行。

以下是bucketing结果的示例。

![bucketingの結果](https://user-images.githubusercontent.com/52813779/208911419-71c00fbb-2ce6-49d5-89b5-b78d7715e441.png)

如果有多个教师数据文件夹，请指定full_path参数并针对每个文件夹执行操作。
```
python prepare_buckets_latents.py --full_path  
    train_data1 meta_clean.json meta_lat1.json model.ckpt 
    --batch_size 4 --max_resolution 512,512 --mixed_precision no

python prepare_buckets_latents.py --full_path 
    train_data2 meta_lat1.json meta_lat2.json model.ckpt 
    --batch_size 4 --max_resolution 512,512 --mixed_precision no

```
虽然可以将读取源和写入目标设置为相同，但将它们设置为不同会更加安全。

__* * 每次更改参数并将其写入不同的元数据文件中是更安全的做法。* *__


