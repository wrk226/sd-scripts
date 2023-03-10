针对非日语用户：目前此自述文件仅提供日语版。对此造成的不便，我们深表歉意。我们将在不久的将来提供英文版。

以下是关于可以在`--dataset_config`中传递的设置文件的说明。

## 概述

通过传递配置文件，用户可以进行详细设置。

* 可设置多个数据集
    * 例如，可以为每个数据集设置“resolution”，然后将它们混合在一起进行训练。
    * 在同时支持 DreamBooth 方法和 fine tuning 方法的学习方法中，可以混合使用 DreamBooth 方法和 fine tuning 方法的数据集。
* 可以针对每个子集更改设置
    * 子集是将数据集分成图像目录或元数据目录的结果。多个子集组成数据集。
    * 可以为每个子集设置“keep_tokens”或“flip_aug”等选项。另一方面，“resolution”或“batch_size”等选项可以针对每个数据集进行设置，并且在属于同一数据集的子集中，该值是共同的。请参见下面的详细信息。

配置文件的格式可以是 JSON 或 TOML。考虑到编写的易读性，建议使用 [TOML](https://toml.io/ja/v1.0.0-rc.2)。以下说明将以使用 TOML 为前提。

以下是使用 TOML 编写的示例配置文件。

```toml
[general]
shuffle_caption = true
caption_extension = '.txt'
keep_tokens = 1

# 这是 DreamBooth 方式的数据集
[[datasets]]
resolution = 512
batch_size = 4
keep_tokens = 2

  [[datasets.subsets]]
  image_dir = 'C:\hoge'
  class_tokens = 'hoge girl'
  # 此子集使用 keep_tokens = 2（使用所属的 datasets 的值）

  [[datasets.subsets]]
  image_dir = 'C:\fuga'
  class_tokens = 'fuga boy'
  keep_tokens = 3

  [[datasets.subsets]]
  is_reg = true
  image_dir = 'C:\reg'
  class_tokens = 'human'
  keep_tokens = 1

# 这是 fine tuning 方式的数据集
[[datasets]]
resolution = [768, 768]
batch_size = 2

  [[datasets.subsets]]
  image_dir = 'C:\piyo'
  metadata_file = 'C:\piyo\piyo_md.json'
  # 此子集使用 keep_tokens = 1（使用 general 的值）
```

在这个例子中，我们将使用3个目录作为DreamBooth数据集进行学习，图像大小为512x512（batch size为4），使用1个目录作为fine tuning数据集进行学习，图像大小为768x768（batch size为2）。

## 数据集子集设置

数据集子集设置分为几个可注册的部分。

* `[general]`
    * 指定适用于整个数据集或整个子集的选项。
    * 如果每个数据集和子集都存在同名选项，则数据集子集的设置将优先考虑。
* `[[datasets]]`
    * `datasets` 是有关数据集设置的注册部分。指定适用于每个数据集的选项的地方。
    * 如果存在子集设置，则子集设置将优先考虑。
* `[[datasets.subsets]]`
    * `datasets.subsets` 是有关子集设置的注册部分。指定适用于每个子集的选项的地方。

下图是有关图像目录和注册部分的对应关系的示意图，与前面的示例相关。

```
C:\
├─ hoge  ->  [[datasets.subsets]] No.1  ┐                        ┐
├─ fuga  ->  [[datasets.subsets]] No.2  |->  [[datasets]] No.1   |->  [general]
├─ reg   ->  [[datasets.subsets]] No.3  ┘                        |
└─ piyo  ->  [[datasets.subsets]] No.4  -->  [[datasets]] No.2   ┘
```

每个图像目录对应一个 `[[datasets.subsets]]`，多个 `[[datasets.subsets]]` 组合成一个 `[[datasets]]`。`[general]` 中包括所有的 `[[datasets]]` 和 `[[datasets.subsets]]`。

虽然每个注册部分可以指定不同的选项，但是如果存在同名选项，则下层注册部分的值优先考虑。可以查看之前例子中的 `keep_tokens` 选项来更好地理解这种处理方式。

此外，根据使用的学习方法不同，可以指定的选项也会有所不同。

* DreamBooth 专用选项
* fine tuning 专用选项
* 可使用 caption dropout 方法的选项

对于同时可用 DreamBooth 和 fine tuning 学习方法的情况，可以同时使用两种方法。
需要注意的是，由于根据数据集单元判断 DreamBooth 还是 fine tuning 方法，因此不能将 DreamBooth 方法的子集和 fine tuning 方法的子集混合在同一个数据集中。
换句话说，如果想要同时使用这两种方法，就需要将不同方法的子集分别分配到不同的数据集中。

程序的行为是，如果存在 `metadata_file` 选项，则判断为 fine tuning 方法的子集。
因此，对于属于同一数据集的子集来说，“全部具有 `metadata_file` 选项”或“全部不具有 `metadata_file` 选项”两者都可以。

以下是可用选项的说明。如果命令行参数与名称相同，则基本省略说明。请参考其他 README。

### 全部学习方法共通的选项

无论使用何种学习方法，都可以指定的选项。

#### 数据集选项

与数据集相关的选项。不能在 `datasets.subsets` 中进行说明。

| 选项名称 | 设置示例 | `[general]` | `[[datasets]]` |

| ---- | ---- | ---- | ---- |
| `batch_size` | `1` | o | o |
| `bucket_no_upscale` | `true` | o | o |
| `bucket_reso_steps` | `64` | o | o |
| `enable_bucket` | `true` | o | o |
| `max_bucket_reso` | `1024` | o | o |
| `min_bucket_reso` | `128` | o | o |
| `resolution` | `256`, `[512, 512]` | o | o |

* `batch_size`
    * 与命令行参数 `--train_batch_size` 相同。

这些设置是针对每个数据集固定的。
换句话说，属于数据集的子集将共享这些设置。
例如，如果要准备具有不同分辨率的数据集，则可以像上面的示例那样将其定义为不同的数据集，以便设置不同的分辨率。

#### 子集选项

与子集相关的选项。

| 选项名称 | 设置示例 | `[general]` | `[[datasets]]` | `[[dataset.subsets]]` |
| ---- | ---- | ---- | ---- | ---- |
| `color_aug` | `false` | o | o | o |
| `face_crop_aug_range` | `[1.0, 3.0]` | o | o | o |
| `flip_aug` | `true` | o | o | o |
| `keep_tokens` | `2` | o | o | o |
| `num_repeats` | `10` | o | o | o |
| `random_crop` | `false` | o | o | o |
| `shuffle_caption` | `true` | o | o | o |

* `num_repeats`
    * 指定子集中图像的重复次数。相当于 fine tuning 中的 `--dataset_repeats`，但 `num_repeats` 可以在任何学习方法中指定。

### fine tuning 方式専用のオプション

fine tuning 方式的选项仅适用于子集选项。

#### 子集选项

与 fine tuning 方式子集相关的选项。

| 选项名称 | 设置示例 | `[general]` | `[[datasets]]` | `[[dataset.subsets]]` |
| ---- | ---- | ---- | ---- | ---- |
| `image_dir` | `‘C:\hoge’` | - | - | o（必須） |
| `caption_extension` | `".txt"` | o | o | o |
| `class_tokens` | `“sks girl”` | - | - | o |
| `is_reg` | `false` | - | - | o |

### fine tuning 方式的选项仅适用于子集选项。

与 fine tuning 方式子集相关的选项。

| 选项名称 | `[general]` | `[[datasets]]` | `[[dataset.subsets]]` |
| ---- | ---- | ---- | ---- |
| `image_dir` | - | - | o |
| `metadata_file` | o（必须） | - | o |

* `image_dir`
    * 指定图像目录的路径。与 DreamBooth 方法不同，必须指定图像文件直接放置的路径。
    * 指定是可选的，但建议设置。
        * 没有必要指定的情况是，在生成元数据文件时使用了 `--full_path` 参数。
    * 图像必须放置在目录的顶层。
* `metadata_file`
    * 指定用于子集的元数据文件的路径。必须指定此选项。
        * 相当于命令行参数 `--in_json`。
    * 由于每个子集需要指定元数据文件，因此最好避免跨目录创建单个元数据文件的情况。强烈建议为每个图像目录准备元数据文件，并将它们注册为单独的子集。

### 仅适用于可使用 caption dropout 方法的情况的选项

只有在可使用 caption dropout 方法的子集中才能指定的选项。

与可使用 caption dropout 方法的子集相关的选项。

| 选项名称 | `[general]` | `[[datasets]]` | `[[dataset.subsets]]` |
| ---- | ---- | ---- | ---- |
| `caption_dropout_every_n_epochs` | o | o | o |
| `caption_dropout_rate` | o | o | o |
| `caption_tag_dropout_rate` | o | o | o |

## 存在重复子集时的行为

对于 DreamBooth 数据集，如果其中包含了具有相同 `image_dir` 的子集，则将其视为重复子集。
对于 fine tuning 数据集，如果其中包含了具有相同 `metadata_file` 的子集，则将其视为重复子集。
如果数据集中存在重复子集，则忽略第二个及以后的子集。

另一方面，如果它们属于不同的数据集，则不会被视为重复。
例如，如果将具有相同 `image_dir` 的子集放入不同的数据集中，则将其视为不重复。
这对于希望以不同的分辨率学习相同图像的情况非常有用。

```toml
# 如果它们存在于不同的数据集中，则不会被视为重复，并且两者都将用于训练

[[datasets]]
resolution = 512

  [[datasets.subsets]]
  image_dir = 'C:\hoge'

[[datasets]]
resolution = 768

  [[datasets.subsets]]
  image_dir = 'C:\hoge'
```

## 与命令行参数的结合使用

在设置文件的选项中，有些选项与命令行参数的选项重复。

以下命令行参数选项将在传递设置文件时被忽略。

* `--train_data_dir`
* `--reg_data_dir`
* `--in_json`

以下命令行参数选项，如果同时在命令行参数和设置文件中指定，则以设置文件的值为优先。除非特别说明，否则将成为同名选项。

| 命令行参数选项     | 优先的设置文件选项     |
| ---------------------------------- | ---------------------------------- |
| `--bucket_no_upscale`              |                                    |
| `--bucket_reso_steps`              |                                    |
| `--caption_dropout_every_n_epochs` |                                    |
| `--caption_dropout_rate`           |                                    |
| `--caption_extension`              |                                    |
| `--caption_tag_dropout_rate`       |                                    |
| `--color_aug`                      |                                    |
| `--dataset_repeats`                | `num_repeats`                      |
| `--enable_bucket`                  |                                    |
| `--face_crop_aug_range`            |                                    |
| `--flip_aug`                       |                                    |
| `--keep_tokens`                    |                                    |
| `--min_bucket_reso`                |                                    |
| `--random_crop`                    |                                    |
| `--resolution`                     |                                    |
| `--shuffle_caption`                |                                    |
| `--train_batch_size`               | `batch_size`                       |

## 错误指南

目前，我们使用外部库检查设置文件的正确性，但维护不到位，错误消息不太明确的问题。

作为临时措施，我们将提供常见错误及其解决方法。
如果您认为设置正确但出现错误，或者无法理解错误消息，请联系我们，这可能是一个错误。

* `voluptuous.error.MultipleInvalid: required key not provided @ ...`: 这是一个指定必要选项未提供的错误。您可能忘记指定选项或错误地编写了选项名称。
  * `...` 中指出了错误发生的位置。例如，如果出现 `voluptuous.error.MultipleInvalid: required key not provided @ data['datasets'][0]['subsets'][0]['image_dir']` 的错误，则表示在第 0 个 `datasets` 中的第 0 个 `subsets` 设置中不存在 `image_dir`。
* `voluptuous.error.MultipleInvalid: expected int for dictionary value @ ...`: 这是一个指定值格式不正确的错误。值的格式可能有误。`int` 部分取决于所涉及的选项。此 README 中的选项示例可能会有所帮助。
* `voluptuous.error.MultipleInvalid: extra keys not allowed @ ...`: 这是一个指定不受支持的选项名称的错误。您可能误写了选项名称或者不小心输错了。


