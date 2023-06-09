# OpenPrompt 自制使用手册

整理人：陈皓_10205102429

### 1.参考文献及使用文档

**文献**：OpenPrompt: An Open-source Framework for Prompt-learning

**文献链接**：https://arxiv.org/abs/2111.01998

**文档网址**：https://thunlp.github.io/OpenPrompt/index.html

### 2. Base Class

#### 2.1 概述

Base Class部分给出的是prompt learning架构的基础。

对于prompt learning，首先需要根据对应的<u>预训练任务</u>，选择一个现有的<u>预训练语言模型</u>。

`Template`类用来建立一个方法，该方法对于初始文本进行包装，使其对应于模型的训练任务。

`Verbalizer`类用来将标签定义到词表中的标签词中。

该代码框架下，后续的类都会继承这些基础的类。

#### 2.2 Prompt Base

##### ①`Template`

**父类：**`torch.nn`

**作用：**构造模板，对于输入的原始文本进行包装，使下游任务适用于预训练语言模型的格式。

**子类：**

**1）初始化参数**

| 名称                | 解释说明                                                     |
| ------------------- | ------------------------------------------------------------ |
| tokenizer           | 用于将输入转化成token <br />类型为`PreTrainedTokenizer`      |
| placeholder_mapping | 用于作原始文本中占位符到token的映射。<br />例子：**dict** = {'<text_a>':'text_a','<text_b>':'text_b'}, |

**2）类内成员变量**

| 名称               | 解释说明                                      |
| ------------------ | --------------------------------------------- |
| tokenizer          | 见初始化参数                                  |
| placehoder_mapping | 见初始化参数                                  |
| _in_on_text_set    | 作为一个对text进行设置的开关，默认为**False** |
| mixed_token_start  | 用于标识token的起点，默认为字符"{"            |
| mixed_token_end    | 用于标识token的终点，默认为字符"}"            |
| text               | 模板文本                                      |

**3）类内函数**

| 名称                              | 解释说明                                                     |
| --------------------------------- | ------------------------------------------------------------ |
| **\_init\_()**                    | **说明**：类的初始化，类内变量赋值                           |
| **get_default_loss_ids()**        | **返回**：List<br />**说明**：这个List标识出模板中哪个位置的token被掩盖掉。<br />列表中取值为0和1，0表示正常的序列token，1表示masked token |
| **get_default_shortenable_ids()** | **返回**：List<br />**说明**：为了适应模型的max_sequence_length，模板中也要有可伸缩的token。<br />默认情况下，input text是可伸缩的，而剩余部分需要保持固定。<br />所以列表中，1表示input tokens，0表示template sequence token。 |
| **get_default_soft_token_ids()**  | **返回**：List<br />**说明**：辨别哪些token是soft token，一般由子类接续编写。 |
| **incorporate_text_example(·)**   | **参数**：①example：类型为`InputExample`<br />②text：输入文本，默认为None<br />**返回**：text<br />**说明**：用于将`InputExample`和模板的text进行合并操作，统一为text的形式 |
| **_check_template_format()**      | **说明**：检查template的格式是否正确，主要判断mask tokens的个数是否为0。 |
| **parse_text(·)**                 | **参数**：①text：输入文本<br />**返回**：List。名称为parsed，存放将text语法分析的结果<br />**说明**：对文本进行语法分析 |
| **wrap_one_example(·)**           | **参数：**①example：类型为`InputExample`<br />**返回**：List[Dict]，将模板文本中的每一个token都转化成字典表示<br />示例：[{"loss_ids": 0, "text": "It was"}, {"loss_ids": 1, "text": "\<mask\>"}, ] |
| **process_batch**                 | 用于对batch操作的一个方法，需要用到的时候由子类完成续写      |
| **post_processing_outputs()**     | **参数**：①outputs<br />**返回**：outputs<br />**说明**：根据模板的需要对语言模型的输出进行后处理，由子类完成续写。 |
| **save(·)**                       | **参数**：path<br />**说明**：根据指定的路径保存`Template`   |
| **text**                          | 写作property，对文本赋值进行一些安全性的检查和设置           |
| **on_text_set**                   | 若在template文本被设置的同时需要完成操作，在子类中写这个函数 |
| **from_file(·)**                  | **参数**：①path:原始文本文件对应的路径<br />②choice：`int`型，默认为0，表示读第几行<br />**说明**：将文本文档中的内容读入，并存放到类中的text当中。 |
| **from_config(·）**               | 从configuration node中加载一个template                       |



##### ②`Verbalizer`

父类：`torch.nn`

**作用**：完成模型输出词语到对应的类别之间的映射。

**1）初始化参数**

| 名称        | 解释说明                                                  |
| ----------- | --------------------------------------------------------- |
| tokenizer   | 将单词转化成token的工具，常常从预训练语言模型里面load出来 |
| classes     | 存储输入数据的类别                                        |
| num_classes | 存储输入数据的类别数                                      |

**2）类内函数**

| 名称                            | 解释说明                                                     |
| ------------------------------- | ------------------------------------------------------------ |
| **label_words**                 | label words表示被特定的标签所投影到的词语集合。<br />如：positive$\rightarrow${wonderful,good},那么wonderful和good就是label words<br />做成了@property,在设置后自动调用函数**on_label_words_set()** |
| **generate_parameters()**       |                                                              |
| **register_calibrate_logits()** |                                                              |
| **process_outputs()**           | 默认情况下，verbalizer的输出就是预训练语言模型的输出。根据不同继承的子类，再编写不同的得到输出的方式 |
| **gather_outputs()**            |                                                              |
| **aggregate**                   |                                                              |
| **normalize**                   |                                                              |
| **project**                     |                                                              |
| **handle_multi_token**          |                                                              |
| **from_config**                 |                                                              |
| **from_file**                   | 从verbalizer文本文件中对类对象进行一个初始化                 |



#### 2.3 Pipeline Base

##### ①`PromptDataLoader`

**作用**：将用于prompt model的数据集进行一个包装



##### ②`PromptModel`

**父类：**`nn.Modules`

**作用：**在Openprompt的框架下将Template和transformers库的PretrainedModel进行了封装，使其构成一个Prompt的模型

**子类：**

+ PromptForClassification
+ PromptForGeneration

**1）初始化参数**

| 名称          | 解释说明                                                     |
| ------------- | ------------------------------------------------------------ |
| plm           | **自行传入**，类型为`PreTrainedModel`，传入预训练语言模型    |
| template      | **自行传入**，类型为`Template`，即构成prompt的一个模板       |
| freeze_plm    | 默认为False                                                  |
| plm_eval_mode | 默认为False。（上两点为一些特定参数的冻结，正常使用情况下无需传参） |

**2）类内函数**

| 名称                           | 解释说明                                                     |
| ------------------------------ | ------------------------------------------------------------ |
| **\_init\_()**                 | 初始化赋值，获取预训练语言模型的参数。随后执行**_prepare_main_input_name()** |
| **_prepare_main_input_name()** | 对类内变量main_input_name根据不同情况进行赋值                |
| **forward()**                  | **参数**：①batch，为 **Union**[Dict, InputFeatures]<br />**返回**：`torch.Tensor`<br />**说明**：从模板中提取出input batch，随后喂给plm得到output，并将output交由Template处理过一次后返回最终的output。<br />（forward函数制定前向传播的规则，用于预测mask tokens的值） |

##### ③`PromptForClassification`

**父类：**`PromptModel`

**说明**：本类在`PromptModel`的基础上加入了一个用于分类的头部，这个头部会通过Verbalizer，将输出映射到类别上

**1）初始化参数**

| 名称          | 解释说明                                                     |
| ------------- | ------------------------------------------------------------ |
| plm           | **自行传入**，为`PretrainedModel`类，为在transformers库中选择的预训练模型 |
| template      | **自行传入**，为`Template`类，是prompt模型当中预先设置的模板 |
| verbalizer    | **自行传入**，为`Verbalizer`类，用于作输出label words到label的映射 |
| freeze_plm    | 表示是否冻结预训练语言模型的所有参数，默认为False            |
| plm_eval_mode | 更强力的freeze，会将dropout一起冻结，默认为False             |

**2）类内函数**

| 名称                  | 解释说明                                                     |
| --------------------- | ------------------------------------------------------------ |
| **\_init\_()**        | 利用传入的参数进行赋值，如使用plm和template定义一个`PromptModel`<br />随后比prompt模型多添置一个Verbalizer，用于标签转化 |
| **extract_at_mask()** | **参数**：①outputs（`torch.Tensor`）：为初始的输出（可能来自于Verbalizer)<br />②batch(Union[`Dict`,`InputFeatures`])：为初始的batch<br />**返回**：outputs，为所有mask token的输出<br />**说明**：主要用于将Verbalizer的输出分析称mask token的输出，作为一种方法用于前向传播函数当中 |
| **forward()**         | 主要过程是将batch喂给预训练语言模型，随后将模型的输出喂给verbalizer,最后将verbalizer的输出分析成每个词语的输出（前向传播） |



##### ④`PromptForGeneration`



### 3.Template

**简介**：模板是prompt-learning中最重要的模块之一，它将原始的输入进行封装和处理

#### 3.1 Manual Template

##### ①`ManualTemplate`

**父类：**`Template`   ([链接](#####①`Template`))

**作用：**该类表示手动构建模板，基本上沿用了父类的架构，没有太大的区别。（一般而言，无特殊要求用这种形式的模板即可）	

**子类：**

**1）初始化参数**

| 名称                | 解释说明                                                     |
| ------------------- | ------------------------------------------------------------ |
| tokenizer           | 用于将输入转化成token <br />类型为`PreTrainedTokenizer`      |
| placeholder_mapping | 用于作原始文本中占位符到token的映射。<br />例子：**dict** = {'<text_a>':'text_a','<text_b>':'text_b'}, |
| text                | 也可在初始化的时候传入text对template的模板进行规定<br />（一般采用from_file的方法） |

**2）类内函数**

| 名称            | 解释说明                                                     |
| --------------- | ------------------------------------------------------------ |
| **\_init\_**()  | 调用父类的初始化函数，若传入text，则对text进行赋值           |
| **on_text_set** | 调用父类的**parse_text()**函数，在文本被设定的同时进行语法分析 |



#### 3.2 Prefix Template

##### ①`PrefixTuningTemplate`

**父类**：`Template`   ([链接](#####①`Template`))

**作用**：实现了prefix-tuning方法中的模板编写。

**参考文献**：[Prefix-Tuning: Optimizing Continuous Prompts for Generation](https://arxiv.org/abs/2101.00190)

我的图片
![1](image/1.png)


**1）初始化参数**

| 名称                          | 解释说明                                                     |
| ----------------------------- | ------------------------------------------------------------ |
| model                         | **自行传入**，为`PreTrainedModel`类。（用于将model内的一些参数传入进来） |
| tokenizer                     | **自行传入**，为`PreTrainedTokenizer`                        |
| mapping_hook                  |                                                              |
| text                          |                                                              |
| num_token                     | 作为前缀的token长度                                          |
| placeholder_mapping           |                                                              |
| prefix_dropout                |                                                              |
| mid_dim                       |                                                              |
| using_encoder_past_key_values |                                                              |
| using_decoder_past_key_values |                                                              |

**2）类内函数**

| 名称                              | 解释说明                                                     |
| --------------------------------- | ------------------------------------------------------------ |
| **\_init\_**()                    | 调用父类的初始化函数，若传入text，则对text进行赋值           |
| **on_text_set()**                 | 文本被安置上之后被自动调用<br />调用父类的**parse_text()**函数进行语义分析。<br />随后调用自身定义的**generate_parameters()**用于生成参数 |
| **get_past_key_values**           | 返回上一个时刻的键值对                                       |
| **generate_parameters**           |                                                              |
| **wrap_one_example**              |                                                              |
| **expand_to_batchsize**           |                                                              |
| **expand_to_batchsize_for_layer** |                                                              |
| **process_batch**                 |                                                              |
| **modify_plm**                    | （仅当T5模型的时候要modify）                                 |

(未整理完毕)



#### 3.3 Ptuning Template

##### ①`PtuningTemplate`



#### 3.4 PTR Template

##### ①`PTRTemplate`



#### 3.5 Mixed Template

##### ①`MixedTemplate`



### 4. Verbalizer

**简介**：verbalizer是prompt-learning当中非常重要的一个组成部分，负责将原始的label投影到单词上

#### 4.1 One to One Verbalizer

##### ①`One2oneVerbalizer`

**父类**：`Verbalizer`

**作用**：实现了最基本的单个单词到类别映射的Verbalizer

**1）初始化参数**

| 名称                | 解释说明                                                     |
| ------------------- | ------------------------------------------------------------ |
| tokenizer           | 来源于预训练语言模型，用于将词表中的词转化为token            |
| classes             | 存放当前任务的类别标签。例子：Agnews中的['World', 'Sports', 'Business', 'Tech'] |
| num_classes         | 类别的总数目                                                 |
| label_words         | 标签词，和类别一一对应                                       |
| prefix              | 前缀词（用在对前缀敏感的预训练语言模型中，如Reberta)         |
| multi_token_handler |                                                              |
| post_log_softmax    | 是否将对数softmax传递给label的概率，默认情况下为True(传)     |

**2）类内函数**

| 名称                     | 解释说明                                                     |
| ------------------------ | ------------------------------------------------------------ |
| **on_label_words_set()** | 将传进来的label_words通过**add_prefix()**方法加上对应的prefix<br />并调用**generate_parameters()**生成参数 |
| **add_prefix**           |                                                              |
| **generate_parameters**  |                                                              |
| **project**              |                                                              |
| **process_logits**       |                                                              |
| **normalize**            |                                                              |
| **calibrate**            |                                                              |

(尚未整理完毕，仔细查看源码后再编写，或可以根据研究进度，此块内容先放一放)



#### 4.2 Manual Verbalizer

##### ①`ManualVerbalizer`

**父类**：`Verbalizer`

**作用**：实现了最基本的人工制定的Verbalizer类。

**1）初始化参数**

| 名称                | 解释说明                                                     |
| ------------------- | ------------------------------------------------------------ |
| tokenizer           | 来源于预训练语言模型，用于将词表中的词转化为token            |
| classes             | 存放当前任务的类别标签。例子：Agnews中的['World', 'Sports', 'Business', 'Tech'] |
| label_words         | 被标签投影的单词                                             |
| prefix              | 前缀词（用在对前缀敏感的预训练语言模型中，如Reberta)         |
| multi_token_handler |                                                              |
| post_log_softmax    | 是否将对数softmax传递给label的概率，默认情况下为True(传)     |

**2）类内函数**

| 名称                      | 解释说明                                                     |
| ------------------------- | ------------------------------------------------------------ |
| **on_label_words_set()**  | 将传进来的label_words通过**add_prefix()**方法加上对应的prefix<br />并调用**generate_parameters()**生成参数 |
| **add_prefix(·)**         | **参数**：①label_words；②prefix<br />**返回**：new_label_words<br />**说明**：将prefix 拼接在label_words的前面 |
| **generate_parameters()** | 对self.label_words进行编码，生成三个编码矩阵：<br />①label_words_ids——每个类别词，都用一个二维矩阵表示，存储为三维矩阵<br />②words_ids_mask——也是对应类别词语的<br />③label_words_mask——留空，弄懂了补回来 |
| **project()**             | 返回每个label对应的可能性，                                  |
| **process_logits**()      | 返回最终每个类的logits                                       |
| **normalize**             | 正则化                                                       |
| **calibrate**             | 校准                                                         |



#### 4.3 Automatic Verbalizer

##### ①`AutomaticVerbalizer`

**参考文献**： [Automatically Identifying Words That Can Serve as Labels for Few-Shot Text Classification](https://aclanthology.org/2020.coling-main.488.pdf).

#### 4.4 Knowledgeable Verbalizer

##### ①`KnowledgeableVerbalizer`

**父类**：`ManualVerbalizer`

**作用**：实现了Knowledgeable Prompt Tuning的模板编写。这个Verbalizer的特点是应用了外部知识

**参考文献**：[Knowledgeable Prompt-tuning: Incorporating Knowledge into Prompt Verbalizer for Text Classification](https://arxiv.org/pdf/2108.02035.pdf).

![2](image/2.png)

注：KPT的特点是运用外部知识，扩充Label Words的范围，充分利用了预训练语言模型的潜能。在前向传播的过程中，预训练语言模型给出词表中填入MASK部分的一个概率，通过KPT，将这个概率分布转化为不同类别的概率，选取最大者作为最终的分类。

**1）初始化参数**

| 名称                | 解释说明                                                     |
| ------------------- | ------------------------------------------------------------ |
| tokenizer           | 来源于预训练语言模型，用于将词表中的词转化为token            |
| classes             | 存放当前任务的类别标签。例子：Agnews中的['World', 'Sports', 'Business', 'Tech'] |
| prefix              | 前缀词（用在对前缀敏感的预训练语言模型中，如Reberta)         |
| multi_token_handler |                                                              |
| max_token_split     |                                                              |
| verbalizer_lr       | Verbalizer优化的学习率                                       |
| candidate_frac      |                                                              |

**2）类内函数**

| 名称                            | 解释说明                                                     |
| ------------------------------- | ------------------------------------------------------------ |
| **on_label_words_set()**        | 传入label words后自动调用。<br />调用类中**delete_common_words()**用于删除不合要求的词语（删除方式参考论文）；**add_prefix()**用于添置一个前缀；**generate_parameters()**用于生成参数。 |
| **generate_parameters()**       | 编码操作，将外部单词扩充进入label_words()的群体当中          |
| **register_calibrate_logits()** |                                                              |
| **project()**                   |                                                              |
| **aggregate()**                 |                                                              |

**一些问题**：当label word set中词语数量变化的时候，需要观察`Knowledgeable`类中的project函数的变化，对应知识库中单词的使用情况（暂时理解成是维度的扩展）

```python
def process_logits(self, logits: torch.Tensor, **kwargs):
        r"""A whole framework to process the original logits over the vocabulary, which contains four steps:

        (1) Project the logits into logits of label words

        if self.post_log_softmax is True:

            (2) Normalize over all label words

            (3) Calibrate (optional)

        (4) Aggregate (for multiple label words)

        Args:
            logits (:obj:`torch.Tensor`): The original logits.

        Returns:
            (:obj:`torch.Tensor`): The final processed logits over the labels (classes).
        """
        # project
        # print("Process call!")
        label_words_logits = self.project(logits, **kwargs)  #Output: (batch_size, num_classes) or  (batch_size, num_classes, num_label_words_per_label)


        if self.post_log_softmax:
            # normalize
            label_words_probs = self.normalize(label_words_logits)

            # calibrate
            if  hasattr(self, "_calibrate_logits") and self._calibrate_logits is not None:
                label_words_probs = self.calibrate(label_words_probs=label_words_probs)

            # convert to logits
            label_words_logits = torch.log(label_words_probs+1e-15)

        # aggregate
        label_logits = self.aggregate(label_words_logits)
        return label_logits
    
    def aggregate(self, label_words_logits: torch.Tensor) -> torch.Tensor:
        r"""Use weight to aggregate the logits of label words.

        Args:
            label_words_logits(:obj:`torch.Tensor`): The logits of the label words.

        Returns:
            :obj:`torch.Tensor`: The aggregated logits from the label words.
        """
        # 将label_words的概率整合成label的概率
        label_words_logits = (label_words_logits * self.label_words_mask).sum(-1)/self.label_words_mask.sum(-1)
        return label_words_logits
```

假设最长词语的长度是n，

label_words_logits的维度就是[a,b,n]

而需要求到的label_logits的维度就是[a,b]，即应用到的Knowledge base的信息需要被提取出来了。

**校准开关启动时，模型的反向传播出错**

#### 4.5 PTR Verbalizer

##### ① `PTRVerbalizer`



#### 4.6 Generation Verbalizer

##### ①`GenerationVerbalizer`



#### 4.7 Soft Verbalizer

##### ① `SoftVerbalizer`

