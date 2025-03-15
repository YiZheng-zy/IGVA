<h2 align="center">Instruction-Guided Fusion of Multi-Layer Visual Features in Large Vision-Language Models</h2>

<div align="center">

[English](README.md) | 简体中文

</div>

The official implementation of the paper "[Instruction-Guided Fusion of Multi-Layer Visual Features in Large Vision-Language Models](https://arxiv.org/abs/2501.08443)" 的官方实现。


## 📣 新闻

- **[Dec 26, 2024]** The paper has been released on [arXiv](https://arxiv.org/abs/2501.08443)发布！
- **[March 10, 202]**  🔥🔥🔥 Code has been released.

## 目录
- [Performance](#Performance)
- [Framework](#Framework)
- [Evaluate](#Evaluate)
- [训练](#训练)
- [评估](#评估)

## Performance

<p align="center">
    <img src="images/radar_chart.png" width="70%"></a>
</p>

Performance comparison of our method against the baseline and competing approaches.
+ We systematically analyze how hierarchical visual features influence LVLM
performance across diverse task categories. Our findings reveal that different
layers of the vision encoder play distinct roles, emphasizing the necessity of
task-aware feature fusion rather than static fusion strategies.
+ We propose an instruction-guided vision aggregator, which dynamically assigns
fusion weights to hierarchical visual features based on task-specific instructions.
This mechanism enables LVLMs to selectively emphasize task-relevant features,
improving adaptability without increasing the number of visual tokens.
+ We integrate the proposed vision aggregator into the LLaVA-v1.5 framework,
achieving significant improvements over the baseline and surpassing existing
hierarchical visual feature fusion methods as well as similarly scaled LVLMs

<p align="center">
    <img src="images/mmfuser-diagram.png" width="95%"></a>
</p>



## Framework

The overall framework is illustrated. It consists of four main modules: a vision encoder $V$, a vision-language adapter $ADP$, an instruction-guided vision aggregator $\mathit{IGVA}$, and an LLM. For demonstration purposes, we use the widely adopted LLaVA-v1.5  as the implementation framework.

![Overview of the proposed framework.](images/Framework.png)

**Vision Encoder**  
We use CLIP-ViT as the vision encoder. It divides an input image $\mathbf{I} \in \mathbb{R}^{C \times H \times W}$ into small patches and processes the patch sequence through a stack of transformer layers. The hierarchical output of the vision encoder is:

```math
\mathbf{F} = V(\mathbf{I}) \in \mathbb{R} ^ {L \times (N+1) \times D},
```

where $L$ is the number of transformer layers, $N$ is the number of patches, and $D$ is the hidden dimension. Each layer produces $N+1$ features, as a `<cls>` token is prepended to the patch sequence to aggregate global information.

---

**Instruction-guided Vision Aggregator**  
The key innovation of our method is the vision aggregator, which dynamically integrates hierarchical visual features based on task instructions. We divide the $L$ layers of the vision encoder into $K$ groups, each containing $L/K$ consecutive layers. For each group, average pooling is applied to the `<cls>` hidden states to obtain a group-wise global feature:

```math
\mathbf{F}^{cls}_{k} = \mathit{Avg}(\mathbf{cls}_{k_1}, \mathbf{cls}_{k_2}, ..., \mathbf{cls}_{k_{L/K}}) \in \mathbb{R}^{ D}, \quad \text{for } k = 1, 2, ..., K,
```

where $\mathbf{cls}_{k_i} \in \mathbb{R}^{D}$ is the `<cls>` feature of the i-th layer in group $k$. The aggregator comprises a sentence embedding model and a weight allocator. The sentence embedding model encodes the textual instruction into a semantic embedding $\mathbf{s}$, which is passed to the weight allocator to compute the importance of each visual group:

```math
\mathbf{w} = \mathit{IGVA}(\mathbf{s}, \mathbf{F}^{cls}_{1}, \mathbf{F}^{cls}_{2}, ..., \mathbf{F}^{cls}_{K}) \in \mathbb{R}^{K}, \quad \text{where } \sum_{k=1}^K w_k = 1.
```

For each group, we apply average pooling to the patch features. Then, we perform weighted summation across the pooled features using the weight vector $\mathbf{w}$:

```math
\mathbf{F}^{patch}_k = \mathit{Avg}(\mathbf{F}^{patch}_{k_1}, \mathbf{F}^{patch}_{k_2}, ..., \mathbf{F}^{patch}_{k_{L/K}}) \in \mathbb{R} ^ {N \times D}, \quad \text{for } k = 1, ..., K.
```

```math
\mathbf{F}_{fused} = \sum_{k=1}^K w_k \times \mathbf{F}^{patch}_k \in \mathbb{R} ^ {N \times D},
```

where $\mathbf{F}^{patch}_{k_i} \in \mathbb{R}^{N\times D}$ represents the patch features of the i-th layer in group $k$. Finally, to preserve semantic richness, the fused features are concatenated with the penultimate layer’s features to form the final visual representation:

```math
\hat{\mathbf{F}} = \mathit{Concate}(\mathbf{F}_{fused}, \mathbf{F}_{penultimate}) \in \mathbb{R} ^ {N \times 2D}.
```

---

**Vision-Language Adapter**  
The vision-language adapter aligns the final visual representation with the embedding space of the LLM. It consists of a two-layer MLP with a GELU activation function:

```math
ADP(\hat{\mathbf{F}}) = \mathit{Proj}_2(\mathit{GELU}(\mathit{Proj}_1(\hat{\mathbf{F}}))) \in \mathbb{R} ^ {N \times D_t},
```

where $D_t$ is the hidden dimension of the LLM.

---

**Large Language Model**  
The LLM first tokenizes the textual instruction and computes embeddings for each token. These text embeddings are then concatenated with the aligned visual features along the sequence dimension. The resulting combined sequence is processed through a stack of transformer layers, ultimately generating a textual response.

## Evaluate
### VQA benchmarks
<p align="center">
    <img src="images/VQA.png" width="90%"></a>
</p>

Performance comparison on mainstream image-based VQA benchmarks. Bold values
indicate the best score in each row, while underlined values represent the second-best score.


Table presents a comparison of our method with the baseline and two existing hi-
erarchical visual feature fusion approaches across 10 image-based VQA benchmarks,
including 6 LVLM-specific benchmarks and 4 traditional VQA datasets.
On LVLM-specific benchmarks, our method significantly outperforms the baseline
and consistently surpasses existing fusion methods, demonstrating its effectiveness in
leveraging hierarchical visual information tailored to different tasks. On traditional
VQA benchmarks, our method achieves the highest scores on SQA, TextVQA, and
VizWiz. While our approach surpasses both the baseline and MMFuser on GQA, it
slightly underperforms DenseConnector. Since GQA is designed for compositional
reasoning, the textual questions contain complex linguistic structures , which
may challenge the sentence embedding model in accurately capturing task-relevant
information. This, in turn, may hinder the vision aggregator from deriving optimal
feature-weighting distributions.

### Visual Grounding and Video Understanding
<p align="center">
    <img src="images/other.png" width="90%"></a>
</p>

Performance comparison on visual grounding and video understanding benchmarks. For
visual grounding, the score of each benchmark is averaged across val and test splits. Bold values
indicate the best score in each column, while underlined values represent the second-best score.

Table presents the results for visual grounding, where our
method consistently achieves the highest scores across all benchmarks. This high-lights its effectiveness in enhancing object-region alignment across diverse image
contexts. The right section of table 5 reports the results for video understanding.
Apart from ActivityNet-QA, where our method achieves performance on par with
DenseConnector, it surpasses both the baseline and DenseConnector across all other
video benchmarks. This demonstrates that our method captures richer task-related
visual representations, which is crucial for modeling temporal dynamics and contex-tual dependencies across video frames.

### Visual Representation Visualization

To intuitively validate the impact of MMFuser on visual features, we present the input and output feature map visualizations for four example images in the figure.

<p align="center">
    <img src="images/attention.png" width="90%"></a>
</p>


## 安装

1. 克隆此存储库并切换到MMFuser文件夹
    ```bash
    git clone git@github.com:yuecao0119/MMFuser.git
    cd MMFuser
    ```

2. 安装程序包

    我们的项目基于[LLaVA-1.5](https://github.com/haotian-liu/LLaVA)并根据[LLaVA-1.5安装](https://github.com/haotian-liu/LLaVA?tab=readme-ov-file#install)创建相关环境。

    ```bash
    conda create -n MMFuser python=3.10 -y
    conda activate MMFuser
    pip install --upgrade pip  # enable PEP 660 support
    pip install -e .
    ```

3. 安装其他软件包

    安装Flash-Attention：

    ```bash
    pip install -e ".[train]"
    pip install flash-attn==2.3.6 --no-build-isolation
    ```

    在我们的项目中使用了[Deformation-DETR](https://github.com/fundamentalvision/Deformable-DETR/tree/main)中的可变形注意力机制。需要运行以下脚本编译CUDA算子：

    ```bash
    cd llava/model/multimodal_projector/deformable_attention/ops
    sh ./make.sh
    # unit test
    python test.py
    ```


## 训练

我们的训练管线和数据集直接取自[LLaVA-v1.5](https://github.com/haotian-liu/LLaVA). 训练包括两个阶段：
- *预训练*: 在~558K图像-文本对的子集上训练projector，以连接冻结的预训练视觉编码器和冻结的大语言模型。
    ```bash
    sh scripts/mmfuser/pertrain.sh
    ```
- *指令微调*: 利用多模态指令数据LLaVA-665K对整个MLLM进行微调。
    ```bash
    sh scripts/mmfuser/finetune.sh
    ```

## 评估

我们遵循[LLaVA-v1.5](https://github.com/haotian-liu/LLaVA/tree/main)进行评估。您应该下载[eval.zip](https://drive.google.com/file/d/1atZSBBrAX54yYpxtVVW33zFvcnaHeFPy/view?usp=sharing)，并将其解压缩到`./playground/data/eval`。请参考[Evaluation.md](./docs/Evaluation.md)准备数据。

然后，您可以在`scripts/v1_5/eval`中运行我们的评估脚本。

并且您可以通过以下代码进行模型推理：

```bash
sh scripts/mmfuser/inference.sh
```

## 👍 致谢

- [LLaVA](https://github.com/haotian-liu/LLaVA) ：我们基于该代码库改进。

## 🔒 许可证

- 该项目的大部分内容都是在[LICENSE](https://github.com/yuecao0119/MMFuser/blob/main/LICENSE)文件中的Apache 2.0许可证下发布的。
- 该服务是一个仅用于非商业用途的研究预览，受LLaMA的[License](https://github.com/facebookresearch/llama/blob/main/MODEL_CARD.md)模型和OpenAI生成的数据的[Terms of Use](https://openai.com/policies/terms-of-use)约束。如果您发现任何潜在的违规行为，请与我们联系。

## 引用

如果这项工作对您的研究有帮助，请考虑引用以下 BibTeX 条目。

```
@article{cao2024mmfuser,
  title={MMFuser: Multimodal Multi-Layer Feature Fuser for Fine-Grained Vision-Language Understanding},
  author={Cao, Yue and Liu, Yangzhou and Chen, Zhe and Shi, Guangchen and Wang, Wenhai and Zhao, Danhuai and Lu, Tong},
  journal={arXiv preprint arXiv:2410.11829},
  year={2024}
}
```
