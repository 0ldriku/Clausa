# Clausa

[English](README.md) · [日本語](README.ja.md) · 简体中文

Clausa 自动完成 utterance fluency 研究中的标注工作。输入口语录音，它会把标注输出为 TextGrid 文件（words、pauses、disfluencies、clauses）。它支持英语和日语，适用于第二语言和第一语言口语。

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/images/overview-dark.svg">
  <img alt="录音经 Clausa 分析后，得到含 words、pauses、disfluency、clauses 四个 tier 的 TextGrid，以及含九项指标的 uf_measures.csv" src="docs/images/overview-light.svg">
</picture>

名称来自 *clause* 和 *pausa*（拉丁语的 pause）。

## 下载

- Mac：[Clausa-1.0.2-macos-arm64.zip](https://github.com/0ldriku/Clausa/releases/download/v1.0.2/Clausa-1.0.2-macos-arm64.zip)
- Windows：[Clausa-1.0.2-windows-x64.zip](https://github.com/0ldriku/Clausa/releases/download/v1.0.2/Clausa-1.0.2-windows-x64.zip)
- Linux：[命令行](docs/command-line.md)（英文）

## 系统要求

| | Mac | Windows |
|---|---|---|
| 系统 | Apple 芯片，macOS 14 或更高版本 | 64 位 Windows 11，或装有 Microsoft Edge 的 Windows 10 |
| 显卡 | 内置 | NVIDIA，显存 8 GB 或以上；可选，没有显卡时约慢 3 倍 |
| 内存 | 16 GB 或以上 | 16 GB 或以上；没有 NVIDIA 显卡时 32 GB |
| 可用空间 | 20 GB | 30 GB |
| 网络 | 首次启动时需要，下载 8.3 GB | 首次启动时需要，下载 11.8 GB |

一切都在你的电脑上运行：Clausa 不收集也不发送任何内容，只在首次启动时联网下载模型。

## 安装

**Mac**

1. 解压下载的文件，把 `Clausa` 文件夹移到个人文件夹。
2. 双击 **First start**。如果 macOS 阻止它运行，打开系统设置 › 隐私与安全性，点击**仍要打开**。
3. 等窗口显示 Ready，然后点击 **Open Clausa**。

之后，打开 **Clausa.app** 即可。

**Windows**

1. 右键单击 zip 文件，选择**全部提取**。
2. 双击 `Clausa` 文件夹中的 **Clausa.exe**。如果 Windows 显示“Windows
   已保护你的电脑”，点击**更多信息**，再点击**仍要运行**。
3. 等窗口显示 Ready，然后点击 **Open Clausa**。

之后，打开 **Clausa.exe** 即可。

`Clausa` 文件夹不要放在 iCloud 云盘、OneDrive 或 Dropbox 中。如果安装中断，再启动一次即可：它会从中断处继续。

## 使用

1. 点击**新建项目**。为项目命名，并选择英语或日语。
2. 点击**添加录音**，或把音频文件（WAV、MP3、FLAC、OGG 或 M4A）拖到窗口上。
3. 点击**开始**。
4. 完成后，点击**打开 CSV** 查看指标。

项目保存在“文稿”文件夹（Windows 上为“文档”）中的 `Clausa projects` 里；可在**设置** › **更改文件夹…**中改为其他文件夹。

## 指标

| 指标 | 含义 |
|---|---|
| AR | 发声时间内每秒的 syllable 数 |
| SR | 每秒的 syllable 数，含 pause |
| MLR | 两次 pause 之间的 syllable 数 |
| MCPR | 每 syllable 的 clause 内 pause 数 |
| ECPR | 每 syllable 的 clause 末 pause 数 |
| PR | 每 syllable 的 pause 总数 |
| MCPD | clause 内 pause 的平均时长，单位为秒 |
| ECPD | clause 末 pause 的平均时长，单位为秒 |
| MPD | 全部 pause 的平均时长，单位为秒 |

[计算方法](docs/measures.md)（英文）。

## 准确性

与文章所用的版本相比，Clausa 已有改进。[详情](docs/improvements.md)（英文）。

要精确得到文章的结果，请参阅[复现论文](docs/reproducing-the-paper.md)（英文）。

## 局限性

Clausa 的结果由程序自动生成，可能有错误。报告之前请先检查，例如在 Praat 中打开标注文件。你对自己报告的结果负责。

在日语中，用数字写的数会被逐位读出。例如，25（にじゅうご，4 个 mora）只算作 2 个 mora。这会影响含数字的录音中使用 mora 的指标。

## 速度

在我们的三台电脑上测得。一个录音是一段 2-3 分钟的音频文件。十个录音是十段各 49 秒到 4 分钟的音频文件，总计约 20 分钟。

| 电脑 | 一个录音 | 十个录音 |
|---|---|---|
| Mac：M1 Max，64 GB | 1 分钟 | 3 分钟 |
| Windows：Ryzen 5 3600，32 GB，RTX 4060 Ti 8 GB | 1.5 分钟 | 5 分钟 |
| Windows，不使用显卡 | 4.5 分钟 | |
| Linux：Ryzen 9 9950X，128 GB，RTX 5090，命令行 | 35 秒 | 1.5 分钟 |

## 常见问题

### Clausa 输出什么？

每个录音的 TextGrid 文件，给出每个单词、pause、disfluency 和 clause 的起止时间。这是主要输出。

Clausa 还会据此计算九项[指标](#指标)，写入 `uf_measures.csv`。其他指标请根据 TextGrid 文件自行计算，这些文件位于每轮分析的 `clauses` 文件夹中。

### 能分析对话或访谈吗？

可以。Clausa 会转写并标注全部语音，但目前还不能区分说话人（说话人分离功能已在计划中）。有多位说话人时，指标会混合所有人的语音，请改用 TextGrid 文件。

### 什么算作 pause？

单词之间 0.25 秒及以上的静音。

### 修改 TextGrid 后能重新计算指标吗？

目前不能。我们计划在以后的版本中把指标计算作为独立的步骤。

### 为什么在另一台电脑上结果不同？

不同的电脑可能给出略有差异的转写。一项研究的所有录音请在同一台电脑上分析。

## 语言

Clausa 支持英语和日语。它的设计不限于这两种语言，但每一步都有专为它们编写的部分。增加一种新语言，需要支持该语言的语音识别器和对齐器、为它编写的 clause 规则，以及统计它的 syllable 的方法。第一步之后的所有步骤都建立在转写文本之上，所以识别器对学习者语音的准确率最为关键。Disfluency 检测器更难扩展：它只用英语和日语训练过。

我们计划让 clause 切分更容易适配新语言。

## 帮助

我们只在自己的几台电脑上测试过 Clausa。如果它在你的电脑上无法运行，或出了别的问题，请在 [GitHub](https://github.com/0ldriku/Clausa/issues) 上提交 issue，或发邮件到 oldriku@proton.me。请附上 `Clausa` 文件夹中的 `logs` 文件夹。日志包含录音中识别出的单词，请确认可以分享后再发送。

## 面向程序员

- [命令行](docs/command-line.md)（英文）
- [复现论文](docs/reproducing-the-paper.md)（英文）
- [开发者指南](docs/development.md)（英文）

## 许可证

代码采用 MIT 许可证。Clausa 的 disfluency 检测器（[oldriku/Clausa](https://huggingface.co/oldriku/Clausa)）采用 CC BY-NC-SA 4.0，仅限非商业用途。详见 [LICENSE.md](LICENSE.md)。

## 引用

如果你使用 Clausa，请引用这篇文章。文章已被录用，即将发表；年份、卷、页码和 DOI 将在正式发表后补充。

> Lu, J., & Sato, R. (in press). Automatic cross-lingual annotation of
> clause-based utterance fluency in L2 speech. *Research Methods in Applied
> Linguistics*.

```bibtex
@article{lu_sato_clausa,
  author  = {Lu, Jialiang and Sato, Reiko},
  title   = {Automatic cross-lingual annotation of clause-based utterance fluency in {L2} speech},
  journal = {Research Methods in Applied Linguistics},
  year    = {in press}
}
```
