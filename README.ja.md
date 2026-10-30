# Clausa

[English](README.md) · 日本語 · [简体中文](README.zh-CN.md)

Clausaは、第二言語の発話の録音から、clauseを単位とするutterance fluencyの指標を作ります。音声を書き起こし、各単語の位置を音声に合わせ、発話をclauseに分け、書き起こしに漏れたfilled pauseを含むdisfluencyを見つけます。そのうえで、録音ごとにspeedとpauseに関する9つの指標を算出します。指標には、clause内のpauseとclause末のpauseを分けたものも含まれます。指標はスプレッドシートに、アノテーションはPraatで開けるTextGridファイルに保存されます。今のところ、Clausaは英語と日本語に対応しています。

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/images/overview-dark.svg">
  <img alt="録音をClausaで分析すると、words、pauses、disfluency、clausesの4つのtierを持つTextGridと、9つの指標を収めたuf_measures.csvができます" src="docs/images/overview-light.svg">
</picture>

名前は、*clause*と、ラテン語でpauseを表す*pausa*から取りました。

## ダウンロード

- Mac: [Clausa-1.0.1-macos-arm64.zip](https://github.com/0ldriku/Clausa/releases/download/v1.0.1/Clausa-1.0.1-macos-arm64.zip)
- Windows: [Clausa-1.0.1-windows-x64.zip](https://github.com/0ldriku/Clausa/releases/download/v1.0.1/Clausa-1.0.1-windows-x64.zip)
- Linux: [コマンドライン](docs/command-line.md)（英語）

## 必要なもの

| | Mac | Windows |
|---|---|---|
| システム | Appleシリコン搭載のMac、macOS 14以降 | 64ビットのWindows 11、またはMicrosoft Edgeの入ったWindows 10 |
| グラフィックカード | 内蔵 | メモリ8GB以上のNVIDIA製。なくても動きますが、約3倍の時間がかかります |
| メモリ | 16GB以上 | 16GB以上。NVIDIA製のグラフィックカードがない場合は32GB |
| 空き容量 | 20GB | 30GB |
| インターネット | 初回の起動時。8.3GBをダウンロードします | 初回の起動時。11.8GBをダウンロードします |

処理はすべてお使いのコンピューターの中で行われ、Clausaは何も収集・送信しません。インターネットを使うのは、初回の起動時にモデルをダウンロードするときだけです。

## セットアップ

**Mac**

1. ダウンロードしたzipを展開し、`Clausa`フォルダーをホームフォルダーに移動します。
2. **First start**をダブルクリックします。macOSにブロックされたら、システム設定 › プライバシーとセキュリティを開き、**このまま開く**をクリックします。
3. ウィンドウにReadyと表示されるまで待ち、**Open Clausa**をクリックします。

以後は**Clausa.app**を開いてください。

**Windows**

1. zipを右クリックし、**すべて展開**を選びます。
2. `Clausa`フォルダーの**Clausa.exe**をダブルクリックします。WindowsがPCを保護したと表示されたら、**詳細情報**をクリックし、次に**実行**をクリックします。
3. ウィンドウにReadyと表示されるまで待ち、**Open Clausa**をクリックします。

以後は**Clausa.exe**を開いてください。

`Clausa`フォルダーは、iCloud Drive、OneDrive、Dropboxの中に置かないでください。セットアップが途中で止まったら、もう一度始めてください。止まったところから続きます。

## 使い方

1. **新規プロジェクト**をクリックします。名前を付け、英語か日本語を選びます。
2. **録音を追加**をクリックするか、音声ファイルをウィンドウにドラッグします。1つのファイルには、学習者1人の発話だけを入れてください。
3. **開始**をクリックします。
4. 終わったら**CSVを開く**をクリックすると、指標が見られます。

プロジェクトは「書類」フォルダー（Windowsでは「ドキュメント」）の`Clausa projects`に保存されます。保存先は**設定** › **フォルダーを変更…**で変えられます。

## 指標

| 指標 | 意味 |
|---|---|
| AR | 発話時間1秒あたりのsyllable数 |
| SR | pauseを含めた1秒あたりのsyllable数 |
| MLR | pauseとpauseの間のsyllable数 |
| MCPR | clause内のpauseの数（syllableあたり） |
| ECPR | clause末のpauseの数（syllableあたり） |
| PR | すべてのpauseの数（syllableあたり） |
| MCPD | clause内のpauseの平均長（秒） |
| ECPD | clause末のpauseの平均長（秒） |
| MPD | すべてのpauseの平均長（秒） |

[計算方法](docs/measures.md)（英語）。

## 精度

Clausaは論文の時点から改良されています。[詳細](docs/improvements.md)（英語）。

論文の結果をそのまま得るには、[論文の再現](docs/reproducing-the-paper.md)（英語）を参照してください。

## 制限

Clausaの結果は自動で作られたもので、誤りを含むことがあります。報告する前に、たとえばアノテーションファイルをPraatで開くなどして確認してください。報告する結果の責任は利用者にあります。

アノテーションファイルには、Clausaが見つけた各単語、pause、disfluency、clauseの開始時刻と終了時刻が入っています。これをもとに独自の指標を計算することもできます。

日本語では、数字で書かれた数は1桁ずつ読まれます。たとえば25（にじゅうご、4 morae）は2 moraeと数えられます。数を含む録音では、moraを使う指標がこの影響を受けます。

## 速度

私たちの3台のコンピューターで測りました。録音1件は2〜3分の音声ファイルです。録音10件は49秒から4分までの音声ファイルで、合計約20分です。

| コンピューター | 録音1件 | 録音10件 |
|---|---|---|
| Mac: M1 Max、64GB | 1分 | 3分 |
| Windows: Ryzen 5 3600、32GB、RTX 4060 Ti 8GB | 1.5分 | 5分 |
| Windows、グラフィックカードなし | 4.5分 | |
| Linux: Ryzen 9 9950X、128GB、RTX 5090、コマンドライン | 35秒 | 1.5分 |

## 注意

1つの研究の録音は、すべて同じコンピューターで分析してください。コンピューターが違うと、結果がわずかに違うことがあります。

## 対応言語

Clausaは英語と日本語に対応しています。設計はこの2言語に限ったものではありませんが、各ステップにはこの2言語向けに書かれた部分があります。新しい言語には、その言語に対応した音声認識器とアライナー、その言語用に書いたclauseの規則、syllableの数え方が必要です。最初のステップより後はすべて書き起こしの上に成り立つので、学習者の発話に対する認識器の精度が最も重要です。Disfluency検出器は拡張が難しく、英語と日本語だけで学習されています。

Clause分割を新しい言語に合わせやすくする予定です。

## 困ったときは

Clausaは私たちの数台のコンピューターでしか試していません。お使いのコンピューターで動かない場合や、ほかに問題が起きた場合は、[GitHub](https://github.com/0ldriku/Clausa/issues)でissueを作成するか、oldriku@proton.meにメールしてください。その際、`Clausa`フォルダーの中の`logs`フォルダーを添付してください。ログには録音から認識された単語が含まれるので、共有してよい場合にだけ送ってください。

## プログラマー向け

- [コマンドライン](docs/command-line.md)（英語）
- [論文の再現](docs/reproducing-the-paper.md)（英語）
- [開発者ガイド](docs/development.md)（英語）

## ライセンス

コードはMITライセンスです。Clausaのdisfluency検出器（[oldriku/Clausa](https://huggingface.co/oldriku/Clausa)）はCC BY-NC-SA 4.0で、非営利利用に限ります。[LICENSE.md](LICENSE.md)を参照してください。

## 引用

Clausaを使用した場合は、論文を引用してください。論文は印刷中で、年、巻、ページ、DOIは出版時に追加します。

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
