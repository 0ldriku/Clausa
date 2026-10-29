# Clausa

Clausa measures how fluently learners speak English or Japanese. It analyses
recordings and gives nine fluency measures for each one in a spreadsheet, plus
annotation files you can open in [Praat](https://www.fon.hum.uva.nl/praat/).

## Download

- Mac: [Clausa-1.0.1-macos-arm64.zip](https://github.com/0ldriku/Clausa/releases/download/v1.0.1/Clausa-1.0.1-macos-arm64.zip)
- Windows: [Clausa-1.0.1-windows-x64.zip](https://github.com/0ldriku/Clausa/releases/download/v1.0.1/Clausa-1.0.1-windows-x64.zip)
- Linux: the [command line](docs/command-line.md)

## Requirements

| | Mac | Windows |
|---|---|---|
| System | Apple Silicon, macOS 14 or later | 64-bit Windows 11, or Windows 10 with Microsoft Edge |
| Graphics card | built in | NVIDIA with 8 GB or more; optional, without it about 3 times slower |
| Free space | 20 GB | 30 GB |
| Internet | for the first start, which downloads 8.3 GB | for the first start, which downloads 11.8 GB |

## Set up

**Mac**

1. Unzip the download and move the `Clausa` folder to your home folder.
2. Double-click **First start**. If macOS blocks it, open System Settings ›
   Privacy & Security and click **Open Anyway**.
3. Wait until the window says Ready, then click **Open Clausa**.

After that, open **Clausa.app**.

**Windows**

1. Right-click the zip and choose **Extract All**.
2. Double-click **Clausa.exe** in the `Clausa` folder. If Windows says it
   protected your PC, click **More info**, then **Run anyway**.
3. Wait until the window says Ready, then click **Open Clausa**.

After that, open **Clausa.exe**.

Keep the `Clausa` folder out of iCloud Drive, OneDrive and Dropbox. If the set
up stops, start it again: it continues where it stopped.

## Use

1. Click **New project**. Name it and choose English or Japanese.
2. Click **Add recordings** or drag audio files onto the window. Each file
   should hold one learner and only the learner's speech.
3. Click **Start**.
4. When it finishes, click **Open CSV** to see the measures.

Projects are saved in `Documents/Clausa projects`. If something goes wrong,
send us the `logs` folder from the `Clausa` folder.

## Measures

| Measure | Meaning |
|---|---|
| AR | syllables per second of speaking time |
| SR | syllables per second, pauses included |
| MLR | syllables between two pauses |
| MCPR | pauses inside clauses, per syllable |
| ECPR | pauses at the end of clauses, per syllable |
| PR | all pauses, per syllable |
| MCPD | mean length of pauses inside clauses, in seconds |
| ECPD | mean length of pauses at the end of clauses, in seconds |
| MPD | mean length of all pauses, in seconds |

[How they are computed](docs/measures.md).

## Accuracy

Clausa has been improved since the article.
[Details](docs/improvements.md).

To get the article's results exactly, see
[Reproducing the paper](docs/reproducing-the-paper.md).

## Limitations

Clausa's results are automatic and can contain errors. Check them before you
use them, for example by opening the annotation files in Praat. You are
responsible for the results you report.

In Japanese, a number written in digits is read one digit at a time. For
example, 25 (にじゅうご, 5 morae) counts as 2 morae. This changes the measures
that use morae in recordings with numbers.

## Speed

Measured on our three computers. One recording is an audio file of 2-3
minutes. Ten recordings are audio files of 49 s to 4 minutes each, about 20
minutes of audio in all.

| Computer | One recording | Ten recordings |
|---|---|---|
| Mac: M1 Max, 64 GB | 1 min | 3 min |
| Windows: Ryzen 5 3600, 32 GB, RTX 4060 Ti 8 GB | 1.5 min | 5 min |
| Windows without its graphics card | 4.5 min | |
| Linux: Ryzen 9 9950X, 128 GB, RTX 5090, command line | 35 s | 1.5 min |

## Notes

Analyse all recordings of one study on the same computer. Different computers
can give slightly different results.

## Future plans

Clausa will support more languages than English and Japanese. To make that
possible, we plan to make the clause segmentation customisable for each
language, or to let an LLM assistant segment the clauses.

## For programmers

- [Command line](docs/command-line.md)
- [Reproducing the paper](docs/reproducing-the-paper.md)
- [Developer guide](docs/development.md)

## Licence

The code is MIT. Clausa's disfluency detector
([oldriku/Clausa](https://huggingface.co/oldriku/Clausa)) is CC BY-NC-SA 4.0,
for non-commercial use. See [LICENSE.md](LICENSE.md).

## Citation

If you use Clausa, please cite the article. Details will be added on
publication.

> *Authors* (2026). Automatic cross-lingual annotation of clause-based
> utterance fluency in L2 speech. *Research Methods in Applied Linguistics*.
> https://doi.org/*to be added*

```bibtex
@article{clausa2026,
  author  = {Authors to be added},
  title   = {Automatic cross-lingual annotation of clause-based utterance fluency in {L2} speech},
  journal = {Research Methods in Applied Linguistics},
  year    = {2026},
  doi     = {to be added}
}
```
