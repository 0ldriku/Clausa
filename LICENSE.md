# Licence

- **Code**: MIT License, below.
- **Fonts** in `runner/app_ui/fonts/`, Geist and Geist Mono: SIL Open Font
  License 1.1, see `runner/app_ui/fonts/OFL.txt`.
- **The name Clausa** in the app and its icon is drawn from Cormorant Garamond,
  copyright 2015 the Cormorant Project Authors, SIL Open Font License 1.1.
- **Disfluency detector**, https://huggingface.co/oldriku/Clausa: CC BY-NC-SA
  4.0. It was trained on the Switchboard Dialog Act corpus, CC BY-NC-SA 3.0,
  whose non-commercial and share-alike terms carry over.
- **Filled-pause classifier**: not shared, because its training data includes
  a CC BY-ND 3.0 part of PodcastFillers. See `models/README.md`.

`training/disfluency_detector/scripts/train.py` aligns labels as the Hugging
Face Transformers token-classification example does, Apache License 2.0.

The other downloaded models keep their own licences: Qwen3-ASR and
Qwen3-ForcedAligner, Apache 2.0; GiNZA-Electra, MIT; the MFA models and
`en_ud_L1L2e_combined_trf`, see their pages.

## MIT License

Copyright (c) 2026 Jialiang Lu and Reiko Sato

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
