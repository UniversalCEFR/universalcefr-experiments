# UniversalCEFR Experiments

The preprocessing scripts for `EFCAMDAT`, `APA-LHA`, `BEA Shared Task 2019 Write and Improve`, and `DEPlain` are found in `preprocessing` folder for transforming these raw datasets to UniversalCEFR format. Feel free to download and edit the scripts (and edit as you need to). We are unable to upload a public version of the derivations of these datasets as stipulated in their Terms of Use, however, you are free to download them from their respective websites. Check the links in the [website](https://universalcefr.github.io/).

## Feature extraction

The [feature_extraction](feature_extraction/) folder provides the linguistic
feature pipeline for [UniversalCEFR (EMNLP 2025)](https://aclanthology.org/2025.emnlp-main.491/).
It includes installation instructions, configurable CPU/GPU extraction, synthetic
examples, tests, and a guide to the paper's 100-feature configuration.
