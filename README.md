# chinese-challenge-set-gender-bias
A Chinese challenge set for evaluating gender bias in Chinese–Portuguese automated translation (EAMT 2026)
# Chinese Challenge Set for Gender Bias Evaluation in Automated Translation

This repository contains the data and analysis code accompanying the paper:

> **A Chinese Challenge Set to Assess Gender Bias in Automated Translation**  
> [Author names] — EAMT 2026

## Contents

| File | Description |
|------|-------------|
| `chinese_pt_gender_bias_challenge_set_annotated.csv` | 495 Chinese source sentences, translations by 8 AT systems, and gender annotations |
| `regression.py` | Logistic regression analysis (Python) |

## Dataset

The challenge set comprises **495 source sentences** constructed from:
- 45 occupations drawn from the U.S. Bureau of Labor Statistics (BLS) 2024 Current Population Survey, classified into male-stereotyped (M), neutral (N), and female-stereotyped (F) categories
- 11 sentence templates varying gender cue type (baseline, explicit classifier, pronoun coreference) and syntactic structure

**Systems evaluated** (outputs included in CSV):
- Google Translate (accessed Feb 26, 2026)
- DeepL Translator (accessed Feb 26, 2026)
- GPT-5.2 / OpenAI (accessed Mar 1, 2026)
- Claude Sonnet 4.6 / Anthropic (accessed Mar 1, 2026)
- Ministral-14B-2512 / Mistral AI (accessed Mar 1, 2026)
- Qwen3.5-397B-A17B / Alibaba (accessed Mar 2, 2026)
- LLaMA-3.3-70B-Instruct / Meta (accessed Mar 1, 2026)
- DeepSeek V3.2 (accessed Mar 1, 2026)

**Annotation:** Gender of the translated occupation noun (M/F/unknown), produced via rule-based semi-automated procedure with manual adjudication of uncertain cases.

## Requirements

```bash
pip install pandas numpy statsmodels openpyxl
```

## Usage

```bash
python regression.py
```

The script loads `chinese_pt_gender_bias_challenge_set_annotated.csv`, reshapes the data into long format, and fits a binary logistic regression predicting correct gender accuracy across pronoun coreference conditions.

## License

This dataset and code are released under the [MIT License](LICENSE).

## Citation

If you use this dataset, please cite:

```bibtex
@inproceedings{[yourcitationkey],
  title     = {A Chinese Challenge Set to Assess Gender Bias in Automated Translation},
  author    = {[Author names]},
  booktitle = {Proceedings of },
  year      = {2026}
}
```
