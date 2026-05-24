# Pathological Upgrade Risk Calculator

This repository contains the public reviewer-facing web calculator for estimating
the risk of pathological upgrade after ESD.

## Public calculator

Main Streamlit file:

```text
app.py
```

The app accepts structured clinical variables and a patient-level image-derived
CNN score as inputs. It does not require a `.pth` file and does not perform
online image inference.

## Required repository files

Recommended minimal file structure:

```text
app.py
requirements.txt
README.md
Training_Cohort_Patients.csv
Internal_Validation_Patients.csv
External_Validation_Patients.csv
```

Optional files such as `run.py`, `分组总表.xlsx`, `训练集.xlsx`, and `验证集.xlsx`
are not required by the public app.

## Streamlit Cloud settings

Set the main file path to:

```text
app.py
```

## Requirements

Use the following `requirements.txt`:

```text
streamlit
pandas
numpy
openpyxl
```

## CNN score

The public app does not calculate the CNN score from images. The CNN score should
be generated offline using the locked image model and entered into the calculator
as the patient-level image-derived CNN score.
