# Data Management and Ethics

Document sources, licenses, and handling rules for each dataset. Avoid committing raw data if it includes sensitive or proprietary content.

## Layout
- `raw/`: immutable copies of original data (store paths or download scripts instead of large files)
- `processed/`: cleaned and transformed datasets generated from `src/data` pipelines
- `external/`: data pulled from third-party APIs or collaborators

## Checklist
- Record source URLs, access dates, and licenses
- Describe preprocessing steps and scripts used
- Track consent/IRB considerations if applicable
- Add `.gitignore` rules for large or sensitive files
