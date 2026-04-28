# Data

This repository does not ship the TalentCLEF 2026 Task A dataset.

Download the official public data from the TalentCLEF 2026 organizers and place it under this directory with the following layout:

```text
data/
├── development/
│   ├── en/
│   └── es/
└── test/
    ├── en/
    └── es/
```

Expected split folders:

- `data/development/en`
- `data/development/es`
- `data/test/en`
- `data/test/es`

The scripts in the repository assume this exact structure.
