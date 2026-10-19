# Personal Meeting

PDF documents for one-on-one meetings, one folder per meeting:

```
03_personal_meeting/
└── YYYY-MM-DD/
    ├── <name>.tex        # LaTeX source
    ├── <name>.pdf        # compiled document
    ├── .latexmkrc        # optional, e.g. LuaLaTeX for Chinese text
    └── figures/          # optional
```

Only `.tex`, `.pdf`, `.latexmkrc` and `figures/` are tracked; LaTeX build files are git-ignored.
