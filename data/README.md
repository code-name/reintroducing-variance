# data/

Working folders for your own data. None of them is tracked by git.

| Folder | Contents |
|---|---|
| `app/` | The rating app's SQLite database and session key (`benchmark.rating`). Holds rater data. |
| `release/` | Release exports written by `python -m benchmark.rating.cli export --release`. Review before publishing. |
| `raw/`, `processed/` | Free for your own source files and tables. |

A data folder for the AI judges and the diagnostic can live anywhere; see the README, "Run it on your own ideas", and `examples/alternative_uses/`.
