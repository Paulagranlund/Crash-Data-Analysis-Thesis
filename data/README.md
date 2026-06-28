# Data

## `raw/` (not committed)

The real Vejdirektoratet (VD) crash extract, exported as `.xlsx` files with the
header on row 3. Access-restricted, so this folder is gitignored and ships
empty. `src/data_load.py` reads every `.xlsx` here, renames the Danish columns
to English, derives the main crash-situation class, and drops rows without a
narrative or with fewer than three words.

## `synthetic/` (committed)

A fake dataset with the same schema, used so the pipeline runs without access to
the real data. It is what `DATA_FOLDER` in `src/config.py` points to by default.

## Schema (after loading)

| column                      | description                                  |
|-----------------------------|----------------------------------------------|
| accident_date               | date of the accident                         |
| report_category             | accident report category                     |
| encoded_accident_situation  | numeric situation code (e.g. 201)            |
| accident_situation          | situation description                        |
| police_narrative            | free-text police description                 |
| year                        | year of the accident                         |
| accident_id                 | unique accident identifier                   |
| main_situation_class        | derived class, code // 100 (e.g. 201 -> 2)   |
