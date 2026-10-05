# Data card

**Dataset.** wikitext-103 (wikitext-103-raw-v1, Salesforce/wikitext).
WikiText (Merity et al., 2016, "Pointer Sentinel Mixture Models") is a collection of English Wikipedia articles
verified as Good or Featured. The raw variant keeps original casing, punctuation and numbers without `<unk>`
substitution.

**Licence.** Creative Commons Attribution-ShareAlike (CC BY-SA), as Wikipedia text. Attribute Wikipedia
contributors and Merity et al. when redistributing derived material.

**Composition.**

| split | rows | words | UTF-8 MB | BPE tokens |
|---|---|---|---|---|
| train | 1,801,350 | 103,227,021 | 539.9 | 125,896,517 |
| validation | 3,760 | 217,646 | 1.1 | 263,951 |
| test | 4,358 | 245,569 | 1.3 | 301,624 |

Words follow the WikiText convention: whitespace-separated words plus one end-of-line per line.

**Integrity.** Every downloaded file was verified against the SHA-256 published by the source repository:
- `train-00000-of-00002.parquet`: `74da360f23826045b3e6ac6375411fdb15f003030aa74f2596ed08b857cb9212`
- `train-00001-of-00002.parquet`: `ba090ac30dbf5461e8dcbdd1a1b8e6f3cf9c2c756d64f0c1220450acd514f720`
- `validation-00000-of-00001.parquet`: `204929b7ff9d6184953f867dedb860e40aa69c078fc1e54b3baaa8fb28511c4c`
- `test-00000-of-00001.parquet`: `5f1bea067869d04849c0f975a2b29c4ff47d867f484f5010ea5e861eab246d91`

**Preprocessing.** Each split's text is its dataset rows joined as lines (empty rows are blank lines). A
16,384-token byte-level BPE (GPT-4 split pattern) was learned from the training split only and
used to tokenize every split. Data fingerprint `15ec0fd6ef24da743d6f24821a7914b360e94c17cdc044743ae2c832bb0d7964`.

**Uses here.** Training (train split), model selection and monitoring (validation split); the test split is
reserved for a final evaluation.
