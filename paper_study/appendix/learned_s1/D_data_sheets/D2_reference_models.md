**Validation cross-entropy against references**

| model | nats/token | perplexity | bits/byte |
|---|---|---|---|
| uniform over the vocabulary | 9.7041 | 16,384.0 | 3.2258 |
| unigram (training frequencies) | 7.2869 | 1,461.1 | 2.4223 |
| interpolated bigram | 5.0957 | 163.3 | 1.6939 |
| this model, best (update 3,000) | 4.2822 | 72.4 | 1.4235 |
