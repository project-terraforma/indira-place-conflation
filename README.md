# Project C: Places conflation with scalable language models
### Problem description
Determine if scalable language models, such as small LLMs or static embedding models, can outperform our existing matching approach.

### Proposed proof-of-concept (POC)
We will conduct a comparative analysis of various language models against Overture's current places matcher. The project will test multiple models and methodologies on a dataset of both matched and unmatched places. The primary goal is to identify a language model that can "beat our matcher" in performance while also providing a better price-to-performance ratio.

### Key questions
1. Can small LLMs outperform Overture’s current place matcher?
2. Which models offer the best price-performance balance?
3. What methods have better performance in language model-based matching?
4. How do models handle different countries, languages, and missing data?

### Key deliverables
* A comparative analysis report detailing the performance of each tested language model against the current Overture matcher
* An evaluation of the price-to-performance ratio for each model
* A recommendation on whether a language model should replace the current conflation model

### Set-Up
Python Libraries Installation:
``` Bash
conda install -c conda-forge lonboard overturemaps-py geopandas pandas shapely
```

### To-do List

