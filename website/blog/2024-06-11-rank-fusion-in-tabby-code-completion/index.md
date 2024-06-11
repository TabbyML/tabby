---
authors: [meng]
tags: [quality, vector embedding]
---

# Rank Fusion for improved Code Context in Tabby

## Introduction

Tabby has made significant advancements in its code context understanding with the introduction of a semantic relevance score (via vector embedding) and rank fusion in version [0.12](https://github.com/TabbyML/tabby/releases/tag/v0.12.0). These enhancements have transformed the way Tabby ranks source code context, resulting in more accurate context for feeding into LLM.

## From BM25 to Rank Fusion

Tabby's initial approach to ranking involved the use of the BM25 algorithm, as described in [Repository context for LLM assisted code completion](/blog/2023/10/16/repository-context-for-code-completion/). This algorithm indexed source code in chunks, which served as the basis for code completion and Q&A. In the latest release, Tabby has augmented this approach with a semantic relevance score calculated from embedding vector distances. This dual scoring system necessitated the implementation of a rank fusion technique to effectively combine these disparate ranks.

## The Mechanics of Reciprocal Rank Fusion

The RRF method adopted by Tabby is a well-established technique in information retrieval. It merges multiple rank lists to produce a single, more accurate ranking. In Tabby, the RRF is applied as follows:

```python title="derived from https://www.elastic.co/guide/en/elasticsearch/reference/current/rrf.html"
score = 0.0
for q in queries:
    if d in result(q):
        score += 1.0 / ( k + rank( result(q), d ) )
return score

# where:
# k is a constant, currently set to 60 in Tabby
# q is a query within the set of queries
# d is a document found in the result set of q
# result(q) is the result set for query q
# rank( result(q), d ) is the ordinal rank of document d within result(q)
```

By introducing the semantic relevance score and rank fusion, Tabby can now provide more accurate code suggestions that are contextually relevant to the user's current work.

For developers using Tabby, the enhanced ranking system requires no additional configuration beyond the repository context setup in the admin UI. The indexing process now includes the computation of embedding vectors, which, while slightly extending the initial indexing time, is mitigated by caching vectors between commits to optimize performance.

*Try **Explain Code** in [Code Browser](https://demo.tabbyml.com)*

![Repository Context Triggered](./repository-context-triggered.png)

## Conclusion

By leveraging a combination of BM25 and semantic relevance scores, Tabby delivers more accurate and contextually appropriate suggestions, streamlining the development process.

As Tabby continues to evolve, users can anticipate ongoing improvements designed to bolster productivity and enrich the coding experience.
