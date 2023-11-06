---
authors: [ gyxlucy ]

tags: [tech design]

---
# Evaluating Coding LLMs

Tabby offers open-source GitHub Copilot alternative solution with easy setup and self-host options. We embrace an open ecosystem to support major open source coding LLMs (e.g. StarCoder, CodeLlama, WizardCoder, etc.), and allows users to easily plug in their own proprietary models to Tabby. We firmly believe in the continuous advancement in coding LLMs, yet we need quantative measurements to guide the direction of product improvement, and help developers decide their model of choice. 

Evaluation coding LLMs has also been a hot topic in academics. Many different metrics targeting different coding tasks have been proposed over the past year. At Tabby, we prioritize on metrics that best resembles real world development workflow, and of course, the metrics should be constructed with non-biased data sources. In this blogpost, we will discuss our thoughts for desired code completion benchmarks, and also review latest academic progress in this area.


## What we are looking for in coding LLM evaluations?

### Scientific and Relevant Setup
The top thing of our interest is what task this metric is evaluating. To be honest, most existing coding LLM evaluations focus on function-level code generation - given a docstring or a function signature at most, the LLM is expected to generate the entire function body. The most widely adopted evaluation of such is **HumanEval** introduced by OpenAI in [Evaluating Large Language Models Trained on Code](https://arxiv.org/pdf/2107.03374.pdf) in July 2021. HumanEval is a hand-crafted dataset, consisting of 164 Python programming problems with unit tests. An example task looks like:

```python
from typing import List 

def below_zero(operations: List[int]) -> bool: 
    
    """ 
    You're given a list of deposit and withdrawal operations on a bank account that starts with zero balance. Your task is to detect if at any point the balance of account fallls below zero, and at that point function should return True. Otherwise it should return False.
    
     >>> below_zero([1, 2, 3]) False 
     
     >>> below_zero([1, 2, -4, 5]) True 
     
    """
```

It was a pioneer research effort, but now suffers from some unfortunate drawbacks:

1. ***Data is likely contaminated.*** HumanEval dataset has been around for over two years and it has been discussed and documented widely online. The latest coding LLMs are likely to have included its test data in training data crawling, which would make the evaluation no longer valid.

2. ***Trivial coding questions that aren't mimicing real engineering setup.*** HumanEval inclues mostly LeetCode's interview-style questions, where they include a single function for LLMs to fill in the body. In a more realistic corporate setup, developers often add code in multiple files in a single PR, and constantly refer to functions implemented in other files. These are indeed more interesting yet challenging tasks for LLMs to perform, but are critical scenarios for AI coding assitants to land in enterprises.

3. ***Limited coverage in programming languages.*** This one is obvious as HumanEval only includes Python code. We ❤️ all programming languages!

Here are what we think a trustworthy evaluation setup should cover:

1. ***Non-trivial code.*** Definitely no more Leetcode-style coding questions! The ideal evaluation should target projects with substantial engineering complexity. Evidences like lines of code, number of files, or number of contributors could serve as good indicators to estimate the code complexity.

2. ***Cross-file references.*** This is a key factor to differentiate a more reliable and practical evaluation from something that only scratches the surface of the coding world. Engineers do not code in silo, but are greatly encouraged to reuse a function or API implemented by colleagues. (💡BTW, Tabby enabled [repository-level context support](https://tabby.tabbyml.com/blog/2023/10/16/repository-context-for-code-completion) in October, a feature exactly to get smarter coding suggestions in such scenario.)

3. ？？？？？？？

### Ease and Cost to Run
The ease and cost to run evaluations is directly correlated to the number of models we can evaluate, and the frequency we can afford to update the results (in the case of refreshed evaluation date, for example). There are efforts to leverage crowdsourcing to rate the quality of LLM responses (e.g. [Glaive arena](https://arena.glaive.ai/)) which excels at receiving high-quality human feedbacks and provides valuable insights to understand user behaviors. However it's harder to scale crowdsourcing ratings and takes longer to receive results. We are iterating quickly on Tabby, and decided that scalability and ease are critical to us now.

### Data Quality and Inclusion
The data quality is critical to maintain the legitimacy of such evaluation. Here's what's important for evaluation data:

1. ***Train/Eval Data Split***. It's one of the most important concepts in your *Machine Learning 101* course. Yet often times it gets so basic that folks neglect the challenges to ensure it in real-world applications over time. For example, HumanEval started as a manually drafted dataset to firmly ensure the data separation. Nevertheless over time, it still faces data contamination issue.

2. ***Evaluation Quality***. One example to illustrate this aspect is the [***HumanEvalPlus***](https://github.com/evalplus/evalplus) metric: Researchers noticed that test cases in HumanEval tasks (on average 7.7 tests per problem) aren't enough to guarantee the correctness of the generated code (e.g. a wrong implementation could still pass all existing tests), and thus augmented test cases in HumanEval benchmark by 80x. Understanding the quality of the evaluation is important for developing a fair sense of the true model performance. We also encourage continuous efforts in improving evaluation quality!💪🏻

![human-eval-plus](./human-eval-plus.png)

3. ***Data Inclusion / Coverage***. In the case of coding, inclusion includes efforts like increasing the support of different programming languages. In practice, choosing the reasonable ratio of each programming language is also tricky yet important.


## Highlights of recent research innovations
In this section, we showcase a few recent research work of from the academics toward building reliable and sound evaluations for coding LLMs. Following these papers, we observe a growing emphasize in evaluating coding LLMs with repository-level context, which indeed aligns with what we have been looking for.

### CrossCodeEval
CrossCodeEval 

### RepoBench

### RepoCoder


