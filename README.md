<p align="center">
    <h1 align="center">
        <b>SUQL (Structured and Unstructured Query Language)</b>
        <br>
        <a href="https://arxiv.org/abs/2311.09818">
            <img src="https://img.shields.io/badge/cs.CL-2311.09818-b31b1b" alt="arXiv">
        </a>
        <a href="https://github.com/stanford-oval/suql/stargazers">
            <img src="https://img.shields.io/github/stars/stanford-oval/suql?style=social" alt="Github Stars">
        </a>
        <a href="https://pypi.org/project/suql/">
            <img alt="PyPI version" src="https://img.shields.io/pypi/v/suql.svg"/>
        </a>
    </h1>
</p>
<p align="center">
    Conversational Search over Structured and Unstructured Data with LLMs
</p>
<p align="center">
    Online demo:
    <a href="https://yelpbot.genie.stanford.edu" target="_blank">
        https://yelpbot.genie.stanford.edu
    </a>
    <br>
</p>


# What is SUQL

SUQL stands for Structured and Unstructured Query Language. It augments SQL with several important free text primitives for a precise, succinct, and expressive representation. It can be used to build chatbots for relational data sources that contain both structured and unstructured information. Similar to how text-to-SQL has seen [great success](https://python.langchain.com/docs/use_cases/qa_structured/sql), SUQL can be used as the semantic parsing target language for hybrid databases, for instance, for:

![An example restaurant relational database](https://github.com/stanford-oval/suql/blob/main/figures/figure1.png?raw=true)

Several important features:

- SUQL seamlessly integrates retrieval models, LLMs, and traditional SQL to deliver a clean, effective interface for hybrid data access;
    - It utilizes techniques inherent to each component: retrieval model and LM for unstructured data and relational SQL for structured data;
- Index of free text fields built with [faiss](https://github.com/facebookresearch/faiss), natively supporting all your favorite dense vector processing methods, e.g. product quantizer, HNSW, etc.;
- A series of important optimizations to minimize expensive LLM calls;
- Scalability to large databases with PostgreSQL;
- Support for general SQLs, e.g. JOINs, GROUP BYs.

## The answer function

One important component of SUQL is the `answer` function. `answer` function allows for constraints from free text to be easily combined with structured constraints. Here is one high-level example:

![An example for using SUQL](https://github.com/stanford-oval/suql/blob/main/figures/figure2.png?raw=true)

For more details, see our paper at https://arxiv.org/abs/2311.09818.

# Installation / Usage tutorial

There are two main ways of installing the SUQL library.

## Install from `pip`

Ideal for integrating the SUQL compiler in a larger codebase / system. See [install_pip.md](https://github.com/stanford-oval/suql/blob/main/docs/install_pip.md) for details.

## Install from source

Ideal for using this repo to build a SUQL-powered conversational interface to your data out-of-the-box, like the one for https://yelpbot.genie.stanford.edu discussed in the paper. See [install_source.md](https://github.com/stanford-oval/suql/blob/main/docs/install_source.md) for details.

## Agent tutorial

Check out [conv_agent.md](https://github.com/stanford-oval/suql/blob/main/docs/conv_agent.md) for more information on best practices for using SUQL to power your conversational agent.

# Release notes

Check [release_notes.md](https://github.com/stanford-oval/suql/blob/main/docs/release_notes.md) for new release notes.

# Bugs / Contribution

If you encounter a problem, first check [known_issues.md](https://github.com/stanford-oval/suql/blob/main/docs/known_issues.md). If it is not listed there, we welcome Issues and/or PRs!

# Paper results

To replicate our results on HybridQA and restaurants in our paper, see [paper_results.md](https://github.com/stanford-oval/suql/blob/main/docs/paper_results.md) for details.

# Citation

If you find this work useful to you, please consider citing us.
```
@inproceedings{liu-etal-2024-suql,
    title = "{SUQL}: Conversational Search over Structured and Unstructured Data with Large Language Models",
    author = "Liu, Shicheng  and
      Xu, Jialiang  and
      Tjangnaka, Wesley  and
      Semnani, Sina  and
      Yu, Chen  and
      Lam, Monica",
    editor = "Duh, Kevin  and
      Gomez, Helena  and
      Bethard, Steven",
    booktitle = "Findings of the Association for Computational Linguistics: NAACL 2024",
    month = jun,
    year = "2024",
    address = "Mexico City, Mexico",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.findings-naacl.283",
    pages = "4535--4555",
    abstract = "While most conversational agents are grounded on either free-text or structured knowledge, many knowledge corpora consist of hybrid sources.This paper presents the first conversational agent that supports the full generality of hybrid data access for large knowledge corpora, through a language we developed called SUQL ($\textbf{S}$tructured and $\textbf{U}$nstructured $\textbf{Q}$uery $\textbf{L}$anguage). Specifically, SUQL extends SQL with free-text primitives (${\small \text{SUMMARY}}$ and ${\small \text{ANSWER}}$), so information retrieval can be composed with structured data accesses arbitrarily in a formal, succinct, precise, and interpretable notation. With SUQL, we propose the first semantic parser, an LLM with in-context learning, that can handle hybrid data sources.Our in-context learning-based approach, when applied to the HybridQA dataset, comes within 8.9{\%} Exact Match and 7.1{\%} F1 of the SOTA, which was trained on 62K data samples. More significantly, unlike previous approaches, our technique is applicable to large databases and free-text corpora. We introduce a dataset consisting of crowdsourced questions and conversations on Yelp, a large, real restaurant knowledge base with structured and unstructured data. We show that our few-shot conversational agent based on SUQL finds an entity satisfying all user requirements 90.3{\%} of the time, compared to 63.4{\%} for a baseline based on linearization.",
}
```
