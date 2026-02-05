# Lost in Plot: Contrastive Learning for Tip-of-the-Tongue Movie Retrieval

## Abstract

Humans often recall a movie by fragments usually using bits of plot, emotion, or striking visuals rather than its exact title. Traditional search engines and keyword-based retrievers struggle with such vague queries, while generative models can hallucinate plausible but incorrect titles. 

This work proposes a dense retrieval approach that leverages contrastive learning to align user descriptions and movie metadata in a shared semantic space. We evaluate this system on a public dataset derived from TMDb and IMDb and compare it with two strong baselines: few-shot prompting of large language models and the vanilla encoder before contrastive fine-tuning.

## Quick Start

```bash
pip install -r requirements.txt
python -m trainer.task_infoNCE --data-path movies.csv --epochs 8 --model-dir ./output