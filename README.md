# 🎬 Semantic Movie Search with SBERT

Search ~35,000 movies by *meaning*, not keywords. Describe a plot in plain English and get back the movies whose stories are most similar, powered by Sentence-BERT embeddings and cosine similarity.

## Overview

Traditional keyword search fails when you don't remember a movie's title or the exact words in its synopsis. This project uses [Sentence-Transformers](https://www.sbert.net/) to turn every movie plot into a dense vector, so a query like *"A young boy discovers he has magical powers and attends a school for wizards"* can be matched against plots that are semantically close, even if they share no words.

## How It Works

1. **Load data**: read release year, title, and plot for each movie from the Wikipedia Movie Plots dataset.
2. **Embed plots**: encode every plot into a 384-dimensional vector using `all-MiniLM-L6-v2`.
3. **Embed query**: encode the user's search query with the same model.
4. **Compare**: compute cosine similarity between the query vector and all plot vectors.
5. **Rank**: return the top 5 movies with the highest similarity scores.

Think of it as placing every movie on a giant map where similar stories sit close together. Your query is dropped onto the same map, and we simply pick its nearest neighbours.

## Tech Stack

- Python 3.12
- [sentence-transformers](https://www.sbert.net/) (`all-MiniLM-L6-v2`)
- PyTorch
- pandas
- Jupyter Notebook

## Dataset

[Wikipedia Movie Plots](https://www.kaggle.com/datasets/jrobischon/wikipedia-movie-plots) (`wiki_movie_plots_deduped.csv`), containing 34,886 movies. Only three columns are used: `Release Year`, `Title`, and `Plot`.

> ⚠️ **The dataset is not included in this repository.** You need to download it yourself from Kaggle: <https://www.kaggle.com/datasets/jrobischon/wikipedia-movie-plots>

Download the CSV (`wiki_movie_plots_deduped.csv`) and place it in the project root.

## Getting Started

### 1. Clone the repository

### 2. Install dependencies

```bash
pip install pandas torch "sentence-transformers>=3.0" jupyter
```

> `model.similarity()` was introduced in Sentence-Transformers v3.0, so make sure you have a recent version.

### 3. Add the dataset

Download `wiki_movie_plots_deduped.csv` from [Kaggle](https://www.kaggle.com/datasets/jrobischon/wikipedia-movie-plots) and place it in the same folder as the notebook.

### 4. Run the notebook

```bash
jupyter notebook sbert_search_movies.ipynb
```

Edit the `query` variable to search for anything you like:

```python
query = "A young boy discovers he has magical powers and attends a school for wizards."
```

> Embedding all 34,886 plots is the slowest step. It runs much faster on a GPU, and you can save the embeddings (e.g. `torch.save(embeddings, "embeddings.pt")`) to avoid recomputing them.

## Example Output

```
Top 5 most similar movies:
Release Year: 2016
Title: The Student
Similarity Score: 0.5191
--------------------------------------------------
Release Year: 2014
Title: Dusari Goshta
Similarity Score: 0.4829
--------------------------------------------------
Release Year: 1959
Title: School Master
Similarity Score: 0.4741
...
```

## Limitations

- **Plot-only matching**: titles, genres, cast, and other metadata are ignored.
- **Input length cap**: `all-MiniLM-L6-v2` truncates text at 256 tokens, so very long plots are only partially embedded.
- **Modest scores**: for the sample query above, the results lean toward the "school" theme rather than the "magic" theme, which shows the small model can miss some nuance. The dataset's coverage also affects results.

## Ideas for Improvement

- Try larger models (e.g. `all-mpnet-base-v2`) and compare results.
- Chunk long plots and average or max-pool the chunk embeddings.
- Cache embeddings to disk for instant startup.
- Use a vector index such as FAISS for faster search at scale.
- Build a simple web UI with Gradio or Streamlit.
- Add filters by release year.

## Acknowledgements

- [Sentence-BERT](https://arxiv.org/abs/1908.10084) by Reimers & Gurevych
- [Wikipedia Movie Plots dataset](https://www.kaggle.com/datasets/jrobischon/wikipedia-movie-plots) on Kaggle
