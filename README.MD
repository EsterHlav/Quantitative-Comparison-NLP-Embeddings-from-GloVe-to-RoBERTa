# Embeddings in Natural Language Processing
## Quantitative Comparison on Downstream Tasks
#### Ester Hlav 2019, Deep Learning Research, Columbia University

Leveraging the power of PyTorch and open source libraries Flair and SentEval, we perform a fair quantitative comparison of natural language processing (NLP) embeddings -- from classical, contextualized architectures like Word2Vec and GloVe to recent state-of-the-art transformer-based embeddings such as BERT and RoBERTa.  We aim to benchmark them on downstream tasks by adding a top layer fine-tuning, while allowing for a certain architecture flexibility for each specific embedding. Using Sequential Bayesian Optimization, we fine-tune our models to achieve optimal perfomance and implement an extension of a hyperparameter optimization library (Hyperopt) to correct for inconsistences in sampling log base 2 and 10 uniform distribution for batch size and learning rate for training. Our results report transformer-based architectures (*i.e* BERT, RoBERTa) to be the leading models on all downstream NLP tasks used. 

Project includes a report with detailed description of embeddings, empirical tests and results, as well as a presentation split in two parts. Part 1 is a general overview of NLP Embeddings: How to represent textual input for neural networks; Part 2 reports empirical results of the quantitative embeddings comparison conducted with PyTorch and libraries SentEval, Flair and Hyperopt, on some of the most common NLP tasks (*e.g.* sentiment analysis, question answering, *etc.*).

## Part1: How to Represent Text for Neural Networks
* Pre-Deep Leearning Embedding methods, such as Eigenvalue decomposition of co-occurrence matrix
* [Word2Vec](https://papers.nips.cc/paper/5021-distributed-representations-of-words-and-phrases-and-their-compositionality.pdf) (2013)
* [GloVe](https://www.aclweb.org/anthology/D14-1162) (2014)
* [Character embeddings](https://arxiv.org/pdf/1508.06615.pdf) (2015)
* [CoVe](https://arxiv.org/pdf/1708.00107.pdf) (2017)
* [ELMo](https://arxiv.org/pdf/1802.05365.pdf) (2018)
* [BERT](https://arxiv.org/pdf/1810.04805.pdf) (2018)
* [XLM](https://arxiv.org/pdf/1901.07291.pdf) (2019)
* [XLNet](https://arxiv.org/pdf/1906.08237.pdf) (2019)
* [RoBERTa](https://arxiv.org/pdf/1907.11692.pdf) (2019)


Following animations provide visual representations of diffrences in sequence-to-sequence (seq2seq) architectures for specific embeddings:

#### Seq2seq Encoder-Decoder
![](https://github.com/EsterHlav/Quantitative-Comparison-NLP-Embeddings-from-GloVe-to-RoBERTa/blob/master/gif/S2StoTransformer_1v2.gif)

#### Seq2seq bi-directional Encoder-Decoder with Attention -- CoVe, ELMo
![](https://github.com/EsterHlav/Quantitative-Comparison-NLP-Embeddings-from-GloVe-to-RoBERTa/blob/master/gif/S2StoTransformer_2v2.gif)

#### Seq2seq Transformer -- BERT, XLM, RoBERTA
In Transformer animations, each appearing element represents a key/value, and the element with the largest presence represents the query element.
![](https://github.com/EsterHlav/Quantitative-Comparison-NLP-Embeddings-from-GloVe-to-RoBERTa/blob/master/gif/S2StoTransformer_3v2.gif)

#### Seq2seq Transformer-XL -- XLNet
![](https://github.com/EsterHlav/Quantitative-Comparison-NLP-Embeddings-from-GloVe-to-RoBERTa/blob/master/gif/S2StoTransformer_4v2.gif)


## Part2: Quantitative Comparison on Downstream Tasks

#### Results -- for Accurracy, Tasks and Architectures
![](https://github.com/EsterHlav/Quantitative-Comparison-NLP-Embeddings-from-GloVe-to-RoBERTa/blob/master/gif/results_table.png)

![](https://github.com/EsterHlav/Quantitative-Comparison-NLP-Embeddings-from-GloVe-to-RoBERTa/blob/master/gif/tasks.png)

![](https://github.com/EsterHlav/Quantitative-Comparison-NLP-Embeddings-from-GloVe-to-RoBERTa/blob/master/gif/architectures.png)

## Code

#### Dependencies
The code runs as an extension of the following libraries:
* [SentEval](https://github.com/facebookresearch/SentEval) from FAIR (modified version)
* [Flair](https://github.com/zalandoresearch/flair) from Zalando Research

#### Running the code

Install the dependencies.
```bash
pip install flair segtok bpemb deprecated pytorch_transformers allennlp
```

Run the bash script to get the data for downstream tasks.
```bash
data/downstream/get_transfer_data.bash
```

Launch the jupyter notebook [Embeddings_benchmark.ipynb](/) for training.
