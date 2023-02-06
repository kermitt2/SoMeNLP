This is a fork of SoMeNLP for experimenting and benchmarking exercices.

After following the instructions of the original project readme (below, note: although not used with SciBERT, generating the features is necessary to avoid raising errors), train a single task model using the dev set to select best model and without test set (as defined in the data configuration):

```console
train_model --model-config configurations/PMC/NER/gold_SciBERT_final.json --data-config configurations/PMC/NER/gold_data_SciBERT_final_2.json
```

The modified data config file `gold_data_SciBERT_final_2.json` should correspond to the usual training scenario (train set and dev set to select the best model). 

The config file of the saved model needs to be edited to add the path of the model weights (it's not updated when saving the weights). If the trained model is saved under the "save" path `/media/lopez/store/save`, edit the config file, for instance `/media/lopez/store/save/Gold-SciBERT/04-02-2023_21-50-57/model_conf.json` and the `checkpoint` attribute:

```
        "checkpoint": {
            "model": "/media/lopez/store/save/Gold-SciBERT/04-02-2023_21-50-57/ep_39_step_0_perf_1.pth",
            "save_dir": "/media/lopez/store/save/Gold-SciBERT/04-02-2023_21-50-57/",
            "log_dir": "/media/lopez/store/save/Gold-SciBERT/04-02-2023_21-50-57/"
        },
```

Then, to benchmark a model against test set (as defined in the data configuration):

```console
bin/benchmark --model-config /media/lopez/store/save/Gold-SciBERT/04-02-2023_21-50-57/model_conf.json --data-config configurations/PMC/NER/gold_data_SciBERT_final_2.json
```

(the `bin/benchmark` script simply loads the best model given in the config and eval against the test set, as defined in the data configuration) 

With 50 training epochs, best models selection with dev set, and evaluation on test set, we have: 

```
Performing a benchmark of the model
===================================
Start testing on corpus 0
Testing on corpus 0 took 103.584 seconds
Classification result on test_0 ep 40:

Application/Precision/test_0:   0.815
Application/Recall/test_0:  0.867
Application/FScore/test_0:  0.84

Total/Precision/test_0: 0.815
Total/Recall/test_0:    0.867
Total/FScore/test_0:    0.84

Total/Loss/test_0:  0
Done

Confusion Matrix for:
['B-Application', 'I-Application', 'O']
[[   514      1     56]
 [     3    841    129]
 [    78    175 336394]]
```

Note: the original data config file indicates 200 epochs, so we will need to use the same number of epochs (but it takes really a lot of time). 

# SoMeNLP

SoMeNLP provides functionality for performing information extraction for software mentions in scientific articles. 
Implemented are: Named Entity Recognition, Relation Extraction and Entity Disambiguation. 

Up to now it has been trained on the SoMeSci dataset (available from [zenodo](https://zenodo.org/record/4701764) or [github](https://github.com/dave-s477/SoMeSci)) and applied on the [PMC OA](https://www.ncbi.nlm.nih.gov/pmc/tools/openftlist/) Subset for information extraction.

## Installing

SoMeNLP is structured as a Python package containing code and command line scripts. 
It is crucial that **Python >= 3.6** is used because the insertion order of dictionaries has to be retained.

The package can be installed by: 
```shell
git clone https://github.com/dave-s477/SoMeNLP
cd SoMeNLP
pip install .
```
or for an editable installation
```shell
pip install -e .
```

There is a long list of dependencies that come with the install:
[gensim](https://pypi.org/project/gensim/), [pytorch](https://pypi.org/project/torch/), [tensorboard](https://pypi.org/project/tensorboard/), [articlenizer](https://github.com/dave-s477/articlenizer), [pandas](https://pypi.org/project/pandas/), [numpy](https://pypi.org/project/numpy/), [beautifulsoup](https://pypi.org/project/beautifulsoup4/), [wiktextract](https://pypi.org/project/wiktextract/), [wget](https://pypi.org/project/wget/), [NLTK](https://pypi.org/project/nltk/), [scikit-learn](https://pypi.org/project/scikit-learn/), [transformers](https://pypi.org/project/transformers/), [SPARQLWrapper](https://pypi.org/project/SPARQLWrapper/), and [python-levenshtein](https://pypi.org/project/python-Levenshtein/).

## Word Embeddings

Word embeddings are required to run the Bi-LSTM-CRF named entity recognition model. 
There are two options for getting an word embedding:
1. Use a publicly available one: `wikipedia-pubmed-and-PMC-w2v.bin` from http://evexdb.org/pmresources/vec-space-models/ 
2. Train a new one, for instance, on the PMC OA subset: JATS files are available from https://ftp.ncbi.nlm.nih.gov/pub/pmc/oa_bulk/
and can be extracted and preprocessed with articlenizer:
```shell
parse_JATS --in-path path/PMC_OA_JATS_folder/ --out-path path/PMC_OA_Text_folder --ncores 60
articlenizer_prepro --in-path path/PMC_OA_Text_folder --out-path path/PMC_OA_prepro --ncores 60 
train_word_emb --in-path path/PMC_OA_prepro --out-path data/word_embs/ --ncores 60
```

## Training

### Data

As data input format BRAT standoff-format is assumed.
It needs to be first transformed into data suited for the models described below.
This can be done by:
```shell
brat_to_bio --in-path data/minimal_example/text/ --out-path data/minimal_example/bio
```
The data also needs to be split into training, development and test set. 
```
split_data --in-path data/minimal_example/bio/ --out-path data/minimal_example/
```
and after SoMeSci was downloaded and BRAT folders were extracted/copied to `data`:
```shell
brat_to_bio --in-path data/PLoS_sentences --out-path data/PLoS_sentences_bio/ --ncores 4
mv data/PLoS_sentences_bio/ data/PLoS_sentences

brat_to_bio --in-path data/PLoS_methods --out-path data/PLoS_methods_bio --ncores 4
split_data --in-path data/PLoS_methods_bio --out-path data/PLoS_methods 

brat_to_bio --in-path data/Pubmed_fulltext --out-path data/Pubmed_fulltext_bio --ncores 4
split_data --in-path data/Pubmed_fulltext_bio --out-path data/Pubmed_fulltext 

brat_to_bio --in-path data/Creation_sentences --out-path data/Creation_sentences_bio --ncores 4
split_data --in-path data/Creation_sentences_bio --out-path data/Creation_sentences 
```
Note that PLoS_sentences is entirely used for training and not split

### Models

Training can be performed by running `bin/train_models`. Hyper-parameter optimization with `bin/tune_model`. Required configurations are available in `configurations`.

#### Bi-LSTM (with custom features)

Generating custom features additionally to word embeddings:
```shell
custom_feature_gen --in-path data/PLoS_methods/ --out-path data/PLoS_methods/
custom_feature_gen --in-path data/PLoS_sentences/ --out-path data/PLoS_sentences/
custom_feature_gen --in-path data/Pubmed_fulltext/ --out-path data/Pubmed_fulltext/
custom_feature_gen --in-path data/Creation_sentences/ --out-path data/Creation_sentences/
```
(to updated distant supervision info run: `bin/distant_supervision`)

Running the Bi-LSTM-CRF:
```shell
train_model --model-config configurations/PMC/NER/gold_feature_LSTM.json --data-config configurations/SoMeSci/named_entity_recognition/SoMeSci_data_software.json
```
The Bi-LSTM is set up to consider only one of the potential tasks. 

#### SciBERT

Download pretrained SciBERT (or BioBERT) model from Huggingface, for instance by:
```shell
mkdir data/pretrained && cd data/pretrained
git lfs clone https://huggingface.co/allenai/scibert_scivocab_cased
```

Running SciBERT on a **single task** by re-train the model:
```shell
train_model --model-config configurations/PMC/NER/gold_SciBERT_final.json --data-config configurations/PMC/NER/gold_data_SciBERT_final.json
```

Running SciBERT on **multiple tasks**:
```shell
train_model --model-config configurations/PMC/NER/gold_multi_opt2_SciBERT.json --data-config configurations/PMC/NER/gold_data_multi_opt2_SciBERT.json
```