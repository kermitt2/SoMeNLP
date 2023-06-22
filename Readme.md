This is a fork of SoMeNLP for experimenting and benchmarking exercices. It includes:

1. reproduced training and evaluation of the single NER task model on the SoMeSci corpus (SciBERT fine-tuned),

2. reproduced training and evaluation of the multi-tasks model on the SoMeSci corpus (SciBERT fine-tuned),

3. evaluation of the SciBERT fine-tuned model trained on the SoMeSci corpus (with the provided configuration), applied to the Softcite holdout set (limited to software names). 

After following the install/setup [instructions of the original project readme](#installing) (below, note: although not used with SciBERT, generating the features is necessary to avoid raising errors), to train a **NER task model on SoMeSci** using the dev set to select best model and without test set (as defined in the data configuration):

```console
./bin/train_model --model-config configurations/PMC/NER/gold_SciBERT_final.json --data-config configurations/PMC/NER/gold_data_SciBERT_final_2.json
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

Then, to benchmark a model against the SoMeSci test set (as defined in the data configuration):

```console
./bin/benchmark --model-config /media/lopez/store/save/Gold-SciBERT/04-02-2023_21-50-57/model_conf.json --data-config configurations/PMC/NER/gold_data_SciBERT_final_2.json
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

Note: the original data config file indicates 200 epochs, although the loss that does not seem to be moving a lot at this training stage, we would need to use the same number of epochs (but it takes really a lot of time). 

To train the **multitask model on SoMeSci** similarly using the dev set to select best model and without test set (as defined in the data configuration):

```console
./bin/train_model --model-config configurations/PMC/NER/gold_multi_opt2_SciBERT.json --data-config configurations/PMC/NER/gold_data_multi_opt2_SciBERT_2.json
```

Similarly after editing the config file of the saved model to point to the saved best model weights, we can benchmark a model against the SoMeSci test set (as defined in the data configuration) as follow:

```console
./bin/benchmark --model-config /home/lopez/SoMeNLP/save/Gold-Multi-Opt2-SciBERT/22-06-2023_00-53-56/model_conf.json --data-config configurations/PMC/NER/gold_data_multi_opt2_SciBERT_2.json
```

The benchmark results against SoMeSci are then as follow (with 50 trainng epoch as for the single task model), for the 3 different tasks: 

* Entity labeling:

```
Performing a benchmark of the model
===================================
Start testing on corpus 0
Testing on corpus 0 took 112.298 seconds
Classification result on software/test_0 ep 37:

software/Abbreviation/Precision/test_0: 0.6
software/Abbreviation/Recall/test_0:    0.923
software/Abbreviation/FScore/test_0:    0.727

software/AlternativeName/Precision/test_0:  1.0
software/AlternativeName/Recall/test_0: 1.0
software/AlternativeName/FScore/test_0: 1.0

software/Application/Precision/test_0:  0.782
software/Application/Recall/test_0: 0.879
software/Application/FScore/test_0: 0.828

software/Citation/Precision/test_0: 0.745
software/Citation/Recall/test_0:    0.888
software/Citation/FScore/test_0:    0.81

software/Developer/Precision/test_0:    0.806
software/Developer/Recall/test_0:   0.908
software/Developer/FScore/test_0:   0.854

software/Extension/Precision/test_0:    0.154
software/Extension/Recall/test_0:   0.333
software/Extension/FScore/test_0:   0.211

software/License/Precision/test_0:  0.786
software/License/Recall/test_0: 0.846
software/License/FScore/test_0: 0.815

software/Release/Precision/test_0:  0.75
software/Release/Recall/test_0: 0.9
software/Release/FScore/test_0: 0.818

software/URL/Precision/test_0:  0.802
software/URL/Recall/test_0: 0.908
software/URL/FScore/test_0: 0.852

software/Version/Precision/test_0:  0.883
software/Version/Recall/test_0: 0.952
software/Version/FScore/test_0: 0.916

software/Total/Precision/test_0:    0.793
software/Total/Recall/test_0:   0.894
software/Total/FScore/test_0:   0.84
```


* software mention typing: 

```
Classification result on soft_type/test_0 ep 37:

soft_type/Application/Precision/test_0: 0.668
soft_type/Application/Recall/test_0:    0.827
soft_type/Application/FScore/test_0:    0.739

soft_type/OperatingSystem/Precision/test_0: 0.708
soft_type/OperatingSystem/Recall/test_0:    0.739
soft_type/OperatingSystem/FScore/test_0:    0.723

soft_type/PlugIn/Precision/test_0:  0.538
soft_type/PlugIn/Recall/test_0: 0.438
soft_type/PlugIn/FScore/test_0: 0.483

soft_type/ProgrammingEnvironment/Precision/test_0:  0.972
soft_type/ProgrammingEnvironment/Recall/test_0: 0.958
soft_type/ProgrammingEnvironment/FScore/test_0: 0.965

soft_type/SoftwareCoreference/Precision/test_0: 0.615
soft_type/SoftwareCoreference/Recall/test_0:    0.889
soft_type/SoftwareCoreference/FScore/test_0:    0.727

soft_type/Total/Precision/test_0:   0.689
soft_type/Total/Recall/test_0:  0.786
soft_type/Total/FScore/test_0:  0.731
```

* Function information on the software mentions:

```
Classification result on mention_type/test_0 ep 37:

mention_type/Creation/Precision/test_0: 0.778
mention_type/Creation/Recall/test_0:    0.836
mention_type/Creation/FScore/test_0:    0.806

mention_type/Deposition/Precision/test_0:   0.633
mention_type/Deposition/Recall/test_0:  0.838
mention_type/Deposition/FScore/test_0:  0.721

mention_type/Mention/Precision/test_0:  0.268
mention_type/Mention/Recall/test_0: 0.542
mention_type/Mention/FScore/test_0: 0.359

mention_type/Usage/Precision/test_0:    0.798
mention_type/Usage/Recall/test_0:   0.831
mention_type/Usage/FScore/test_0:   0.814

mention_type/Total/Precision/test_0:    0.74
mention_type/Total/Recall/test_0:   0.807
mention_type/Total/FScore/test_0:   0.769
```

To benchmark a model against the **Softcite holdout set** (defined in the indicated data configuration `gold_data_SciBERT_final_holdout.json`) - the holdout set is 20% of the Softcite dataset with full article content to evaluate the extraction on real mention distribution:

```console
bin/benchmark --model-config /media/lopez/store/save/Gold-SciBERT/04-02-2023_21-50-57/model_conf.json --data-config configurations/PMC/NER/gold_data_SciBERT_final_holdout.json
```

```
Performing a benchmark of the model
===================================
Start testing on corpus 0
Testing on corpus 0 took 1756.502 seconds
Classification result on test_1 ep 40:

Application/Precision/test_1:   0.529
Application/Recall/test_1:  0.76
Application/FScore/test_1:  0.624

Total/Precision/test_1: 0.529
Total/Recall/test_1:    0.76
Total/FScore/test_1:    0.624

Total/Loss/test_1:  0
Done

Confusion Matrix for:
['B-Application', 'I-Application', 'O']
[[    534       5     110]
 [     18     886     341]
 [    358     580 5490639]]
```

**Note:** we match against only software name mentions in the Softcite corpus. 

**Some software annotations are different between Softcite and SoMeSci due to annotation scope and guidelines. These differences in term of definition of what are "software mentions" can lead to false matches when applied to the Softcite holdout set that are not related to the model and the training approach used in SoMeNLP. Determining the proportion of false matches due to different definition and scope of software mentions and due to the distribution of software mentions and quality of the training data of SoMeNLP will require addition work.**

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