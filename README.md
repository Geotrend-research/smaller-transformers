# Smaller Multilingual Transformers

This repository shares smaller versions of multilingual transformers that keep the same representations offered by the original ones. The idea came from a simple observation: *after massively multilingual pretraining, not all embeddings are needed to perform finetuning and inference*. In practice one would rarely require a model that supports more than 100 languages as the original [mBERT](https://github.com/google-research/bert/blob/master/multilingual.md). Therefore, we extracted several smaller versions that handle fewer languages. Since most of the parameters of multilingual transformers are located in the embeddings layer, our models are between 21% and 45% smaller in size.

The table bellow compares two of our exracted versions with the original mBERT. It shows the models size, memory footprint and the obtained accuracy on the [XNLI dataset](https://github.com/facebookresearch/XNLI) (Cross-lingual Transfer from english for french). These measurements have been computed on a [Google Cloud n1-standard-1 machine (1 vCPU, 3.75 GB)](https://cloud.google.com/compute/docs/machine-types\#n1_machine_type).

|            Model                | Num parameters |   Size   |  Memory  | Accuracy |
| ----------------------------    | -------------- | -------- | -------- | -------- |
| bert-base-multilingual-cased    |   178 million  |  714 MB  | 1400 MB  |   73.8   |
| Geotrend/bert-base-15lang-cased |   141 million  |  564 MB  | 1098 MB  |   74.1   |
| Geotrend/bert-base-en-fr-cased  |   124 million  |  447 MB  |  878 MB  |   73.8   |

Reducing the size of multilingual transformers facilitates their deployment on public cloud platforms. 
For instance, Google Cloud Platform requires that the model size on disk should be lower than 500 MB for serveless deployments (Cloud Functions / Cloud ML).

For more information, please refer to our paper: [Load What You Need](https://arxiv.org/abs/2010.05609).

## Available Models

Until now, we generated 30 smaller models from the original mBERT cased version. These models have been uploaded to the [Hugging Face Model Hub](https://huggingface.co/models) in order to facilitate their use: https://huggingface.co/Geotrend.

They can be downloaded easily using the [transformers library](https://github.com/huggingface/transformers):

```python
from transformers import AutoTokenizer, AutoModel

tokenizer = AutoTokenizer.from_pretrained("Geotrend/bert-base-en-fr-cased")
model = AutoModel.from_pretrained("Geotrend/bert-base-en-fr-cased")

```

More models will be released soon.

## Generating new Models

We also share a python script that allows users to generate smaller transformers by their own based on a subset of the original vocabulary (the method does not only concern multilingual transformers):

```bash

pip install requirements.txt

python3 reduce_model.py \
	--source_model bert-base-multilingual-cased \
	--vocab_file vocab_5langs.txt \
	--output_model bert-base-5lang-cased \
	--convert_to_tf False
```

Where:
- `--source_model` is the multilingual transformer to reduce
- `--vocab_file` is the intended vocabulary file path
- `--output_model` is the name of the final reduced model
- `--convert_to_tf` tells the scipt whether to generate a tenserflow version or not

## How to Cite

```bibtex
@inproceedings{smallermbert,
  title={Load What You Need: Smaller Versions of Mutlilingual BERT},
  author={Abdaoui, Amine and Pradel, Camille and Sigel, Grégoire},
  booktitle={SustaiNLP / EMNLP},
  year={2020}
}
```

## Contact 

Please contact amine@geotrend.fr for any question, feedback or request.
