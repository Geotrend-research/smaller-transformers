import os
import json
import argparse
from torch import nn
from transformers import AutoTokenizer
from transformers import AutoModel
from transformers import TFAutoModel


def select_embeddings(model, old_vocab, new_vocab, model_name='new_model'):
    # Get old embeddings from model
    old_embeddings = model.get_input_embeddings()
    old_num_tokens, old_embedding_dim = old_embeddings.weight.size()
    
    if old_num_tokens != len(old_vocab):
        print('len(old_vocab) != len(model.old_embeddings)')
        return old_embeddings
    
    new_num_tokens = len(new_vocab)
    if new_vocab is None:
        print('nothing to copy')
        return old_embeddings
    
    # Build new embeddings
    print('reducing model size ...')
    new_embeddings = nn.Embedding(new_num_tokens, old_embedding_dim)
    new_embeddings.to(old_embeddings.weight.device)
    
    # Copy weights
    i = 0
    j = 0
    vocab = []
    for token in old_vocab:
        if token in new_vocab:
            vocab.append(token)
            new_embeddings.weight.data[i, :] = old_embeddings.weight.data[j, :]
            i += 1
        j += 1
    
    model.set_input_embeddings(new_embeddings)
    
    # Update base model and current model config
    model.config.vocab_size = new_num_tokens
    model.vocab_size = new_num_tokens

    # Tie weights
    model.tie_weights()
    
    # Save new model
    model.save_pretrained(model_name)
    print(model_name, " - num_parameters : ", model.num_parameters())
    print(model_name, " - num_tokens : ", len(vocab))
    
    # Save vocab
    fw = open(os.path.join(model_name, 'vocab.txt'), 'w')
    for token in vocab:
        fw.write(token+'\n')
    fw.close()
    
    # Save tokenizer config
    fw = open(os.path.join(model_name, 'tokenizer_config.json'), 'w')
    json.dump({"do_lower_case": False, "model_max_length": 512}, fw)
    fw.close()
    
    return new_embeddings


def main():
    parser = argparse.ArgumentParser(description="reducing transformers size")
    parser.add_argument("--source_model",    
                        type=str,
                        required=False, 
                        default='bert-base-multilingual-cased',
                        help="The multilingual transformer to start from")
    parser.add_argument("--vocab_file", 
                        type=str,
                        required=False,
                        default='vocab_5langs.txt',
                        help="The intended vocabulary file path")
    parser.add_argument("--output_model",
                        type=str,
                        required=False,
                        default='output_model',
                        help="The name of the final reduced model")
    parser.add_argument("--convert_to_tf",
                        type=str,
                        required=False,
                        default=False, 
                        help="Wether to generate a tenserflow version or not")

    args = parser.parse_args()
    
    # Load original tokenizer, model and vocab
    print('starting from model:', args.source_model)
    tokenizer = AutoTokenizer.from_pretrained(args.source_model)
    model = AutoModel.from_pretrained(args.source_model)
    vocab = list(tokenizer.vocab.keys())

    print(args.source_model, " - num_parameters : ", model.num_parameters())
    print(args.source_model, " - num_tokens : ", len(vocab))

    # Load new vocab
    new_vocab = open(args.vocab_file).read().splitlines()

    # Rebuild pytorch model
    new_embs = select_embeddings(model, vocab, new_vocab, args.output_model)

    # convert to tensorflow
    if (args.convert_to_tf):
        tf_model = TFAutoModel.from_pretrained(args.output_model, from_pt=True)
        tf_model.save_pretrained(args.output_model)


if __name__ == "__main__":
    main()