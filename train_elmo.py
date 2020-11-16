# -*- coding: utf-8 -*-
"""
Created on Tue Sep  1 17:07:47 2020

@author: USER
"""
import os
import sys
import argparse
from google.cloud import storage
from flair.data import Dictionary
from flair.models import LanguageModel
from flair.trainers.language_model_trainer import LanguageModelTrainer, TextCorpus



def parse_args(args):
    parser = argparse.ArgumentParser()
    parser.add_argument('--corpus_path',help= 'path to corpus',default='corpus')
    parser.add_argument('--mapfile',help= 'mapfile name',default='mapfile')
    parser.add_argument('--save_model',help= 'path to save model')
    parser.add_argument('--is_forward_lm',help= '# are you training a forward or backward LM? ',action='store_false')
    parser.add_argument('--seq_length',help= 'sequence length',default=250,type=int)
    parser.add_argument('--mini_batch',help= 'mini batch size',default=100, type=int)
    parser.add_argument('--hidden_size',help= 'hidden_size dimenstion',default=1024, type=int)
    parser.add_argument('--nlayers',help= 'nlayers dimenstion',default=1, type=int)
    parser.add_argument('--epochs',help= 'Number of epochs',default=2000, type=int)
    parser.add_argument('--checkpoint',help= 'Save checkpoits?',action='store_false')
    parser.add_argument('--bucket_name',help= 'Bucket name',default='elmo_data')
    parser.add_argument('--bucket_prefix',help= 'Bucket prefix',default='')
    parser.add_argument('--checkpoint_path',help= 'checkpoint path ',default='')
    parser.add_argument('--finetune',help= '# are you training a forward or backward LM? ',action='store_true')
    parser.add_argument('--finetune_checkpoint',help= '# are you training a forward or backward LM? ',action='store_true')
    args = parser.parse_args()
    return args


def download_corpus(args):
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name=args.bucket_name)
    blobs = bucket.list_blobs(prefix=args.bucket_prefix)  # Get list of files
    for blob in blobs:
        
        print(blob.name)
        os.makedirs(os.path.dirname(blob.name), exist_ok=True)
        try:
            blob.download_to_filename(blob.name)  # Download
        except Exception as e:
            print(e)
            

def create_corpus(args,load_dict_from_lm=False,return_back='both'):
    if not load_dict_from_lm:
        dictionary: Dictionary = Dictionary.load(os.path.join(args.corpus_path,args.mapfile))
        
    else:
        print("loading dictionary from finetune model")
        from flair.embeddings import FlairEmbeddings
        dictionary = FlairEmbeddings('he-forward').lm.dictionary
        
    language_model = LanguageModel(dictionary,
                                   args.is_forward_lm,
                                   hidden_size=args.hidden_size,
                                   nlayers=1)
        
    corpus = TextCorpus(args.corpus_path,
                        dictionary,
                        args.is_forward_lm,
                        character_level=True)
    if return_back == 'both':   
        return language_model,corpus
    elif return_back == 'language_model':
        return language_model
    elif return_back == 'corpus':
        return corpus
    else:
        print('Specified what to return back')
            
    
def train_elmo(args):
    
    

    if args.finetune and args.checkpoint_path == '':
        print("finetune")
        from flair.embeddings import FlairEmbeddings
        language_model = FlairEmbeddings('he-forward').lm
        corpus: TextCorpus = TextCorpus(args.corpus_path,language_model.dictionary,language_model.is_forward_lm,character_level=True)
        trainer = LanguageModelTrainer(language_model, corpus)
		
    elif args.checkpoint_path == '' and not args.finetune :
        
        # Training from scrach
        print('Training from scarch')
        
        #Downloading data
        if not os.path.exists(args.corpus_path):
            print('Corpus _path',args.corpus_path)
            download_corpus(args) 
            

        language_model, corpus = create_corpus(args)
        trainer = LanguageModelTrainer(language_model, corpus)
    
        
    
    else:
        print("Training from checpoint")
        
        from pathlib import Path
        checkpoint = Path(args.checkpoint_path)
        if args.finetune:
            load_dict_from_lm = True
        else:
            load_dict_from_lm = False
            
        trainer = LanguageModelTrainer.load_from_checkpoint(checkpoint, create_corpus(args,load_dict_from_lm,return_back='corpus'))
        
        
    
    trainer.train(args.save_model,
                  sequence_length=args.seq_length,
                  mini_batch_size=args.mini_batch,
              max_epochs=args.epochs,
              checkpoint=args.checkpoint)

def main(args):
    train_elmo(args)
    
    
if __name__ == '__main__':
    args = sys.argv[1:]       
    args = parse_args(args)
    main(args)


