import argparse
import torch
from typing import Tuple, List
from abc import ABC, abstractmethod
import numpy as np
import enum
import openai
import wandb
import os
import hydra
from omegaconf import DictConfig, OmegaConf
from pathlib import Path
from evaluate import load
from torch import autograd
from copy import deepcopy
from tqdm.auto import tqdm
from torch.optim import AdamW, SGD, Adam
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
from transformers import pipeline, get_scheduler, get_cosine_schedule_with_warmup
from transformers import StoppingCriteria, StoppingCriteriaList
from datasets import load_dataset, load_from_disk, load_metric, Dataset, DatasetDict


log_dir = None
config_dict = None
MASK_TOKEN = '<[.MASK.]>'


@hydra.main(config_path="configs", config_name="config")
def container_init(config):
    global log_dir, config_dict
    # Variables that we will use throughout the rest of the streamlit execution
    config_dict = OmegaConf.to_container(config, resolve=True, throw_on_missing=True)
    # hydra changes working directories
    log_dir = str(Path.joinpath(Path(os.getcwd()), config.run_params.log_dir))
    # initialize wandb
    wandb.login()
    # tracks everything that TensorBoard tracks
    # writes to same dir as TensorBoard    


def parse_args() -> Tuple[dict, torch.device]:
    parser = argparse.ArgumentParser(description='Process model parameters.')
    parser.add_argument('--max_length', type=int, default=256,
                        help='maximum sequence length (default: 256)')
    parser.add_argument('--device', type=str, default='cpu',
                        help='select GPU or CPU device to run the model on (default: cpu)')
    parser.add_argument('--dtype', type=str, default='float32',
                        help='select the floating point data type (default: float32)')
    parser.add_argument('--no_sample', action=argparse.BooleanOptionalAction,
                        help='do not sample for deterministic text generation')
    parser.add_argument('--debug', action=argparse.BooleanOptionalAction,
                        help='enable debug messages')
    parser.add_argument('--seed', type=int, default=42,
                        help='set seed value (default: 42)')
    parser.add_argument('--temperature', type=float, default=0.6,
                        help='set temperature parameter for generation function (default: 0.6)')
    parser.add_argument('--model', type=str, default='gpt2',
                        help='initialize model from repository (default: gpt2)')
    parser.add_argument('--revision', type=str, default=None,
                        help='select model repository revision (default: None)')
    parser.add_argument('--log_dir', type=str, default='runs',
                        help='paths for logging (default: runs)')
    parser.add_argument('--learning_rate', type=float, default=0.0005,
                        help='set learning rate for optimizer (default: 0.0005)')
    parser.add_argument('--grad_clipping', type=float, default=2.0,
                        help='set grad_clipping (default: 2.0)')
    parser.add_argument('--num_epochs', type=int, default=4,
                        help='set number of epochs (default: 4)')
    parser.add_argument('--train_subset_size', type=int, default=100,
                        help='set train subset size (default: 100)')
    parser.add_argument('--eval_subset_size', type=int, default=100,
                        help='set eval subset size (default: 100)')
    parser.add_argument('--checkpoint_interval', type=int, default=1,
                        help='set checkpoint interval (default: 1)')
    parser.add_argument('--model_path', type=str, default='models/demo',
                        help='set model path (default: models/demo)')
    args = parser.parse_args()

    if args.device != 'cpu':
        device = torch.device(
            args.device if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device('cpu')

    if args.dtype == 'float16':
        args.dtype = torch.float16
    elif args.dtype == 'float32':
        args.dtype = torch.float32
    else:
        raise ValueError(f'Unsupported dtype [{args.dtype}]!')

    print(f"Loading {args.model} model...")
    if args.revision:
        print(f"Revision {args.revision}")
        
    container_init()
    args.log_dir = log_dir
    args.config_dict = config_dict
    return args, device


def masking_data_collator(tokenizer, batch, device, wwm_probability=0.1):
    batch = {
        'input_ids': torch.stack(batch['input_ids'], dim=0).T.to(device),
        'attention_mask': torch.stack(batch['attention_mask'], dim=0).T.to(device).half(),
        'labels': torch.stack(batch['labels'], dim=0).T.to(device)
    }
    
    mask = torch.from_numpy(np.random.binomial(1, wwm_probability, batch['input_ids'].shape)).to(device)
    batch['input_ids'] = torch.where(mask == 1, tokenizer.mask_token_id, batch['input_ids'])
    
    return batch


class Trainer(ABC):
    def __init__(self, args: dict, device: torch.device) -> None:
        super().__init__()
        self.args: dict = args
        self.device: torch.device = device
        self.model = None
        self.tokenizer = None
        
    def init_model(self):
        if self.args.seed is not None:
            set_seed(self.args.seed)
        self.tokenizer = AutoTokenizer.from_pretrained(self.args.model, use_fast=False, local_files_only=False, output_hidden_states=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.args.model, revision=self.args.revision, torch_dtype=self.args.dtype, low_cpu_mem_usage=True, local_files_only=False, pad_token_id=self.tokenizer.eos_token_id)
        
        # add new mask token
        self.tokenizer.add_tokens(MASK_TOKEN)
        self.model.resize_token_embeddings(len(self.tokenizer))
        self.model.to(self.device)
    
    @torch.no_grad()
    def eval_run(self, eval_dataloader, epoch):
        self.model.eval()
        input_texts = []
        tuned_model_perplexity = []
        losses = []
        for batch in eval_dataloader:
            batch = masking_data_collator(self.tokenizer, batch, self.model.device)
            with torch.no_grad():
                outputs = self.model(**batch)

            predicted_label_classes = outputs.logits.argmax(dim=-1)
            pp = torch.exp(outputs.loss)
            input_texts += self.tokenizer.batch_decode(batch['labels'], skip_special_tokens=True)
            tuned_model_perplexity.append(pp.cpu().item())
            losses.append(outputs.loss.cpu().item())
            
        table = wandb.Table(data=[[self.tokenizer.batch_decode(batch['input_ids'], skip_special_tokens=False), 
                                self.tokenizer.batch_decode(batch['labels'], skip_special_tokens=False), 
                                self.tokenizer.batch_decode(predicted_label_classes, skip_special_tokens=False)]], columns = ["inputs", "labels", "predictions"])

        wandb.log({
            "eval/loss": np.mean(losses), "eval/epoch": epoch,
            "eval/perplexity": np.mean(tuned_model_perplexity),
            "eval/text": table
        })
        
        return input_texts, np.mean(tuned_model_perplexity)
        
        
    def train_run(self, train_dataloader, optimizer, scheduler, epoch, progress_bar):
        self.model.train()
        losses = []
        pps = []
        for batch in train_dataloader:
            optimizer.zero_grad()
            
            batch = masking_data_collator(self.tokenizer, batch, self.model.device)
            outputs = self.model(**batch)
            loss = outputs.loss
            pp = torch.exp(loss)
            
            losses.append(loss.cpu().item())
            pps.append(pp.cpu().item())

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.grad_clipping)
            optimizer.step()        
            scheduler.step()

            predicted_label_classes = outputs.logits.argmax(dim=-1)
            progress_bar.update(1)
        
        if epoch % self.args.checkpoint_interval == 0:
            self.model.save_pretrained(self.args.model_path)
            self.tokenizer.save_pretrained(self.args.model_path)
        
        # decode last prediction and batch
        table = wandb.Table(data=[[self.tokenizer.batch_decode(batch['input_ids'], skip_special_tokens=False), 
                                self.tokenizer.batch_decode(batch['labels'], skip_special_tokens=False), 
                                self.tokenizer.batch_decode(predicted_label_classes, skip_special_tokens=False)]], columns = ["inputs", "labels", "predictions"])
        wandb.log({
            "train/loss": np.mean(losses), "train/epoch": epoch,
            "train/perplexity": np.mean(pps),
            "train/text": table,
            'lr': scheduler.get_last_lr()
        })

        
    def fine_tune_model_runner(self, dataset_name='yelp_review_full'):
        reinit = None
        if 'run' in self.args and self.args.run is not None:
            reinit=True
        self.args.run = wandb.init(reinit=reinit, name=self.args.log_dir,
                                dir=self.args.log_dir, save_code=False, config=self.args.config_dict,
                                settings=wandb.Settings(start_method="fork"))
        
        if self.tokenizer.mask_token == '' or self.tokenizer.mask_token is None:
            self.tokenizer.mask_token = MASK_TOKEN

        path = f'/publicwork/dinu/workspace/poc-log-enrichtment/data/{dataset_name}/processed'
        lm_datasets = load_from_disk(path)
        train_dataset = lm_datasets["train"].shuffle(seed=self.args.seed).select(range(self.args.train_subset_size))
        eval_dataset = lm_datasets["test"].shuffle(seed=self.args.seed).select(range(self.args.eval_subset_size))
        
        train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=8)
        eval_dataloader = DataLoader(eval_dataset, batch_size=8)
        
        num_training_steps = self.args.num_epochs * len(train_dataloader)
        kwargs = {
            'betas': (0.9, 0.999),
            'eps': 1e-08
        }
        params = []
        for name, mod in self.model.named_modules(): 
            if name=='lm_head': params += list(mod.parameters())
        if len(params) <= 0: params = self.model.parameters()
        optimizer = Adam(params, lr=self.args.learning_rate, **kwargs)
        scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=5, num_training_steps=num_training_steps)
        perplexity = load("perplexity", module_type="metric")
        
        # main train / eval loop
        progress_bar = tqdm(range(num_training_steps))
        self.eval_run(eval_dataloader, 0)
        with autograd.detect_anomaly():
            for epoch in range(self.args.num_epochs):
                self.train_run(train_dataloader, optimizer, scheduler, epoch+1, progress_bar)
                input_texts, model_perplexity = self.eval_run(eval_dataloader, epoch+1)
                
        self.model.save_pretrained(self.args.model_path)
        self.tokenizer.save_pretrained(self.args.model_path)

        # compute baseline perplexity
        results = perplexity.compute(model_id=self.args.model,
                                    add_start_token=False,
                                    predictions=input_texts)
            
        return {
            'ori_model_perplexity': np.mean(results['mean_perplexity']), 
            'tuned_model_perplexity': model_perplexity
        }


def run():
    args, device = parse_args()
    trainer = Trainer(args, device)
    trainer.init_model()
    trainer.fine_tune_model_runner()


if __name__ == "__main__":
    run()
