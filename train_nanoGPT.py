import datetime
import logging
import socket
import os
import math
from time import perf_counter_ns

import pickle
import numpy as np
from contextlib import nullcontext

import torch
import torch.distributed as dist
from torch._C._distributed_c10d import ReduceOp

from model import GPTConfig, GPT

million = 1e6


class OmniLearnTrainNanoGPT(object):
    def __init__(self, args):
        self.out_dir = args.out_dir + 'g' + str(args.rank) + '/'
        self.eval_interval = args.eval_interval
        self.log_interval = args.log_interval
        self.eval_iters = args.eval_iters
        self.eval_only = False if args.eval_only == 'False' else True
        self.always_save_checkpoint = True if args.always_save_checkpoint == 'True' else True
        self.init_from = args.init_from  # 'scratch' or 'resume' or 'gpt2*'
        # wandb logging
        self.wandb_log = False  if args.wandb_log == 'False' else True
        self.wandb_project = args.wandb_project
        self.wandb_run_name = args.wandb_run_name
        # data
        self.dataset = args.dataset
        self.gradient_accumulation_steps = args.gradient_accumulation_steps
        self.batch_size = args.batch_size
        self.block_size = args.block_size
        # model
        self.n_layer = args.n_layer
        self.n_head = args.n_head
        self.n_embd = args.n_embd
        self.dropout = args.dropout
        self.bias = args.bias
        # adamw optimizer
        self.learning_rate = args.learning_rate
        self.max_iters = args.max_iters
        self.weight_decay = args.weight_decay
        self.beta1 = args.beta1
        self.beta2 = args.beta2
        self.grad_clip = args.grad_clip
        # learning rate decay settings
        self.decay_lr = True  if args.decay_lr == 'True' else False
        self.warmup_iters = args.warmup_iters
        self.lr_decay_iters = args.lr_decay_iters
        self.min_lr = args.min_lr
        # DDP settings
        self.backend = args.backend
        # system
        self.device = torch.device(args.device)
        self.dtype = args.dtype
        self.compile = True  if args.compile == 'True' else False

        # self.config_keys = [k for k, v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
        # exec(open('configurator.py').read())  # overrides from command line or config file
        # self.config = {k: globals()[k] for k in self.config_keys}  # will be useful for logging

        timeout = datetime.timedelta(seconds=3000 * 60 * 60 * 100)
        tcp_addr = 'tcp://' + str(args.master_addr) + ':' + str(args.master_port)
        args.tcp_addr = tcp_addr
        self.rank = args.rank
        self.world_size = args.world_size
        self.model_name = 'gpt2'
        self.logdir = args.dir
        dist.init_process_group(backend=self.backend, init_method=tcp_addr, rank=self.rank, world_size=self.world_size, timeout=timeout)
        logging.basicConfig(filename=self.logdir + '/g' + str(self.rank) + '/' + self.model_name + '-' + str(self.rank)
                                     + '.log', level=logging.INFO)

        assert self.gradient_accumulation_steps % self.world_size == 0
        self.gradient_accumulation_steps //= self.world_size
        self.tokens_per_iter = self.gradient_accumulation_steps * self.world_size * self.batch_size * self.block_size
        torch.manual_seed(1337 + self.rank)
        self.ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[self.dtype]
        self.device_type = 'cpu' if args.device == 'cpu' else 'cuda'
        self.ctx = nullcontext() if self.device_type == 'cpu' else torch.amp.autocast(device_type=self.device_type, dtype=self.ptdtype)
        self.model = None
        if self.rank == 0:
            os.makedirs(self.out_dir, exist_ok=True)

        self.data_dir = os.path.join('data', self.dataset)
        args.hostname = socket.gethostname()
        args.tokens_per_iter = self.tokens_per_iter
        logging.info(f'model arguments are {args}')

    def get_batch(self, split):
        if split == 'train':
            data = np.memmap(os.path.join(self.data_dir, 'train.bin'), dtype=np.uint16, mode='r')
        else:
            data = np.memmap(os.path.join(self.data_dir, 'val.bin'), dtype=np.uint16, mode='r')
        ix = torch.randint(len(data) - self.block_size, (self.batch_size,))
        x = torch.stack([torch.from_numpy((data[i:i + self.block_size]).astype(np.int64)) for i in ix])
        y = torch.stack([torch.from_numpy((data[i + 1:i + 1 + self.block_size]).astype(np.int64)) for i in ix])
        if self.device_type == 'cuda':
            # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
            x, y = x.pin_memory().to(self.device, non_blocking=True), y.pin_memory().to(self.device, non_blocking=True)
        else:
            x, y = x.to(self.device), y.to(self.device)
        return x, y

    @torch.no_grad()
    def estimate_loss(self):
        out = {}
        self.model.eval()
        for split in ['train', 'val']:
            losses = torch.zeros(self.eval_iters)
            for k in range(self.eval_iters):
                X, Y = self.get_batch(split)
                with self.ctx:
                    logits, loss = self.model(X, Y)
                losses[k] = loss.item()
            out[split] = losses.mean()
        self.model.train()
        return out

    # learning rate decay scheduler (cosine with warmup)
    def get_lr(self, it):
        # 1) linear warmup for warmup_iters steps
        if it < self.warmup_iters:
            return self.learning_rate * it / self.warmup_iters
        # 2) if it > lr_decay_iters, return min learning rate
        if it > self.lr_decay_iters:
            return self.min_lr
        # 3) in between, use cosine decay down to min learning rate
        decay_ratio = (it - self.warmup_iters) / (self.lr_decay_iters - self.warmup_iters)
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # coeff ranges 0..1
        return self.min_lr + coeff * (self.learning_rate - self.min_lr)

    def launch_training(self):
        iter_num = 0
        best_val_loss = 1e9
        meta_path = os.path.join(self.data_dir, 'meta.pkl')
        meta_vocab_size = None
        if os.path.exists(meta_path):
            with open(meta_path, 'rb') as f:
                meta = pickle.load(f)
            meta_vocab_size = meta['vocab_size']
            logging.info(f"found vocab_size = {meta_vocab_size} (inside {meta_path})")

        model_args = dict(n_layer=self.n_layer, n_head=self.n_head, n_embd=self.n_embd, block_size=self.block_size,
                          bias=self.bias, vocab_size=None, dropout=self.dropout)

        if self.init_from == 'scratch':
            # init a new model from scratch
            print("Initializing a new model from scratch")
            # determine the vocab size we'll use for from-scratch training
            if meta_vocab_size is None:
                logging.info("defaulting to vocab_size of GPT-2 to 50304 (50257 rounded up for efficiency)")
            model_args['vocab_size'] = meta_vocab_size if meta_vocab_size is not None else 50304
            gptconf = GPTConfig(**model_args)
            self.model = GPT(gptconf)

        elif self.init_from == 'resume':
            print(f"Resuming training from {self.out_dir}")
            # resume training from a checkpoint.
            ckpt_path = os.path.join(self.out_dir, 'ckpt.pt')
            checkpoint = torch.load(ckpt_path, map_location=self.device)
            checkpoint_model_args = checkpoint['model_args']
            # force these config attributes to be equal otherwise we can't even resume training
            # the rest of the attributes (e.g. dropout) can stay as desired from command line
            for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
                model_args[k] = checkpoint_model_args[k]
            # create the model
            gptconf = GPTConfig(**model_args)
            self.model = GPT(gptconf)
            state_dict = checkpoint['model']
            # fix the keys of the state dictionary :(
            # honestly no idea how checkpoints sometimes get this prefix, have to debug more
            unwanted_prefix = '_orig_mod.'
            for k, v in list(state_dict.items()):
                if k.startswith(unwanted_prefix):
                    state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
            self.model.load_state_dict(state_dict)
            iter_num = checkpoint['iter_num']
            best_val_loss = checkpoint['best_val_loss']

        elif self.init_from.startswith('gpt2'):
            print(f"Initializing from OpenAI GPT-2 weights: {self.init_from}")
            # initialize from OpenAI GPT-2 weights
            override_args = dict(dropout=self.dropout)
            self.model = GPT.from_pretrained(self.init_from, override_args)
            # read off the created config params, so we can store them into checkpoint correctly
            # for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
            #     model_args[k] = getattr(self.model.config, k)

        # crop down the model block size if desired, using model surgery
        if self.block_size < self.model.config.block_size:
            self.model.crop_block_size(self.block_size)
            model_args['block_size'] = self.block_size  # so that the checkpoint will have the right value

        self.model.to(self.device)
        scaler = torch.cuda.amp.GradScaler(enabled=(self.dtype == 'float16'))
        optimizer = self.model.configure_optimizers(self.weight_decay, self.learning_rate, (self.beta1, self.beta2), self.device_type)
        if self.init_from == 'resume':
            optimizer.load_state_dict(checkpoint['optimizer'])

        checkpoint = None  # free up memory

        # compile the model
        if compile:
            print("compiling the model... (takes a ~minute)")
            unoptimized_model = self.model
            self.model = torch.compile(self.model)  # requires PyTorch 2.0

        # training loop
        X, Y = self.get_batch('train')  # fetch the very first batch
        local_iter_num = 0  # number of iterations in the lifetime of this process
        running_mfu = -1.0
        while True:
            lr = self.get_lr(iter_num) if self.decay_lr else self.learning_rate
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

            if iter_num % self.eval_interval == 0 and self.rank == 0:
                losses = self.estimate_loss()
                print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
                if self.wandb_log:
                    logging.info({"iter": iter_num, "train/loss": losses['train'], "val/loss": losses['val'], "lr": lr,
                        "mfu": running_mfu * 100,})
                if losses['val'] < best_val_loss or self.always_save_checkpoint:
                    best_val_loss = losses['val']
                    if iter_num > 0:
                        checkpoint = {
                            # 'model': raw_model.state_dict(),
                            'model': self.model.state_dict(),
                            'optimizer': optimizer.state_dict(),
                            'model_args': model_args,
                            'iter_num': iter_num,
                            'best_val_loss': best_val_loss,
                            #'config': self.config,
                        }
                        print(f"saving checkpoint to {self.out_dir}")
                        torch.save(checkpoint, os.path.join(self.out_dir, 'ckpt.pt'))
            if iter_num == 0 and self.eval_only:
                break

            compute_time = None
            for micro_step in range(self.gradient_accumulation_steps):
                self.model.require_backward_grad_sync = (micro_step == self.gradient_accumulation_steps - 1)
                with self.ctx:
                    begin = perf_counter_ns()
                    logits, loss = self.model(X, Y)
                    loss = loss / self.gradient_accumulation_steps

                X, Y = self.get_batch('train')
                scaler.scale(loss).backward()
                compute_time += (perf_counter_ns() - begin) / million

            if self.grad_clip != 0.0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)

            # log communication time
            begin = perf_counter_ns()
            for param in self.model.parameters():
                dist.all_reduce(tensor=param.grad, op=ReduceOp.SUM, async_op=False)
            sync_time = (perf_counter_ns() - begin) / million

            scaler.step(optimizer)
            scaler.update()
            # flush the gradients as soon as we can, no need for this memory anymore
            optimizer.zero_grad(set_to_none=True)

            iter_num += 1
            local_iter_num += 1
            if iter_num > self.max_iters:
                break

            logging.info(f'nanoGPT tstep {iter_num} local_step {local_iter_num} compute_time {compute_time} ms '
                         f'sync_time {sync_time} ms')

            # implement OmniLearn PID-control here

        logging.info(f'completed a training epoch at tstep {iter_num} on rank {self.rank}')