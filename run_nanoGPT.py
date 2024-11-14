import os
import argparse

from train_nanoGPT import  OmniLearnTrainNanoGPT

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--out-dir', type=str, default='out')
    parser.add_argument('--eval-interval', type=int, default=2000)
    parser.add_argument('--log-interval', type=int, default=1)
    parser.add_argument('--eval-iters', type=int, default=200)
    parser.add_argument('--eval-only', type=str, default='False')
    parser.add_argument('--always-save-checkpoint', type=str, default='True')
    parser.add_argument('--init-from', type=str, default='scratch')
    parser.add_argument('--wandb-log', type=str, default='False')
    parser.add_argument('--wandb-project', type=str, default='owt')
    parser.add_argument('--wandb-run-name', type=str, default='gpt2')
    parser.add_argument('--dataset', type=str, default='openwebtext')
    parser.add_argument('--gradient-accumulation-steps', type=int, default=1)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--block-size', type=int, default=1024)
    parser.add_argument('--n-layer', type=int, default=12)
    parser.add_argument('--n-head', type=int, default=12)
    parser.add_argument('--n-embd', type=int, default=768)
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--bias', type=str, default='False')
    parser.add_argument('--learning-rate', type=float, default=6e-4)
    parser.add_argument('--max-iters', type=int, default=600000)
    parser.add_argument('--weight-decay', type=float, default=1e-1)
    parser.add_argument('--beta1', type=float, default=0.9)
    parser.add_argument('--beta2', type=float, default=0.95)
    parser.add_argument('--grad-clip', type=float, default=1.0)
    parser.add_argument('--decay-lr', type=str, default='True')
    parser.add_argument('--warmup-iters', type=int, default=2000)
    parser.add_argument('--lr-decay-iters', type=int, default=600000)
    parser.add_argument('--min-lr', type=float, default=6e-5)
    parser.add_argument('--backend', type=str, default='gloo')
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--dtype', type=str, default='float16')
    parser.add_argument('--compile', type=str, default='False')
    parser.add_argument('--rank', type=int, default=0)
    parser.add_argument('--world-size', type=int, default=1)
    parser.add_argument('--dir', type=str, default='/')
    parser.add_argument('--interface', type=str, default='eth0')
    parser.add_argument('--master-addr', type=str, default='127.0.0.1')

    args = parser.parse_args()

    if args.backend == 'gloo':
        os.environ['GLOO_SOCKET_IFNAME'] = args.interface

    OmniLearnTrainNanoGPT(args=args).launch_training()