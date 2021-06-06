import argparse
import numpy as np
import os
from glob import glob
import time
import torch
import torch.optim as optim
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.debug.metrics as met
import torch_xla.distributed.parallel_loader as pl
import torch_xla.distributed.xla_multiprocessing as xmp
import torch_xla.utils.utils as xu

from transformers import (
    BertForMaskedLM,
    BertConfig,
    BertJapaneseTokenizer,
    DataCollatorForLanguageModeling,
)

from datasets import Dataset

parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", type=str, help="data dir in the gcs bucket")
parser.add_argument("--train_samples", type=int, help="number of train samples")
parser.add_argument("--vocab_size", type=int, help="tokenizer's vocab size", default=32000)
parser.add_argument("--mlm_probability", type=str, help="Masked Language Model's mask probability", default=0.15)
parser.add_argument("--batch_size", type=int, help="batch size", default=4)
parser.add_argument("--num_workers", type=int, help="number of workers in data preprocessing", default=4)
parser.add_argument("--lr", type=str, help="learning rate", default="0.001")
parser.add_argument("--momentum", type=str, help="optimizer's momentum", default="0.5")
parser.add_argument("--num_cores", type=int, help="number of tpu cores", default=8)
parser.add_argument("--num_epochs", type=int, help="training epochs", default=1)
parser.add_argument("--log_steps", type=int, help="log steps", default=100)

args = parser.parse_args()

SERIAL_EXEC = xmp.MpSerialExecutor()

DATASET_PATH = args.data_dir

TOKENIZER = BertJapaneseTokenizer.from_pretrained('cl-tohoku/bert-base-japanese-v2')

config = BertConfig.from_pretrained('cl-tohoku/bert-base-japanese-whole-word-masking',
                                    vocab_size=args.vocab_size)

# Only instantiate model weights once in memory.
WRAPPED_MODEL = xmp.MpModelWrapper(BertForMaskedLM(config=config))

def train_bert():
    torch.manual_seed(1)
  
    def get_dataset():
        dataset = Dataset.load_from_disk(DATASET_PATH)
        if args.train_samples and args.train_samples < len(dataset):
            dataset = torch.utils.data.Subset(dataset, np.random.choice(args.train_samples, replace=False))
        return dataset

  
    # Using the serial executor avoids multiple processes to
    # download the same data.
    train_dataset = SERIAL_EXEC.run(get_dataset)

    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset,
        num_replicas=xm.xrt_world_size(),
        rank=xm.get_ordinal(),
        shuffle=True)
    
    data_collator = DataCollatorForLanguageModeling(tokenizer=TOKENIZER,
                                                    mlm_probability=float(args.mlm_probability),
                                                    pad_to_multiple_of=512)
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        collate_fn = data_collator,
        num_workers=args.num_workers,
        drop_last=True)

    # Scale learning rate to world size
    lr = float(args.lr) * xm.xrt_world_size()

    # Get loss function, optimizer, and model
    device = xm.xla_device()
    model = WRAPPED_MODEL.to(device)
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=float(args.momentum))

    def train_loop_fn(loader):
        tracker = xm.RateTracker()
        model.train()
        for x, data in enumerate(loader):
            optimizer.zero_grad()
            output = model(data)
            loss = output.loss
            loss.backward()
            xm.optimizer_step(optimizer)
            tracker.add(args.batch_size)
            if x % args.log_steps == 0:
                print('[xla:{}]({}) Loss={:.5f} Rate={:.2f} GlobalRate={:.2f} Time={}'.format(
                    xm.get_ordinal(), x, loss.item(), tracker.rate(),
                    tracker.global_rate(), time.asctime()), flush=True)

    # Train loops
    data, pred = None, None
    for epoch in range(1, args.num_epochs + 1):
        para_loader = pl.ParallelLoader(train_loader, [device])
        train_loop_fn(para_loader.per_device_loader(device))
        xm.master_print("Finished training epoch {}".format(epoch))

    return data, pred

# Start training processes
def _mp_fn(rank, args):
    global FLAGS
    FLAGS = args
    torch.set_default_tensor_type('torch.FloatTensor')
    data, pred = train_bert()

def main():
    xmp.spawn(_mp_fn, args=(args,), nprocs=args.num_cores,
            start_method='fork')

if __name__ ==  '__main__':
    main()
