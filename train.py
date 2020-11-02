import os
os.environ["CUDA_VISIBLE_DEVICES"] = '1'
import math
import logging
from pprint import pformat
from argparse import ArgumentParser

import torch
from torch.optim.lr_scheduler import LambdaLR
from torch.nn.parallel import DistributedDataParallel
from ignite.engine import Engine, Events
from ignite.handlers import ModelCheckpoint
from ignite.metrics import Accuracy, Loss, MetricsLambda, RunningAverage
from ignite.contrib.handlers import ProgressBar, PiecewiseLinear, LRScheduler
from ignite.contrib.handlers.tensorboard_logger import TensorboardLogger, OutputHandler, OptimizerParamsHandler
from gpt_model import AdamW, GPT2LMHeadModel, GPT2Config, BertTokenizer, WEIGHTS_NAME, CONFIG_NAME

from utils.utils import make_logdir
from featurizer.get_dataloader import build_dataloader

PROJECT_FOLDER = os.path.dirname(os.path.realpath(__file__))
MODEL_CHECKPOINT = os.path.join(PROJECT_FOLDER, "runs/pretrained")
IMG_FEATURE_FOLDER = os.path.join(PROJECT_FOLDER, "data/output_data/train_images_feature.pkl")
IMG_FOLDER = os.path.join(PROJECT_FOLDER, "data/raw_data/images_train_dev")
DATA_FOLDER = os.path.join(PROJECT_FOLDER, "data/output_data/gpt2_train_dev_persona_BERTvocab.json")
DATA_CACHE = os.path.join(PROJECT_FOLDER, "data/output_data/gpt2_train_dev_persona_BERTvocab")
# DATA_FOLDER = os.path.join(PROJECT_FOLDER, "data/output_data/test_del.json")
# DATA_CACHE = os.path.join(PROJECT_FOLDER, "data/output_data/test_del")
# DATA_FOLDER = os.path.join(PROJECT_FOLDER, "data/output_data/gpt2_images.json")
# DATA_CACHE = os.path.join(PROJECT_FOLDER, "data/output_data/gpt2_images")
VOCAB_PATH = os.path.join(PROJECT_FOLDER, "config/Custom/vocab_custom.txt")
CONFIG_PATH = os.path.join(PROJECT_FOLDER, "config/Custom/config_base.json")

ATTR_TO_SPECIAL_TOKEN = {'additional_special_tokens': ['<num>', '<img>', '<url>', '#E-s', '|||', '[UNK]']}

logger = logging.getLogger(__file__)


def average_distributed_scalar(scalar, args):
    """ Average a scalar over the nodes if we are in distributed training. We use this for distributed evaluation. """
    if args.local_rank == -1:
        return scalar
    scalar_t = torch.tensor(scalar, dtype=torch.float, device=args.device) / torch.distributed.get_world_size()
    torch.distributed.all_reduce(scalar_t, op=torch.distributed.ReduceOp.SUM)
    return scalar_t.item()


def add_special_tokens_(model, tokenizer):
    """ Add special tokens to the tokenizer and the model if they have not already been added. """
    num_added_tokens = tokenizer.add_special_tokens(ATTR_TO_SPECIAL_TOKEN)  # doesn't add if they are already there
    if num_added_tokens > 0:
        model.resize_token_embeddings(new_num_tokens=len(tokenizer))


def train():
    parser = ArgumentParser()
    parser.add_argument("--dataset_path", type=str, default=DATA_FOLDER, help="Path of the dataset.")
    parser.add_argument("--image_path", type=str, default=IMG_FOLDER, help="Path of the images.")
    parser.add_argument("--images_feature_path", type=str, default=IMG_FEATURE_FOLDER, help="Path of the images.")
    parser.add_argument("--dataset_cache", type=str, default=DATA_CACHE, help="Path of the dataset cache_no_pretrained")
    parser.add_argument("--model_checkpoint", type=str, default="gpt2", help="Path, url or short name of the model")
    parser.add_argument('--dhead_gpt2', action='store_true', default=False, help="use double head gpt2")
    parser.add_argument("--from_step", type=int, default=-1, help="Init learning rate from this step")
    parser.add_argument('--pretrained', action='store_true', default=True, help="If False train from scratch")
    parser.add_argument("--num_candidates", type=int, default=1, help="Number of candidates for training")
    parser.add_argument("--max_history", type=int, default=3, help="Number of previous turns to keep in history")
    parser.add_argument("--max_length", type=int, default=256, help="Max length of input sentence")
    parser.add_argument("--train_batch_size", type=int, default=58, help="Batch size for training")
    parser.add_argument("--valid_batch_size", type=int, default=32, help="Batch size for validation")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=9, help="Accumulate gradients on several steps")
    parser.add_argument("--lr", type=float, default=6.25e-5, help="Learning rate")
    parser.add_argument("--scheduler", type=str, default="linear", choices=['noam', 'linear'], help="method of optim")
    parser.add_argument("--n_emd", type=int, default=768, help="Number of n_emd in config file (for noam)")
    parser.add_argument("--warmup_steps", type=int, default=5000, help="Warm up steps")
    parser.add_argument("--lm_coef", type=float, default=2.0, help="LM loss coefficient")
    parser.add_argument("--mc_coef", type=float, default=1.0, help="Multiple-choice loss coefficient")
    parser.add_argument("--max_norm", type=float, default=1.0, help="Clipping gradient norm")
    parser.add_argument("--n_epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--num_workers", type=int, default=0, help="Number of subprocesses for data loading")
    parser.add_argument("--personality_permutations", type=int, default=1, help="Number of permutations of personality sentences")
    parser.add_argument("--eval_before_start", action='store_true', help="If true start with a first evaluation before training")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device (cuda or cpu)")
    parser.add_argument("--fp16", type=str, default="O1", help="Set to O0, O1, O2 or O3 for fp16 training (see apex documentation)")
    parser.add_argument("--local_rank", type=int, default=-1, help="Local rank for distributed training (-1: not distributed)")
    args = parser.parse_args()

    # logging is set to INFO (resp. WARN) for main (resp. auxiliary) process. logger.info => log main process only, logger.warning => log all processes
    logging.basicConfig(level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    logger.warning("Running process %d", args.local_rank)  # This is a logger.warning: it will be printed by all distributed processes
    logger.info("Arguments: %s", pformat(args))

    # Initialize distributed training if needed
    args.distributed = (args.local_rank != -1)
    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        args.device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl', init_method='env://')
    logger.info("Prepare tokenizer, pretrained model and optimizer.")
    tokenizer_class = BertTokenizer
    config_class = GPT2Config  # GPT2Config if "gpt2" in args.model_checkpoint else OpenAIGPTConfig
    model_class = GPT2LMHeadModel  # GPT2DoubleHeadsModel if "gpt2" in args.model_checkpoint else OpenAIGPTDoubleHeadsModel
    if args.pretrained:
        tokenizer = tokenizer_class.from_pretrained(MODEL_CHECKPOINT, do_lower_case=False)
        # tokenizer = tokenizer_class(vocab_file=VOCAB_PATH, do_lower_case=True)
        model = model_class.from_pretrained(MODEL_CHECKPOINT)
    else:
        tokenizer = tokenizer_class(vocab_file=VOCAB_PATH, do_lower_case=False)
        tokenizer.add_special_tokens(ATTR_TO_SPECIAL_TOKEN)
        config = config_class.from_json_file(CONFIG_PATH)
        model = model_class(config)
    model.to(args.device)
    # Add special tokens if they are not already added
    # add_special_tokens_(model, tokenizer)
    # optimizer = AdamW(model.parameters(), lr=args.lr, correct_bias=True)
    optimizer = AdamW([{'params': model.parameters(), 'initial_lr': args.lr}], lr=args.lr, correct_bias=True)

    # Prepare model for FP16 and distributed training if needed (order is important, distributed should be the last)
    if args.fp16:
        from apex import amp  # Apex is only required if we use fp16 training
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16)
    if args.distributed:
        model = DistributedDataParallel(model, device_ids=[args.local_rank], output_device=args.local_rank)

    logger.info("Prepare datasets")
    train_loader, val_loader, train_sampler, valid_sampler = build_dataloader(args, tokenizer, logger)

    def update(engine, batch):
        model.train()
        batch = tuple(torch.tensor(input_data).to(args.device) if idx not in [2, 3] else input_data for idx, input_data in enumerate(batch))
        input_ids, token_type_ids, input_images, image_ids, lm_labels, mc_token_ids, mc_labels = batch
        if args.dhead_gpt2:
            (lm_loss), (mc_loss), *_ = model(input_ids,
                                             token_type_ids=token_type_ids,
                                             mc_token_ids=mc_token_ids,
                                             mc_labels=mc_labels,
                                             lm_labels=lm_labels)
            loss = (lm_loss * args.lm_coef + mc_loss * args.mc_coef) / args.gradient_accumulation_steps
        else:
            (lm_loss), *_ = model(input_ids,
                                  labels=lm_labels,
                                  token_type_ids=token_type_ids,
                                  input_images=input_images,
                                  image_ids=image_ids)
            loss = lm_loss / args.gradient_accumulation_steps
        if args.fp16:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
            torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_norm)
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_norm)
        if engine.state.iteration % args.gradient_accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
        return loss.item() #, optimizer.param_groups[0]['lr']
    trainer = Engine(update)

    # Evaluation function and evaluator (evaluator output is the input of the metrics)
    def inference(engine, batch):
        model.eval()
        with torch.no_grad():
            batch = tuple(input_tensor.to(args.device) for input_tensor in batch)
            input_ids, mc_token_ids, lm_labels, mc_labels, token_type_ids = batch
            # logger.info(tokenizer.decode(input_ids[0, -1, :].tolist()))
            # if we dont send labels to model, it doesnt return losses
            if args.dhead_gpt2:
                lm_logits, mc_logits, *_ = model(
                    input_ids, token_type_ids=token_type_ids, mc_token_ids=mc_token_ids,
                )
                lm_logits_flat_shifted = lm_logits[..., :-1, :].contiguous().view(-1, lm_logits.size(-1))
                lm_labels_flat_shifted = lm_labels[..., 1:].contiguous().view(-1)
                return (lm_logits_flat_shifted, mc_logits), (lm_labels_flat_shifted, mc_labels)
            else:
                lm_logits, *_ = model(input_ids, token_type_ids=token_type_ids)
                lm_logits_flat_shifted = lm_logits[..., :-1, :].contiguous().view(-1, lm_logits.size(-1))
                lm_labels_flat_shifted = lm_labels[..., 1:].contiguous().view(-1)
                return lm_logits_flat_shifted, lm_labels_flat_shifted
    evaluator = Engine(inference)

    # Attach evaluation to trainer: we evaluate when we start the training and at the end of each epoch
    # trainer.add_event_handler(Events.EPOCH_COMPLETED, lambda _: evaluator.run(val_loader))
    if args.n_epochs < 1:
        trainer.add_event_handler(Events.COMPLETED, lambda _: evaluator.run(val_loader))
    if args.eval_before_start:
        trainer.add_event_handler(Events.STARTED, lambda _: evaluator.run(val_loader))

    # Make sure distributed data samplers split the dataset nicely between the distributed processes
    if args.distributed:
        trainer.add_event_handler(Events.EPOCH_STARTED, lambda engine: train_sampler.set_epoch(engine.state.epoch))
        evaluator.add_event_handler(Events.EPOCH_STARTED, lambda engine: valid_sampler.set_epoch(engine.state.epoch))

    # Linearly decrease the learning rate from lr to zero
    model_size = args.n_emd
    noam_lambda = lambda step: (
            model_size ** (-0.5) * min((step + 1) ** (-0.5), (step + 1) * args.warmup_steps ** (-1.5)))
    noam_scheduler = LambdaLR(optimizer, lr_lambda=noam_lambda, last_epoch=args.from_step)
    scheduler = LRScheduler(noam_scheduler)
    if args.scheduler == "linear":
        scheduler = PiecewiseLinear(optimizer, "lr", [(0, args.lr), (args.n_epochs * len(train_loader), 0.0)])
    trainer.add_event_handler(Events.ITERATION_STARTED, scheduler)

    # Prepare metrics - note how we compute distributed metrics
    RunningAverage(output_transform=lambda x: x).attach(trainer, "loss")
    metrics = {"nll": Loss(torch.nn.CrossEntropyLoss(ignore_index=-100), output_transform=lambda x: (x[0][0], x[1][0])),
               "accuracy": Accuracy(output_transform=lambda x: (x[0][1], x[1][1]))}
    metrics.update({"average_nll": MetricsLambda(average_distributed_scalar, metrics["nll"], args),
                    "average_accuracy": MetricsLambda(average_distributed_scalar, metrics["accuracy"], args)})
    metrics["average_ppl"] = MetricsLambda(math.exp, metrics["average_nll"])
    for name, metric in metrics.items():
        metric.attach(evaluator, name)

    # On the main process: add progress bar, tensorboard, checkpoints and save model, configuration and tokenizer before we start to train
    if args.local_rank in [-1, 0]:
        pbar = ProgressBar(persist=True)
        pbar.attach(trainer, metric_names=["loss"])
        evaluator.add_event_handler(Events.COMPLETED, lambda _: pbar.log_message("Validation: %s" % pformat(evaluator.state.metrics)))

        log_dir = make_logdir(args.model_checkpoint)
        tb_logger = TensorboardLogger(log_dir)

        tb_logger.attach(trainer, log_handler=OutputHandler(tag="training", metric_names=["loss"]), event_name=Events.ITERATION_COMPLETED)
        tb_logger.attach(trainer, log_handler=OptimizerParamsHandler(optimizer), event_name=Events.ITERATION_STARTED)
        # tb_logger.attach(evaluator, log_handler=OutputHandler(tag="validation", metric_names=list(metrics.keys()), another_engine=trainer), event_name=Events.EPOCH_COMPLETED)

        checkpoint_handler = ModelCheckpoint(log_dir, 'checkpoint', n_saved=None)
        trainer.add_event_handler(Events.EPOCH_COMPLETED(every=1), checkpoint_handler, {'mymodel': getattr(model, 'module', model)})  # "getattr" takes care of distributed encapsulation

        torch.save(args, log_dir + '/model_training_args.bin')
        getattr(model, 'module', model).config.to_json_file(os.path.join(log_dir, CONFIG_NAME))
        tokenizer.save_pretrained(log_dir)

    # Run the training
    trainer.run(train_loader, max_epochs=args.n_epochs)

    # On the main process: close tensorboard logger and rename the last checkpoint (for easy re-loading with OpenAIGPTModel.from_pretrained method)
    if args.local_rank in [-1, 0] and args.n_epochs > 0:
        os.rename(os.path.join(log_dir, checkpoint_handler._saved[-1][1]), os.path.join(log_dir, WEIGHTS_NAME))  # TODO: PR in ignite to have better access to saved file paths (cleaner)
        tb_logger.close()


if __name__ == "__main__":
    train()
