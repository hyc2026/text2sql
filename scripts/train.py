import os,sys 
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse
import functools
import collections
import datetime
import itertools
import json
import os
import traceback
from typing import Type, List

import _jsonnet
import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter 
import shutil 

# noinspection PyUnresolvedReferences
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader

# noinspection PyUnresolvedReferences
from duorat import datasets

# noinspection PyUnresolvedReferences
from duorat import preproc

# noinspection PyUnresolvedReferences
from duorat import models

# noinspection PyUnresolvedReferences
from duorat.asdl.lang.dusql.spider_transition_system import SpiderTransitionSystem
from duorat.types import RATPreprocItem

# noinspection PyUnresolvedReferences
from duorat.utils import optimizers
from duorat.utils import registry, parallelizer
from duorat.utils import random_state
from duorat.utils import saver as saver_mod
from duorat.utils.evaluation import evaluate_default, load_from_lines
from third_party.spider.evaluation import LEVELS
from scripts.preprocess import Preprocessor
from duorat.api import fix_detokenization
import time 

class TensorboardLogger():

    def __init__(self, log_path=None):
        t = time.strftime("%Y-%m-%d_%H_%M_%S", time.localtime())
        path = os.path.join(log_path, 'tensorboard_log_'+t)
        self.writer = SummaryWriter(path)
    
    def log_train(self, loss, lrs, step):
        self.writer.add_scalar("Loss/train", loss, step)
        for lr in lrs:
            self.writer.add_scalar('lr/'+lr['name'], lr['value'], step)
        
    def log_eval(self, valid_loss, em, prec, recall, f1, step):
        self.writer.add_scalar('Loss/valid', valid_loss, step)
        self.writer.add_scalar('valid/extract_match', em, step)
        self.writer.add_scalar('valid/prec', prec, step)
        self.writer.add_scalar('valid/recall', recall, step)
        self.writer.add_scalar('valid/f1', f1, step)


class Logger:
    def __init__(self, log_path=None, reopen_to_flush=False):
        self.log_file = None
        self.reopen_to_flush = reopen_to_flush
        if log_path is not None:
            os.makedirs(os.path.dirname(log_path), exist_ok=True)
            self.log_file = open(log_path, "a+")

    def log(self, msg):
        formatted = "[{}] {}".format(
            datetime.datetime.now().replace(microsecond=0).isoformat(), msg
        )
        print(formatted)
        if self.log_file:
            self.log_file.write(formatted + "\n")
            if self.reopen_to_flush:
                log_path = self.log_file.name
                self.log_file.close()
                self.log_file = open(log_path, "a+")
            else:
                self.log_file.flush()


class Trainer:
    def __init__(self, logger, tensor_logger,config):
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")

        self.logger = logger
        self.tensor_logger = tensor_logger
      
        self.config = config

        self.data_random = random_state.RandomContext(
            self.config["train"].get("data_seed", None)
        )
        self.model_random = random_state.RandomContext(
            self.config["train"].get("model_seed", None)
        )

        self.init_random = random_state.RandomContext(
            self.config["train"].get("init_seed", None)
        )
        with self.init_random:
            # 0. Construct preprocessors
            self.model_preproc = registry.construct(
                "preproc", self.config["model"]["preproc"],
            )
            self.model_preproc.load()

            # 1. Construct model
            self.model = registry.construct(
                "model", self.config["model"], preproc=self.model_preproc,
            )
            self.model.to(device=device)

    def _log_loss(self, last_step, loss):
        self.logger.log("Step {}: loss={:.8f}".format(last_step, loss))

    def _log_lr(self, last_step, lrs: List[dict]):
        for lr in lrs:
            self.logger.log(
                "Step {}: lr[{}]={:.8f}".format(last_step, lr["name"], lr["value"])
            )

    def _log_stats(self, last_step, eval_section, stats):
        self.logger.log(
            "Step {} stats, {}: {}".format(
                last_step,
                eval_section,
                ", ".join("{} = {}".format(k, v) for k, v in stats.items()),
            )
        )

    def train(self, modeldir, step=None):
        # Save the config info
        with open(
            os.path.join(
                modeldir,
                "config-{}.json".format(
                    datetime.datetime.now().strftime("%Y%m%dT%H%M%S%Z")
                ),
            ),
            "w",
        ) as f:
            json.dump(self.config, f, sort_keys=True, indent=4)

        # slight difference here vs. unrefactored train: The init_random starts over here. Could be fixed if it was important by saving random state at end of init
        with self.init_random:
            # We may be able to move optimizer and lr_scheduler to __init__ instead. Empirically it works fine. I think that's because saver.restore
            # resets the state by calling optimizer.load_state_dict.
            # But, if there is no saved file yet, I think this is not true, so might need to reset the optimizer manually?
            # For now, just creating it from scratch each time is safer and appears to be the same speed, but also means you have to pass in the config to train which is kind of ugly.
            optimizer = registry.construct(
                "optimizer",
                self.config["optimizer"],
                params=[
                    {
                        "name": "no-bert",
                        "params": (
                            parameters
                            for name, parameters in self.model.named_parameters()
                            if not name.startswith("initial_encoder.bert")
                        ),
                    },
                    {
                        "name": "bert",
                        "params": (
                            parameters
                            for name, parameters in self.model.named_parameters()
                            if name.startswith("initial_encoder.bert")
                        ),
                    },
                ],
            )
           
            lr_scheduler = registry.construct(
                "lr_scheduler",
                self.config.get("lr_scheduler", {"name": "noop"}),
                optimizer=optimizer,
            )

        # 1.5. Initialize Automatic Mixed Precision (AMP) training
        scaler = GradScaler(enabled=self.config["train"]["amp_enabled"])

        # 2. Restore model parameters
        saver = saver_mod.Saver(
            self.model, optimizer
        )
        last_step, best_val_all_exact = saver.restore(modeldir, load_best=True, step=step)
        if last_step is 0 and self.config["train"].get("initialize_from", False):
            saver.restore(self.config["train"]["initialize_from"])
            self.logger.log(
                "Model initialized from {}".format(
                    self.config["train"]["initialize_from"]
                )
            )
        else:
            self.logger.log(f"Model restored, the last step is {last_step}, best val_all_exact is {best_val_all_exact}")

        # 3. Get training data somewhere
        with self.data_random:
            data_splits = self.config["train"].get("data_splits", None)
            if data_splits is not None:
                self.logger.log(f"Using custom training data splits: {data_splits}.")
            else:
                data_splits = [
                    section for section in self.config["data"] if "train" in section
                ]
            train_data = list(
                itertools.chain.from_iterable(
                    self.model_preproc.dataset(split) for split in data_splits
                )
            )
            self.logger.log(f"{len(train_data)} training examples")

            train_data_loader = self._yield_batches_from_epochs(
                DataLoader(
                    train_data,
                    batch_size=self.config["train"]["batch_size"],
                    shuffle=True,
                    drop_last=True,
                    collate_fn=lambda x: x,
                )
            )
        train_eval_data_loader = DataLoader(
            train_data,
            batch_size=self.config["train"]["eval_batch_size"],
            collate_fn=lambda x: x,
        )
        val_data = self.model_preproc.dataset("val")
        val_data_loader = DataLoader(
            val_data,
            batch_size=self.config["train"]["eval_batch_size"],
            collate_fn=lambda x: x,
        )

        def _evaluate_model():
            # A model that is not evaluated (if eval_on_val=False, or if step < infer_min_n)
            # is given a performance of 0
            val_all_exact = 0
            if self.config["train"]["eval_on_train"]:
                self.logger.log("Evaluate on the training set")
                self._eval_model(modeldir, last_step, train_eval_data_loader, "train")
            if self.config["train"]["eval_on_val"]:
                self.logger.log("Evaluate on the validation set")
                val_all_exact = self._eval_model(modeldir, last_step, val_data_loader, "val")
            return val_all_exact
        """ 
        # 是否训练前先验证一下
        val_all_exact = _evaluate_model()
        saver.save(
             modeldir,
             last_step,
             is_best=val_all_exact > best_val_all_exact,
             best_validation_metric=max(val_all_exact, best_val_all_exact)
         )
        best_val_all_exact = max(val_all_exact, best_val_all_exact)
        """
        # Counter for grad aggregation
        grad_accumulation_counter = 0
        losses = []

        # 4. Start training loop
        with self.data_random:
            self.logger.log("Enter training loop")
            for batch in train_data_loader:
                # Quit if too long
                if last_step >= self.config["train"]["max_steps"]:
                    break

                # Compute and apply gradient
                with self.model_random:
                    with autocast(enabled=self.config["train"]["amp_enabled"]):
                        loss = self.model.compute_loss(batch)
                        loss /= self.config["train"]["n_grad_accumulation_steps"]

                scaler.scale(loss).backward()

                losses.append(
                    loss.item() * self.config["train"]["n_grad_accumulation_steps"]
                )

                grad_accumulation_counter += 1
                # Update params every `n_grad_accumulation_steps` times
                if (
                    grad_accumulation_counter
                    % self.config["train"]["n_grad_accumulation_steps"]
                    == 0
                ):
                    # may call unscale_ here to allow clipping unscaled gradients,
                    # see https://pytorch.org/docs/stable/notes/amp_examples.html#gradient-accumulation
                    # torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

                    lr_scheduler.update_lr(last_step)
                    try:
                        scaler.step(optimizer)
                        scaler.update()
                    except RuntimeError as e:
                        self.logger.log(
                            f"Caught error '{e}' while updating model. Continuing training ..."
                        )
                    optimizer.zero_grad()

                    last_step += 1
                    # Report metrics
                    cur_loss = np.mean(losses)
                    cur_lrs = [
                        {"name": param_group["name"], "value": param_group["lr"]}
                        for param_group in optimizer.param_groups
                    ]
                    if last_step % self.config["train"]["report_every_n"] == 0:
                        self._log_stats(
                            last_step,
                            "train",
                            {
                                "step": last_step,
                            },
                        )
                        self._log_loss(last_step, cur_loss)
                        self._log_lr(last_step, cur_lrs)
                        self.tensor_logger.log_train(cur_loss, cur_lrs, last_step)

                    # Evaluate model
                    if last_step % self.config["train"]["eval_every_n"] == 0:
                        val_all_exact = _evaluate_model()
                        # Run saver
                        saver.save(
                            modeldir,
                            last_step,
                            is_best=val_all_exact > best_val_all_exact,
                            best_validation_metric=max(val_all_exact, best_val_all_exact)
                        )
                        best_val_all_exact = max(val_all_exact, best_val_all_exact)
                        
                    # Reset the list of losses
                    losses = []

    @staticmethod
    def _yield_batches_from_epochs(loader):
        while True:
            for batch in loader:
                yield batch

    def _eval_model(self, modeldir, last_step, eval_data_loader, eval_section):
        num_eval_items = self.config["train"]["num_eval_items"]
        stats = collections.defaultdict(float)
        self.model.eval()
        with torch.no_grad():
            for eval_batch in eval_data_loader:
                batch_res = self.model.eval_on_batch(eval_batch)
                for k, v in batch_res.items():
                    stats[k] += v
                if num_eval_items and stats["total"] > num_eval_items:
                    break
        self.model.train()

        # Divide each stat by 'total'
        for k in stats:
            if k != "total":
                stats[k] /= stats["total"]
        if "total" in stats:
            del stats["total"]

        all_exact = 0
        if last_step >= self.config["train"]["infer_min_n"]:
            metrics = self._infer(modeldir, last_step, eval_section)
            stats.update(
                {
                    "{}_exact".format(level): metrics["total_scores"][level]["exact"]
                    for level in LEVELS
                }

            )
            
            partial_scores = metrics['total_scores']['partial_scores']
            
            partial_acc = partial_scores['acc']/partial_scores['acc_count']
            partial_rec = partial_scores['rec']/partial_scores['rec_count']
            partial_f1 = 2*partial_acc*partial_rec/(partial_acc+partial_rec)
            stats.update({'partial_score':'acc:{}, rec:{}, f1:{}'.format(partial_acc, partial_rec, partial_f1)}
            )
            all_exact = metrics["total_scores"]["all"]["exact"]
            self.tensor_logger.log_eval(
                                stats['loss'], 
                                metrics["total_scores"]['all']["exact"],
                                partial_acc, 
                                partial_rec, 
                                partial_f1,
                                last_step
                                )
        self._log_stats(last_step, eval_section, stats)
            
        return all_exact

    def _infer(self, modeldir, last_step, eval_section):
        self.logger.log("Inferring...")

        orig_data = registry.construct("dataset", self.config["data"][eval_section])
        preproc_data: List[RATPreprocItem] = self.model_preproc.dataset(eval_section)

        # preproc_data = preproc_data[:30]
        # orig_data.examples = orig_data.examples[:30]

        self.model.eval()
        with torch.no_grad():
            if isinstance(self.model.preproc.transition_system, SpiderTransitionSystem):
                # Orig data only used with SpiderTransitionSystem
                assert len(orig_data) == len(preproc_data)
            inferred_lines = list(
                self._inner_infer(
                    model=self.model,
                    orig_data=orig_data,
                    preproc_data=preproc_data,
                    nproc=self.config["train"]["eval_nproc"],
                    beam_size=self.config["train"]["eval_beam_size"],
                    decode_max_time_step=self.config["train"][
                        "eval_decode_max_time_step"
                    ],
                )
            )
            infer_res = [x[1] for x in inferred_lines]
            inferred_lines = [x[0] for x in inferred_lines]
        self.model.train()

        with open(f"{modeldir}/output-{last_step}", "w") as output_dst:
            for line in infer_res:
                output_dst.write(line)

        if isinstance(self.model.preproc.transition_system, SpiderTransitionSystem):
            final_res, eval_res = evaluate_default(orig_data, load_from_lines(inferred_lines))
            
            with open(f"{modeldir}/output-{last_step}-error.txt", "w") as output_dst:
                for res in eval_res:
                    if res[-1] == 0:
                        output_dst.write('quesion_id:'+res[0]+'\n')
                        output_dst.write('gold:'+res[1]+'\n')
                        output_dst.write('pred:'+res[2]+'\n\n')
                        
            return final_res 
        else:
            raise NotImplementedError

    @classmethod
    def _inner_infer(
        cls, model, orig_data, preproc_data, nproc, beam_size, decode_max_time_step
    ):
        if torch.cuda.is_available():
            cp = parallelizer.CUDAParallelizer(nproc)
        else:
            cp = parallelizer.CPUParallelizer(nproc)
        return cp.parallel_map(
            [
                (
                    functools.partial(
                        cls._parse_single2,
                        model,
                        beam_size=beam_size,
                        decode_max_time_step=decode_max_time_step,
                    ),
                    list(enumerate(zip(orig_data, preproc_data))),
                )
            ]
        )

    @staticmethod
    def _parse_single(model, item, beam_size, decode_max_time_step):
        index, (orig_item, preproc_item) = item
        batch = [preproc_item]

        try:
            beams = model.parse(batch, decode_max_time_step, beam_size)

            decoded = []
            for beam in beams:
                asdl_ast = beam.ast
                if isinstance(model.preproc.transition_system, SpiderTransitionSystem):
                    tree = model.preproc.transition_system.ast_to_surface_code(
                        asdl_ast=asdl_ast
                    )
                    inferred_code = model.preproc.transition_system.spider_grammar.unparse(
                        tree=tree, spider_schema=orig_item.spider_schema
                    )
                    inferred_code_readable = ""
                else:
                    raise NotImplementedError

                decoded.append(
                    {
                        "model_output": asdl_ast.pretty(),
                        "inferred_code": fix_detokenization(inferred_code),
                        "inferred_code_readable": inferred_code_readable,
                        "score": beam.score,
                    }
                )
            result = {
                "index": index,
                "beams": decoded,
            }
        except Exception as e:
            result = {"index": index, "error": str(e), "trace": traceback.format_exc()}
        return json.dumps(result, ensure_ascii=False) + "\n"

    # 将输出格式改为提交要求格式
    @staticmethod
    def _parse_single2(model, item, beam_size, decode_max_time_step):
        index, (orig_item, preproc_item) = item
        batch = [preproc_item]
        question_id = orig_item.orig['question_id']
        try:
            beams = model.parse(batch, decode_max_time_step, beam_size)

            decoded = []
            for beam in beams:
                asdl_ast = beam.ast
                if isinstance(model.preproc.transition_system, SpiderTransitionSystem):
                    tree = model.preproc.transition_system.ast_to_surface_code(
                        asdl_ast=asdl_ast
                    )
                    inferred_code = model.preproc.transition_system.spider_grammar.unparse(
                        tree=tree, spider_schema=orig_item.spider_schema
                    )
                    inferred_code_readable = ""
                else:
                    raise NotImplementedError

                decoded.append(
                    {
                        "model_output": asdl_ast.pretty(),
                        "inferred_code": fix_detokenization(inferred_code),
                        "inferred_code_readable": inferred_code_readable,
                        "score": beam.score,
                    }
                )

            infer_result = {
                "index": index,
                "beams": decoded,
            }   
            result = question_id+'\t'+decoded[0]['inferred_code']+'\n'
        except Exception as e:
            infer_result = {"index": index, "error": str(e), "trace": traceback.format_exc()}
            result = question_id + '\t' + "SELECT * FROM 娱乐明星" + '\n'
        return json.dumps(infer_result, ensure_ascii=False) + "\n", result


def main(
    args=None, logdir_suffix: List[str] = None, trainer_class: Type[Trainer] = Trainer
):
    parser = argparse.ArgumentParser()
    parser.add_argument("--logdir", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--config-args")
    parser.add_argument("--step")
    args = parser.parse_args(args)

    # 此后关于 config 的叙述均根据配置文件 duorat-finetune-bert-large.jsonnet

    if args.config_args:
        config = json.loads(
            _jsonnet.evaluate_file(args.config, tla_codes={"args": args.config_args})
        )
    else:
        config = json.loads(_jsonnet.evaluate_file(args.config))

    if logdir_suffix:
        args.logdir = os.path.join(args.logdir, *logdir_suffix)

    if "model_name" in config:
        args.logdir = os.path.join(args.logdir, config["model_name"])

    # Initialize the logger
    reopen_to_flush = config.get("log", {}).get("reopen_to_flush")
    logger = Logger(os.path.join(args.logdir, "log.txt"), reopen_to_flush)
    tensor_logger = TensorboardLogger(args.logdir)

    logger.log("Logging to {}".format(args.logdir))

    preproc_data_path = os.path.join(args.logdir, "data")
    logger.log(f"Overwriting preproc save_path with: {preproc_data_path}")
    config['model']['preproc']['save_path'] = preproc_data_path

    if os.path.exists(preproc_data_path):
        logger.log("Skip preprocessing..")
    else:
        logger.log("Running preprocessing...")
        sections = config["data"].keys()
        keep_vocab = False
        preprocessor = Preprocessor(config)
        preprocessor.preprocess(sections, keep_vocab)
        print('finish preprocessing!')

    #Construct trainer and do training
    trainer = trainer_class(logger, tensor_logger, config)
    trainer.train(modeldir=args.logdir, step=args.step)


if __name__ == "__main__":
    # python scripts/train.py --config configs/duorat/duorat-finetune-bert-large.jsonnet --logdir /logdir/duorat-bert
    main()
