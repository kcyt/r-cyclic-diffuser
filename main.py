import argparse, os, sys, datetime, glob, importlib, csv
import numpy as np
import time
import torch
import torchvision
import copy
from packaging import version
import shutil

from omegaconf import OmegaConf
from torch.utils.data import random_split, DataLoader, Dataset, Subset
from torch.utils.data.dataloader import default_collate
from functools import partial
from PIL import Image

# make sure you are using  pytorch 2 and its associated pytorch_lightning version.
import pytorch_lightning as pl
from pytorch_lightning.utilities.deepspeed import convert_zero_checkpoint_to_fp32_state_dict
from pytorch_lightning.utilities.distributed import rank_zero_only
from pytorch_lightning import seed_everything
from pytorch_lightning.trainer import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, Callback, LearningRateMonitor
from pytorch_lightning.utilities import rank_zero_info

from ldm.data.base import Txt2ImgIterableBaseDataset
from ldm.util import instantiate_from_config


NUM_OF_BLENDWEIGHT_MAPS_TO_USE = 4 


@rank_zero_only
def rank_zero_print(*args):
    print(*args)

def modify_weights(w, scale = 1e-6, n=2):
    """Modify weights to accomodate concatenation to unet"""
    extra_w = scale*torch.randn_like(w)
    new_w = w.clone()
    for i in range(n):
        new_w = torch.cat((new_w, extra_w.clone()), dim=1)
    return new_w

def get_parser(**parser_kwargs):
    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ("yes", "true", "t", "y", "1"):
            return True
        elif v.lower() in ("no", "false", "f", "n", "0"):
            return False
        else:
            raise argparse.ArgumentTypeError("Boolean value expected.")

    parser = argparse.ArgumentParser(**parser_kwargs)
    parser.add_argument(
        "--finetune_from",
        type=str,
        nargs="?",
        default="",
        help="path to checkpoint to load model state from"
    )
    parser.add_argument(
        "-n",
        "--name",
        type=str,
        const=True,
        default="",
        nargs="?",
        help="postfix for logdir",
    )
    parser.add_argument(
        "-r",
        "--resume",
        type=str,
        const=True,
        default="",
        nargs="?",
        help="resume from logdir or checkpoint in logdir",
    )
    parser.add_argument(
        "-b",
        "--base",
        nargs="*",
        metavar="base_config.yaml",
        help="paths to base configs. Loaded from left-to-right. "
             "Parameters can be overwritten or added with command-line options of the form `--key value`.",
        default=list(),
    )
    parser.add_argument(
        "-t",
        "--train",
        type=str2bool,
        const=True,
        default=False,
        nargs="?",
        help="train",
    )
    parser.add_argument(
        "--no-test",
        type=str2bool,
        const=True,
        default=False,
        nargs="?",
        help="disable test",
    )
    parser.add_argument(
        "-p",
        "--project",
        help="name of new or path to existing project"
    )
    parser.add_argument(
        "-d",
        "--debug",
        type=str2bool,
        nargs="?",
        const=True,
        default=False,
        help="enable post-mortem debugging",
    )
    parser.add_argument(
        "-s",
        "--seed",
        type=int,
        default=23,
        help="seed for seed_everything",
    )
    parser.add_argument(
        "-f",
        "--postfix",
        type=str,
        default="",
        help="post-postfix for default name",
    )
    parser.add_argument(
        "-l",
        "--logdir",
        type=str,
        default="logs",
        help="directory for logging dat shit",
    )


    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help="resolution of image",
    )

    return parser


def nondefault_trainer_args(opt):
    parser = argparse.ArgumentParser()
    parser = Trainer.add_argparse_args(parser)
    args = parser.parse_args([])
    return sorted(k for k in vars(args) if getattr(opt, k) != getattr(args, k))


class WrappedDataset(Dataset):
    """Wraps an arbitrary object with __len__ and __getitem__ into a pytorch dataset"""

    def __init__(self, dataset):
        self.data = dataset

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def worker_init_fn(_):
    worker_info = torch.utils.data.get_worker_info()

    dataset = worker_info.dataset
    worker_id = worker_info.id

    if isinstance(dataset, Txt2ImgIterableBaseDataset):
        split_size = dataset.num_records // worker_info.num_workers
        # reset num_records to the true number to retain reliable length information
        dataset.sample_ids = dataset.valid_ids[worker_id * split_size:(worker_id + 1) * split_size]
        current_id = np.random.choice(len(np.random.get_state()[1]), 1)
        return np.random.seed(np.random.get_state()[1][current_id] + worker_id)
    else:
        return np.random.seed(np.random.get_state()[1][0] + worker_id)


class DataModuleFromConfig(pl.LightningDataModule):
    def __init__(self, batch_size, train=None, validation=None, test=None, predict=None,
                 wrap=False, num_workers=None, shuffle_test_loader=False, use_worker_init_fn=False,
                 shuffle_val_dataloader=False):
        super().__init__()
        self.batch_size = batch_size
        self.dataset_configs = dict()
        self.num_workers = num_workers if num_workers is not None else batch_size * 2
        self.use_worker_init_fn = use_worker_init_fn
        if train is not None:
            self.dataset_configs["train"] = train
            self.train_dataloader = self._train_dataloader
        if validation is not None:
            self.dataset_configs["validation"] = validation
            self.val_dataloader = partial(self._val_dataloader, shuffle=shuffle_val_dataloader)
        if test is not None:
            self.dataset_configs["test"] = test
            self.test_dataloader = partial(self._test_dataloader, shuffle=shuffle_test_loader)
        if predict is not None:
            self.dataset_configs["predict"] = predict
            self.predict_dataloader = self._predict_dataloader
        self.wrap = wrap
        #self.train_index_for_test = 0

    def set_is_train(self, train_flag, dataset_type="train"):
        self.datasets[dataset_type].set_is_train(train_flag)

    def get_sample_for_test(self, dataset_type="train"):
        original_train_flag = self.datasets[dataset_type].get_is_train()
        if original_train_flag: # Currently, dataset is in train mode
            self.set_is_train(train_flag=False, dataset_type=dataset_type) # set the dataset to test mode
                
        example = self.datasets[dataset_type].get_sample_for_test()

        self.set_is_train(train_flag=original_train_flag, dataset_type=dataset_type) # set the dataset to the original mode
        return example

    def prepare_data(self):
        for data_cfg in self.dataset_configs.values():
            instantiate_from_config(data_cfg)

    def setup(self, stage=None):
        self.datasets = dict(
            (k, instantiate_from_config(self.dataset_configs[k]))
            for k in self.dataset_configs)
        if self.wrap:
            for k in self.datasets:
                self.datasets[k] = WrappedDataset(self.datasets[k])

    def _train_dataloader(self):
        is_iterable_dataset = isinstance(self.datasets['train'], Txt2ImgIterableBaseDataset)
        if is_iterable_dataset or self.use_worker_init_fn:
            init_fn = worker_init_fn
        else:
            init_fn = None
        return DataLoader(self.datasets["train"], batch_size=self.batch_size,
                          num_workers=self.num_workers, shuffle=False if is_iterable_dataset else True,
                          worker_init_fn=init_fn)

    def _val_dataloader(self, shuffle=False):
        if isinstance(self.datasets['validation'], Txt2ImgIterableBaseDataset) or self.use_worker_init_fn:
            init_fn = worker_init_fn
        else:
            init_fn = None
        return DataLoader(self.datasets["validation"],
                          batch_size=self.batch_size,
                          num_workers=self.num_workers,
                          worker_init_fn=init_fn,
                          shuffle=shuffle)

    def _test_dataloader(self, shuffle=False):
        is_iterable_dataset = isinstance(self.datasets['train'], Txt2ImgIterableBaseDataset)
        if is_iterable_dataset or self.use_worker_init_fn:
            init_fn = worker_init_fn
        else:
            init_fn = None

        # do not shuffle dataloader for iterable dataset
        shuffle = shuffle and (not is_iterable_dataset)

        return DataLoader(self.datasets["test"], batch_size=self.batch_size,
                          num_workers=self.num_workers, worker_init_fn=init_fn, shuffle=shuffle)

    def _predict_dataloader(self, shuffle=False):
        if isinstance(self.datasets['predict'], Txt2ImgIterableBaseDataset) or self.use_worker_init_fn:
            init_fn = worker_init_fn
        else:
            init_fn = None
        return DataLoader(self.datasets["predict"], batch_size=self.batch_size,
                          num_workers=self.num_workers, worker_init_fn=init_fn)


class SetupCallback(Callback):
    def __init__(self, resume, now, logdir, ckptdir, cfgdir, config,
                 lightning_config, debug):
        super().__init__()
        self.resume = resume
        self.now = now
        self.logdir = logdir
        self.ckptdir = ckptdir
        self.cfgdir = cfgdir
        self.config = config
        self.lightning_config = lightning_config
        self.debug = debug

    def on_keyboard_interrupt(self, trainer, pl_module):
        if not self.debug and trainer.global_rank == 0:
            rank_zero_print("Summoning checkpoint.")
            ckpt_path = os.path.join(self.ckptdir, "last.ckpt")
            trainer.save_checkpoint(ckpt_path)





    def _routine_before_training(self, trainer, pl_module):
        if trainer.global_rank == 0:
            # Create logdirs and save configs
            os.makedirs(self.logdir, exist_ok=True)
            os.makedirs(self.ckptdir, exist_ok=True)
            os.makedirs(self.cfgdir, exist_ok=True)

            if "callbacks" in self.lightning_config:
                if 'metrics_over_trainsteps_checkpoint' in self.lightning_config['callbacks']:
                    os.makedirs(os.path.join(self.ckptdir, 'trainstep_checkpoints'), exist_ok=True)
            rank_zero_print("Project config")
            rank_zero_print(OmegaConf.to_yaml(self.config))

            OmegaConf.save(self.config,
                           os.path.join(self.cfgdir, "{}-project.yaml".format(self.now)))

            rank_zero_print("Lightning config")
            rank_zero_print(OmegaConf.to_yaml(self.lightning_config))
            OmegaConf.save(OmegaConf.create({"lightning": self.lightning_config}),
                           os.path.join(self.cfgdir, "{}-lightning.yaml".format(self.now)))

        else:
            # ModelCheckpoint callback created log directory --- remove it
            if not self.resume and os.path.exists(self.logdir):
                dst, name = os.path.split(self.logdir)
                dst = os.path.join(dst, "child_runs", name)
                os.makedirs(os.path.split(dst)[0], exist_ok=True)
                try:
                    os.rename(self.logdir, dst)
                except FileNotFoundError:
                    pass



    def on_pretrain_routine_start(self, trainer, pl_module):
        self._routine_before_training(trainer, pl_module)


class ImageLogger(Callback):
    def __init__(self, batch_frequency, max_images, clamp=True, increase_log_steps=True,
                 rescale=True, disabled=False, log_on_batch_idx=False, log_first_step=False,
                 log_images_kwargs=None, log_all_val=False, testtube_no_image_log=True, 
                 to_validate_the_images_only=False, seed=0):
        super().__init__()
        self.rescale = rescale
        self.batch_freq = batch_frequency
        self.max_images = max_images

        self.logger_log_images = {
                pl.loggers.TestTubeLogger: self._testtube,
            }

        
        self.log_steps = [2 ** n for n in range(int(np.log2(self.batch_freq)) + 1)] # E.g. if self.batch_freq == 1000, then self.log_steps = [1,2,4,8,16,32,64,128,256,512] 
        if not increase_log_steps:
            self.log_steps = [self.batch_freq]
        self.clamp = clamp
        self.disabled = disabled
        self.log_on_batch_idx = log_on_batch_idx
        self.log_images_kwargs = log_images_kwargs if log_images_kwargs else {}
        self.log_first_step = log_first_step
        self.log_all_val = log_all_val

        self.testtube_no_image_log = testtube_no_image_log
        self.to_validate_the_images_only = to_validate_the_images_only

        self.seed = seed

        if self.to_validate_the_images_only:
            self.batch_freq = 1  
            self.log_all_val = True 

    @rank_zero_only
    def _testtube(self, pl_module, images, batch_idx, split):
        for k in images:
            grid = torchvision.utils.make_grid(images[k])
            grid = (grid + 1.0) / 2.0  # -1,1 -> 0,1; c,h,w

            tag = f"{split}/{k}"
            pl_module.logger.experiment.add_image(
                tag, grid,
                global_step=pl_module.global_step)

    @rank_zero_only
    def log_local(self, save_dir, split, images,
                  global_step, current_epoch, batch_idx, angle_index=-1):
        root = os.path.join(save_dir, "images", split)
        for k in images: 
            grid = torchvision.utils.make_grid(images[k], nrow=4) # grid is (c,h,w) bur images[k] is (b,c,h,w)
            if self.rescale: # is True
                grid = (grid + 1.0) / 2.0  # -1,1 -> 0,1; c,h,w
            grid = grid.transpose(0, 1).transpose(1, 2).squeeze(-1) # (h,w,c)
            grid = grid.numpy()
            grid = (grid * 255).astype(np.uint8)
            filename = "{}_gs-{:06}_e-{:06}_b-{:06}.png".format(
                k,
                global_step,
                current_epoch,
                batch_idx)

            if angle_index > -1:
                filename = filename.replace('.png', '_{0}_.png'.format(angle_index) )

            path = os.path.join(root, filename)
            os.makedirs(os.path.split(path)[0], exist_ok=True)

            Image.fromarray(grid).save(path)

    def log_img(self, trainer, pl_module, batch, batch_idx, split="train"):
        check_idx = batch_idx if self.log_on_batch_idx else pl_module.global_step
        if self.log_all_val and split == "val":
            should_log = True
        else:
            should_log = self.check_frequency(check_idx)
        
        if (should_log and  (check_idx % self.batch_freq == 0) and
                hasattr(pl_module, "log_images") and
                callable(pl_module.log_images) and
                self.max_images > 0):

            del batch 
            
            _split = split
            if _split == "val":
                _split = "validation"
            
            batch = trainer.datamodule.get_sample_for_test(dataset_type=_split)
            batch = default_collate([batch])


            logger = type(pl_module.logger)

            is_train = pl_module.training
            if is_train:
                pl_module.eval()



            batches_list = [batch]
            angle_index = -1


            for curr_batch in batches_list:
                with torch.no_grad():
                    images = pl_module.log_images(curr_batch, split=split, seed=self.seed, **self.log_images_kwargs)

                for k in images:
                    N = min(images[k].shape[0], self.max_images)
                    images[k] = images[k][:N]
                    if isinstance(images[k], torch.Tensor):
                        images[k] = images[k].detach().cpu()
                        if self.clamp:
                            images[k] = torch.clamp(images[k], -1., 1.)

                self.log_local(pl_module.logger.save_dir, split, images,
                               pl_module.global_step, pl_module.current_epoch, batch_idx, angle_index=angle_index)

                angle_index += 1

                if self.testtube_no_image_log: # True by default
                    pass 
                else: # log images into TestTube or Tensorboard
                    logger_log_images = self.logger_log_images.get(logger, lambda *args, **kwargs: None)
                    logger_log_images(pl_module, images, pl_module.global_step, split)



            if is_train:
                pl_module.train()

    def check_frequency(self, check_idx):

        if ((check_idx % self.batch_freq) == 0 or (check_idx in self.log_steps)) and (
                check_idx > 0 or self.log_first_step):
            try:
                self.log_steps.pop(0)
            except IndexError as e:
                rank_zero_print(e)
                pass
            return True
        return False



    def _on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=None):
        if not self.disabled and (pl_module.global_step > 0 or self.log_first_step):
            self.log_img(trainer, pl_module, batch, batch_idx, split="train")



    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx): # a pytorch lightning function
        self._on_train_batch_end(trainer, pl_module, outputs, batch, batch_idx, dataloader_idx)




    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):  # a pytorch lightning function
        
        if self.to_validate_the_images_only:
            self.log_img(trainer, pl_module, batch, batch_idx, split="val")
        else:

            if not self.disabled and pl_module.global_step > 0:
                self.log_img(trainer, pl_module, batch, batch_idx, split="val")
            if hasattr(pl_module, 'calibrate_grad_norm'):
                if (pl_module.calibrate_grad_norm and batch_idx % 25 == 0) and batch_idx > 0:
                    self.log_gradients(trainer, pl_module, batch_idx=batch_idx)


    @rank_zero_only
    def on_validation_epoch_end(self, trainer, pl_module):
        if self.to_validate_the_images_only:

            image_folder = os.path.join(pl_module.logger.save_dir, 'images' ,'val')
            #os.makedirs(image_folder, exist_ok=True)

            from subprocess import Popen
            validation_process = Popen(
                ['nohup' , 'python' , 'validate_images_helper.py', image_folder ],
                stdout = open('{0}/validation_logfile.txt'.format(image_folder) , 'a' ),
                stderr = open('{0}/debugging_error_logfile.txt'.format(image_folder), 'a'),
                start_new_session=True )
            print("Validation Process created!")

            print("Waiting for Validation Process before exiting..")
            validation_process.wait()
            print("Validation Process is completed ... Exiting!")


            raise Exception("Script to_validate_the_images_only has ended")



class CUDACallback(Callback):


    # see https://github.com/SeanNaren/minGPT/blob/master/mingpt/callback.py
    def on_train_epoch_start(self, trainer, pl_module):
        trainer_root_gpu = trainer.root_gpu

        # Reset the memory use counter
        torch.cuda.reset_peak_memory_stats(trainer_root_gpu)
        torch.cuda.synchronize(trainer_root_gpu)
        self.start_time = time.time()




    def _on_train_epoch_end(self, trainer_root_gpu, trainer, pl_module, outputs=None):
        torch.cuda.synchronize(trainer_root_gpu)
        max_memory = torch.cuda.max_memory_allocated(trainer_root_gpu) / 2 ** 20
        epoch_time = time.time() - self.start_time

        try:
            max_memory = trainer.training_type_plugin.reduce(max_memory)
            epoch_time = trainer.training_type_plugin.reduce(epoch_time)

            rank_zero_info(f"Average Epoch time: {epoch_time:.2f} seconds")
            rank_zero_info(f"Average Peak memory {max_memory:.2f}MiB")
        except AttributeError:
            pass


    def on_train_epoch_end(self, trainer, pl_module, outputs):
        trainer_root_gpu = trainer.root_gpu
        self._on_train_epoch_end(trainer_root_gpu, trainer, pl_module, outputs)




if __name__ == "__main__":
    # custom parser to specify config files, train, test and debug mode,
    # postfix, resume.
    # `--key value` arguments are interpreted as arguments to the trainer.
    # `nested.key=value` arguments are interpreted as config parameters.
    # configs are merged from left-to-right followed by command line parameters.

    # model:
    #   base_learning_rate: float
    #   target: path to lightning module
    #   params:
    #       key: value
    # data:
    #   target: main.DataModuleFromConfig
    #   params:
    #      batch_size: int
    #      wrap: bool
    #      train:
    #          target: path to train dataset
    #          params:
    #              key: value
    #      validation:
    #          target: path to validation dataset
    #          params:
    #              key: value
    #      test:
    #          target: path to test dataset
    #          params:
    #              key: value
    # lightning: (optional, has sane defaults and can be specified on cmdline)
    #   trainer:
    #       additional arguments to trainer
    #   logger:
    #       logger to instantiate
    #   modelcheckpoint:
    #       modelcheckpoint to instantiate
    #   callbacks:
    #       callback1:
    #           target: importpath
    #           params:
    #               key: value

    now = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")

    # add cwd for convenience and to make classes in this file available when
    # running as `python main.py`
    # (in particular `main.DataModuleFromConfig`)
    sys.path.append(os.getcwd())

    parser = get_parser()
    parser = Trainer.add_argparse_args(parser) # extends the parser's arguments with the default arguments of pytorch_lightning's Trainer class.

    opt, unknown = parser.parse_known_args()
    if opt.name and opt.resume:
        raise ValueError(
            "-n/--name and -r/--resume cannot be specified both."
            "If you want to resume training in a new log folder, "
            "use -n/--name in combination with --resume_from_checkpoint"
        )
    if opt.resume: 
        if not os.path.exists(opt.resume):
            raise ValueError("Cannot find {}".format(opt.resume))
        if os.path.isfile(opt.resume):  # e.g. opt.resume = "/home/astar/Documents/zero123-main/zero123/logs/2023-08-26T16-46-58_trial_sd-objaverse-finetune-c_concat-256/checkpoints/last.ckpt"
            paths = opt.resume.split("/")
            # idx = len(paths)-paths[::-1].index("logs")+1
            # logdir = "/".join(paths[:idx])
            logdir = "/".join(paths[:-2])
            ckpt = opt.resume
        else: # e.g. opt.resume = "/home/astar/Documents/zero123-main/zero123/logs/2023-08-26T16-46-58_trial_sd-objaverse-finetune-c_concat-256
            assert os.path.isdir(opt.resume), opt.resume
            logdir = opt.resume.rstrip("/")
            ckpt = os.path.join(logdir, "checkpoints", "last.ckpt")

        opt.resume_from_checkpoint = ckpt # "resume_from_checkpoint" is an argument that would be later sent to the Trainer object. Trainer object has a predefined "resume_from_checkpoint" parameter that, if not None, will load the checkpoint from the filepath specified in "resume_from_checkpoint".
        
        # load the configs used by the loaded checkpoint
        base_configs = sorted(glob.glob(os.path.join(logdir, "configs/*.yaml")))
        opt.base = base_configs + opt.base

        _tmp = logdir.split("/")
        nowname = _tmp[-1]
    else: # New experiment, not loading from a checkpoint
        if opt.name:
            name = "_" + opt.name
        elif opt.base:
            cfg_fname = os.path.split(opt.base[0])[-1] # e.g. trial_sd-objaverse-finetune-c_concat-256.yaml
            cfg_name = os.path.splitext(cfg_fname)[0] # e.g. trial_sd-objaverse-finetune-c_concat-256
            name = "_" + cfg_name
        else:
            name = ""
        nowname = now + name + opt.postfix
        logdir = os.path.join(opt.logdir, nowname)

    ckptdir = os.path.join(logdir, "checkpoints")
    cfgdir = os.path.join(logdir, "configs")
    seed_everything(opt.seed)

    try:
        # init and save configs
        configs = [OmegaConf.load(cfg) for cfg in opt.base]
        cli = OmegaConf.from_dotlist(unknown)
        config = OmegaConf.merge(*configs, cli)

        config.lightning.callbacks.image_logger.params['seed'] = opt.seed


        to_validate_the_images_only = config.get("to_validate_the_images_only", False) 
        if to_validate_the_images_only:
 
            config.data.params.validation.target = "ldm.data.thuman.ThumanDatasetNovelValidation"
            config.data.params.validation.params.justFrontal = True
            
            config.lightning.trainer.num_sanity_val_steps = -1 
            config.lightning.callbacks.image_logger.params['to_validate_the_images_only'] = to_validate_the_images_only
            config["model"]["params"]["to_validate_the_images_only"] = True


        lightning_config = config.pop("lightning", OmegaConf.create())
        # merge trainer cli with config
        trainer_config = lightning_config.get("trainer", OmegaConf.create())

        config_model_params = config["model"]["params"]

        set_smplx_conditioning = config_model_params.get("set_smplx_conditioning", False)
        if set_smplx_conditioning:
            set_smplx_conditioning_suboption_useBlendweights = config_model_params.get("set_smplx_conditioning_suboption_useBlendweights", False)
            if set_smplx_conditioning_suboption_useBlendweights:
                config["model"]["params"]["unet_config"]["params"]["in_channels"] += 3*NUM_OF_BLENDWEIGHT_MAPS_TO_USE  

                config["data"]["params"]["train"]["params"]["useBlendweights"] = True
                config["data"]["params"]["validation"]["params"]["useBlendweights"] = True
            else:
                config["model"]["params"]["unet_config"]["params"]["in_channels"] += 3 


            print("No. of in-channels in U-Net modified due to use of smplx conditioning!")



        set_sequential_conditioning = config_model_params.get("set_sequential_conditioning", False)
        set_sequential_conditioning_suboption_useCrossAttention = config_model_params.get("set_sequential_conditioning_suboption_useCrossAttention", False)
        set_sequential_conditioning_suboption_useSmplx = config_model_params.get("set_sequential_conditioning_suboption_useSmplx", False)
        if (set_smplx_conditioning == False) and (set_sequential_conditioning_suboption_useSmplx == True):
            raise Exception("Setting set_smplx_conditioning to be False and set_sequential_conditioning_suboption_useSmplx is True is Not properly implemented yet!") 


        if set_sequential_conditioning:
            config["model"]["params"]["unet_config"]["params"]["in_channels"] += 8 # need to add 2 more embedded images, so 2*4=8 channels
            print("No. of in-channels in U-Net modified due to use of sequential conditioning!")

            if set_sequential_conditioning_suboption_useSmplx:

                if set_smplx_conditioning_suboption_useBlendweights:
                    config["model"]["params"]["unet_config"]["params"]["in_channels"] += 3*(3*NUM_OF_BLENDWEIGHT_MAPS_TO_USE) # need to add 3 more embedded smplx blendweight images, so 3*(3*NUM_OF_BLENDWEIGHT_MAPS_TO_USE)= x channels
                else:
                    config["model"]["params"]["unet_config"]["params"]["in_channels"] += 9 # need to add 3 more downsampled smplx images, so 3*3=9 channels

                print("No. of in-channels in U-Net modified due to use of sequential smplx conditioning!")

                config["data"]["params"]["train"]["params"]["useSequentialSmplx"] = True
                config["data"]["params"]["validation"]["params"]["useSequentialSmplx"] = True


            config["data"]["params"]["train"]["target"] = "ldm.data.thuman.ThumanDatasetNovelTrain"

            if to_validate_the_images_only:
                pass # config["data"]["params"]["validation"]["target"] is already set!
            else:
                config["data"]["params"]["validation"]["target"] = "ldm.data.thuman.ThumanDatasetNovelValidation"
            print("Train and Validation Datasets changed to ThumanDatasetNovel due to use of sequential conditioning!")


            if set_sequential_conditioning_suboption_useCrossAttention:
                config["model"]["params"]["unet_config"]["params"]["context_dim"] += 1536 # need to add 2 more CLIP-embedded images, so 2*768=1536 channels
                print("No. of context_dim in U-Net modified due to use of cross attention in sequential conditioning!")



        
        trainer_config["accelerator"] = "ddp"
        
        for k in nondefault_trainer_args(opt):
            trainer_config[k] = getattr(opt, k)
        if not "gpus" in trainer_config:
            del trainer_config["accelerator"]
            cpu = True
        else:
            gpuinfo = trainer_config["gpus"]
            rank_zero_print(f"Running on GPUs {gpuinfo}")
            cpu = False
        trainer_opt = argparse.Namespace(**trainer_config)
        lightning_config.trainer = trainer_config


        # model
        model = instantiate_from_config(config.model)
        model.cpu()

        if not opt.finetune_from == "":

            if opt.resume:
                raise Exception("opt.resume and opt.finetune_from cannot be both specified!")

            rank_zero_print(f"Attempting to load state from {opt.finetune_from}")
            old_state = torch.load(opt.finetune_from, map_location="cpu")

            if "state_dict" in old_state:
                rank_zero_print(f"Found nested key 'state_dict' in checkpoint, loading this instead")
                old_state = old_state["state_dict"]

            new_state = model.state_dict()
            
            if "model.diffusion_model.input_blocks.0.0.weight" not in old_state: # if true, then must be a model saved by DeepSpeed 
                misnamed_keys = []
                for k,v in old_state.items():
                    if '_forward_module.' in k:
                        misnamed_keys.append(k)
                for k in misnamed_keys:
                    edited_k = k.replace('_forward_module.','')
                    old_state[edited_k] = old_state[k] 
                    old_state.pop(k)


            else:
                # Check if we need to port weights from 4ch input to 8ch
                in_filters_load = old_state["model.diffusion_model.input_blocks.0.0.weight"]
                in_filters_current = new_state["model.diffusion_model.input_blocks.0.0.weight"]
                in_shape = in_filters_current.shape
                if in_shape != in_filters_load.shape:
                    input_keys = [
                        "model.diffusion_model.input_blocks.0.0.weight",
                        "model_ema.diffusion_modelinput_blocks00weight",
                    ]
                    
                    for input_key in input_keys:
                        if input_key not in old_state or input_key not in new_state:
                            continue
                        input_weight = new_state[input_key]
                        if input_weight.size() != old_state[input_key].size():
                            print(f"Manual init: {input_key}")
                            input_weight.zero_()

                            num_of_in_channels_in_old_state = old_state[input_key].shape[1]
                            num_of_in_channels_in_new_state = input_weight.shape[1]
                            if num_of_in_channels_in_old_state <= num_of_in_channels_in_new_state: 
                                #input_weight[:, :4, :, :].copy_(old_state[input_key]) # own modifications: commented out
                                input_weight[:, :num_of_in_channels_in_old_state, :, :].copy_(old_state[input_key]) 
                            else:
                                old_input_weight = old_state[input_key]
                                input_weight[:, :, :, :].copy_(old_input_weight[:, :num_of_in_channels_in_new_state, :, :]) 


                            old_state[input_key] = torch.nn.parameter.Parameter(input_weight)





            # loading the trained autoencoder weights (imported from stable-diffusion)
            _load_path = "trained_trial_autoencoder_kl_32x32x4/latest.ckpt"

            _load_state = torch.load(_load_path, map_location="cpu")
            if "state_dict" in _load_state:
                _load_state = _load_state["state_dict"]

            for k,v in new_state.items():
                if 'first_stage_model' in k:
                    edited_k = k.replace('first_stage_model.' , '')
                    old_state[k] = _load_state[edited_k]



            # check if attention weight shapes need to be ported too:
            first_attention_weight_name = "model.diffusion_model.input_blocks.1.1.transformer_blocks.0.attn2.to_k.weight"
            if first_attention_weight_name in old_state and first_attention_weight_name in new_state:
                old_attention_weights = old_state[first_attention_weight_name]
                new_attention_weights = new_state[first_attention_weight_name]
            if old_attention_weights.shape != new_attention_weights.shape:
                
                old_state_keys = old_state.keys()
                new_state_keys = new_state.keys()
                for input_key in old_state_keys:
                    if input_key not in new_state_keys:
                        continue
                    input_weight = new_state[input_key]
                    if input_weight.size() != old_state[input_key].size():
                        print(f"Manual init: {input_key}")
                        input_weight.zero_()

                        num_of_in_channels_in_old_state = old_state[input_key].shape[1]
                        num_of_in_channels_in_new_state = input_weight.shape[1]
                        if num_of_in_channels_in_old_state <= num_of_in_channels_in_new_state: 
                            input_weight[:, :num_of_in_channels_in_old_state].copy_(old_state[input_key]) 
                        else:
                            old_input_weight = old_state[input_key]
                            input_weight[:, :].copy_(old_input_weight[:, :num_of_in_channels_in_new_state]) 

                        old_state[input_key] = torch.nn.parameter.Parameter(input_weight)



            m, u = model.load_state_dict(old_state, strict=False)

            if len(m) > 0:
                rank_zero_print("missing keys:")
                rank_zero_print(m)
            if len(u) > 0:
                rank_zero_print("unexpected keys:")
                rank_zero_print(u)

        # trainer and callbacks
        trainer_kwargs = dict()

        # default logger configs
        default_logger_cfgs = {
            "wandb": {
                "target": "pytorch_lightning.loggers.WandbLogger",
                "params": {
                    "name": nowname,
                    "save_dir": logdir,
                    "offline": opt.debug,
                    "id": nowname,
                }
            },
            "testtube": {
                "target": "pytorch_lightning.loggers.TestTubeLogger",
                "params": {
                    "name": "testtube",
                    "save_dir": logdir,
                }
            },
            "tensorboard": {
                "target": "pytorch_lightning.loggers.TensorBoardLogger",
                "params": {
                    "name": "testtube",
                    "save_dir": logdir,
                }
            },

        }

        default_logger_cfg = default_logger_cfgs["testtube"]

        if "logger" in lightning_config:
            logger_cfg = lightning_config.logger
        else:
            logger_cfg = OmegaConf.create()

        logger_cfg = OmegaConf.merge(default_logger_cfg, logger_cfg)

        trainer_kwargs["logger"] = instantiate_from_config(logger_cfg)

        # modelcheckpoint - use TrainResult/EvalResult(checkpoint_on=metric) to
        # specify which metric is used to determine best models
        default_modelckpt_cfg = {
            "target": "pytorch_lightning.callbacks.ModelCheckpoint",
            "params": {
                "dirpath": ckptdir,
                "filename": "{epoch:06}",
                "verbose": True,
                "save_last": True,
            }
        }
        if hasattr(model, "monitor"):
            rank_zero_print(f"Monitoring {model.monitor} as checkpoint metric.")
            default_modelckpt_cfg["params"]["monitor"] = model.monitor
            default_modelckpt_cfg["params"]["save_top_k"] = 3

        if "modelcheckpoint" in lightning_config:
            modelckpt_cfg = lightning_config.modelcheckpoint
        else:
            modelckpt_cfg =  OmegaConf.create()
        modelckpt_cfg = OmegaConf.merge(default_modelckpt_cfg, modelckpt_cfg)
        rank_zero_print(f"Merged modelckpt-cfg: \n{modelckpt_cfg}")
        if version.parse(pl.__version__) < version.parse('1.4.0'):
            trainer_kwargs["checkpoint_callback"] = instantiate_from_config(modelckpt_cfg)

        # add callback which sets up log directory
        default_callbacks_cfg = {
            "setup_callback": {
                "target": "main.SetupCallback",
                "params": {
                    "resume": opt.resume,
                    "now": now,
                    "logdir": logdir,
                    "ckptdir": ckptdir,
                    "cfgdir": cfgdir,
                    "config": config,
                    "lightning_config": lightning_config,
                    "debug": opt.debug,
                }
            },
            "image_logger": {
                "target": "main.ImageLogger",
                "params": {
                    "batch_frequency": 750,
                    "max_images": 4,
                    "clamp": True
                }
            },
            "learning_rate_logger": {
                "target": "main.LearningRateMonitor",
                "params": {
                    "logging_interval": "step",
                    # "log_momentum": True
                }
            },
            "cuda_callback": {
                "target": "main.CUDACallback"
            },
        }
        if version.parse(pl.__version__) >= version.parse('1.4.0'):
            default_callbacks_cfg.update({'checkpoint_callback': modelckpt_cfg})

        if "callbacks" in lightning_config:
            callbacks_cfg = lightning_config.callbacks
        else:
            callbacks_cfg = OmegaConf.create()


        callbacks_cfg = OmegaConf.merge(default_callbacks_cfg, callbacks_cfg)


        trainer_kwargs["callbacks"] = [instantiate_from_config(callbacks_cfg[k]) for k in callbacks_cfg]
        if not "plugins" in trainer_kwargs:
            trainer_kwargs["plugins"] = list()
        if not lightning_config.get("find_unused_parameters", True):
            from pytorch_lightning.plugins import DDPPlugin
            trainer_kwargs["plugins"].append(DDPPlugin(find_unused_parameters=False))





        trainer = Trainer.from_argparse_args(trainer_opt, **trainer_kwargs)
        
        trainer.logdir = logdir  ###

        # data
        data = instantiate_from_config(config.data)
        # NOTE according to https://pytorch-lightning.readthedocs.io/en/latest/datamodules.html
        # calling these ourselves should not be necessary but it is.
        # lightning still takes care of proper multiprocessing though
        data.prepare_data()
        data.setup()
        rank_zero_print("#### Data ####")
        try:
            for k in data.datasets:
                rank_zero_print(f"{k}, {data.datasets[k].__class__.__name__}, {len(data.datasets[k])}")
        except:
            rank_zero_print("datasets not yet initialized.")




        # configure learning rate
        bs, base_lr = config.data.params.batch_size, config.model.base_learning_rate
        if not cpu:
            ngpu = len(lightning_config.trainer.gpus.strip(",").split(','))
        else:
            ngpu = 1
        if 'accumulate_grad_batches' in lightning_config.trainer:
            accumulate_grad_batches = lightning_config.trainer.accumulate_grad_batches
        else:
            accumulate_grad_batches = 1
        rank_zero_print(f"accumulate_grad_batches = {accumulate_grad_batches}")
        lightning_config.trainer.accumulate_grad_batches = accumulate_grad_batches



        model.learning_rate = base_lr
        rank_zero_print("++++ NOT USING LR SCALING ++++") # legacy purposes
        rank_zero_print(f"Setting learning rate to {model.learning_rate:.2e}")



        # allow checkpointing via USR1
        def melk(*args, **kwargs):
                # run all checkpoint hooks
                if trainer.global_rank == 0:
                    rank_zero_print("Summoning checkpoint.")
                    ckpt_path = os.path.join(ckptdir, "last.ckpt")
                    trainer.save_checkpoint(ckpt_path)

        def divein(*args, **kwargs):
            if trainer.global_rank == 0:
                import pudb;
                pudb.set_trace()



        # run
        if opt.train:
            try:
                trainer.fit(model, data)
            except Exception:
            #    if not opt.debug:
            #        melk()
                print("skipping melk()..")
                raise
        if not opt.no_test and not trainer.interrupted:
            trainer.test(model, data)






    except Exception:
        if opt.debug and trainer.global_rank == 0:
            try:
                import pudb as debugger
            except ImportError:
                import pdb as debugger
            debugger.post_mortem()
        raise
    finally:
        # move newly created debug project to debug_runs
        if opt.debug and not opt.resume and trainer.global_rank == 0:
            dst, name = os.path.split(logdir)
            dst = os.path.join(dst, "debug_runs", name)
            os.makedirs(os.path.split(dst)[0], exist_ok=True)
            os.rename(logdir, dst)
        if trainer.global_rank == 0:
            rank_zero_print(trainer.profiler.summary())
