import os
import json
import time
import torch
import types
import inspect
import argparse
import datetime
import numpy as np
from pathlib import Path
import albumentations as albu
import timm.optim.optim_factory as optim_factory
from torch.utils.tensorboard import SummaryWriter
import dfformer.training_scripts.utils.misc as misc
#Register for generate function or CLASS from string
from dfformer.registry import MODELS, POSTFUNCS
from dfformer.datasets import ManiDataset, JsonDataset, BalancedDataset
from dfformer.transforms import RandomCopyMove, RandomInpainting
from dfformer.evaluation import PixelF1, ImageF1 # TODO You can select evaluator you like here
from dfformer.training_scripts.tester import test_one_epoch
from dfformer.training_scripts.trainer import train_one_epoch

def get_args_parser():
    parser = argparse.ArgumentParser('IMDLBenCo training launch!', add_help=True)

    # -------------------------------
    # Model name
    parser.add_argument('--model', default='DFFormer', type=str,
                        help='The name of applied model')#, required=True)
    # 可以接受label的模型是否接受label输入，并启用相关的loss。
    parser.add_argument('--if_predict_label', default=False, #action='store_true',
                        help='Does the model that can accept labels actually take label input and enable the corresponding loss function?')
    # ----Dataset parameters 数据集相关的参数----
    parser.add_argument('--image_size', default=256, type=int,
                        help='image size of the images in datasets')
    parser.add_argument('--if_padding', default=False, #action='store_true', #is_padding and is_resizing can not be True at the same time
                         help='padding all images to same resolution.')
    parser.add_argument('--if_resizing', default=True, #action='store_true',
                        help='resize all images to same resolution.')
    parser.add_argument('--edge_mask_width', default=0, type=int, #default=0,must be a odd
                        help='Edge broaden size (in pixels) for edge maks generator.')
    #parser.add_argument('--train_data_path', default='/root/Dataset/CASIA2.0/', type=str,
    parser.add_argument('--train_data_path', default='/home/eva/DFFormer/dfformer/datasets/lists/casiav2.txt', type=str, #/home/user/datasets/CASIA/CASIA2/
                        help='dataset path, should be our json_dataset or mani_dataset format. Details are in readme.md')
    #parser.add_argument('--test_data_path', default='/root/Dataset/CASIA1.0', type=str,
    parser.add_argument('--test_data_path', default='/home/eva/DFFormer/dfformer/datasets/lists/casiav1.txt', type=str,
                        help='test dataset path, should be our json_dataset or mani_dataset format. Details are in readme.md')
    # ------------------------------------
    # training related
    parser.add_argument('--batch_size', default=8, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--test_batch_size', default=8, type=int,
                        help="batch size for testing")
    parser.add_argument('--epochs', default=200, type=int)
    # Test related
    parser.add_argument('--no_model_eval', default=False, #action='store_true',
                       help='Do not use model.eval() during testing.')
    parser.add_argument('--test_period', default=1, type=int,
                        help="how many epoch per testing one time")
    
    # 一个epoch在tensorboard中打几个loss的data point
    parser.add_argument('--log_per_epoch_count', default=20, type=int,
                        help="how many loggings (data points for loss) per testing epoch in Tensorboard")
    
    parser.add_argument('--find_unused_parameters', default=False, #action='store_true',
                        help='find_unused_parameters for DDP. Mainly solve issue for model with image-level prediction but not activate during training.')
    
    # 不启用AMP（自动精度）进行训练
    parser.add_argument('--if_not_amp', default=True, # mvssnet set true only action='store_false',
                        help='Do not use automatic precision.')
    parser.add_argument('--accum_iter', default=16, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')

    # Optimizer parameters
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')
    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=2e-4, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--min_lr', type=float, default=5e-7, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')
    parser.add_argument('--warmup_epochs', type=int, default=4, metavar='N',
                        help='epochs to warmup LR')

    # ----输出的日志相关的参数----------
    # ----output related parameters----
    parser.add_argument('--output_dir', default='./weights_dir', #'./weights_dir',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default='./weights_dir',
                        help='path where to tensorboard log')
    # -----------------------
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default= '',#'./weights_dir/DFFormer/checkpoint-78.pth',#''
                        help='resume from checkpoint, input the path of a ckpt.')

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--num_workers', default=1, type=int)
    parser.add_argument('--pin_mem', default=True,#action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', default=False, #action='store_false',
                        dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--rank', type=int, default=0)
    parser.add_argument('--local_rank', default=0, type=int)
    parser.add_argument('--dist_on_itp', default=False)  #action='store_true')
    parser.add_argument('--dist_url', default='tcp://127.0.0.1:12345',#'env://',
                        help='url used to set up distributed training')


    args, remaining_args = parser.parse_known_args()
    # 获取对应的模型类
    # Get the corresponding model class
    model_class = MODELS.get(args.model)

    # 根据模型类动态创建参数解析器
    # Dynamically create a parameter parser based on the model class
    model_parser = misc.create_argparser(model_class)
    model_args = model_parser.parse_args(remaining_args)

    return args, model_args

def main(args, model_args):
    # init parameters for distributed training
    misc.init_distributed_mode(args)
    import torch.multiprocessing
    torch.multiprocessing.set_sharing_strategy('file_system')
    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("=====args:=====")
    print("{}".format(args).replace(', ', ',\n'))
    print("=====Model args:=====")
    print("{}".format(model_args).replace(', ', ',\n'))
    device = torch.device(args.device)
    
    # fix the seed for reproducibility
    seed = args.seed + misc.get_rank()
    misc.seed_torch(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    """=========================================================
    You Can Modify code below to customize your data augmentation TODO
    ========================================================="""
    train_transform = albu.Compose([
            # Rescale the input image by a random factor between 0.8 and 1.2
            albu.RandomScale(scale_limit=0.2, p=1), 
            RandomCopyMove(p = 0.1),
            RandomInpainting(p = 0.1),
            # Flips
            # albu.Resize(512, 512),
            albu.HorizontalFlip(p=0.5),
            albu.VerticalFlip(p=0.5),
            # Brightness and contrast fluctuation
            albu.RandomBrightnessContrast(
                brightness_limit=(-0.1, 0.1),
                contrast_limit=0.1,
                p=1
            ),
            albu.ImageCompression(
                quality_lower = 70,
                quality_upper = 100,
                p = 0.2
            ),
            # Rotate
            albu.RandomRotate90(p=0.5),
            # Blur
            albu.GaussianBlur(
                blur_limit = (3, 7),
                p = 0.2
            ),
    ])
    # TODO
    test_transform = albu.Compose([
        # ---Blow for robustness evalution---
        # albu.Resize(512, 512),
        #   albu.JpegCompression(
        #         quality_lower = 89,
        #         quality_upper = 90,
        #         p = 1
        #   ),
        #  albu.GaussianBlur(
        #         blur_limit = (5, 5),
        #         p = 1
        #     ),
        
        # albu.GaussNoise(
        #     var_limit=(15, 15),
        #     p = 1
        # )
    ])

    print("Train transform: ", train_transform)
    print("Test transform: ", test_transform)

    # get post function (if have)
    post_function_name = f"{args.model}_post_func".lower()
    print(f"Post function check: {post_function_name}")
    print(POSTFUNCS)
    if POSTFUNCS.has(post_function_name):
        post_function = POSTFUNCS.get(post_function_name)
    else:
        post_function = None
    # ---- dataset with crop augmentation ----
    #if os.path.isdir(args.train_data_path):
    if os.path.isfile(args.train_data_path): #added by xiang 2025.8.7
        dataset_train = ManiDataset(
            args.train_data_path,
            is_padding=args.if_padding,
            is_resizing=args.if_resizing,
            output_size=(args.image_size, args.image_size),
            common_transforms=train_transform,
            edge_width=args.edge_mask_width,
            post_funcs=post_function
        )
    else:
        try:
            dataset_train = JsonDataset(
                args.train_data_path,
                is_padding=args.if_padding,
                is_resizing=args.if_resizing,
                output_size=(args.image_size, args.image_size),
                common_transforms=train_transform,
                edge_width=args.edge_mask_width,
                post_funcs=post_function
            )
        except:
            dataset_train = BalancedDataset(
                args.train_data_path,
                is_padding=args.if_padding,
                is_resizing=args.if_resizing,
                output_size=(args.image_size, args.image_size),
                common_transforms=train_transform,
                edge_width=args.edge_mask_width,
                post_funcs=post_function
            )
    
    test_datasets = {}
    #if os.path.isdir(args.test_data_path):
    if os.path.isfile(args.test_data_path): #added by xiang 2025.8.7
        dataset_test = ManiDataset(
            args.test_data_path,
            is_padding=args.if_padding,
            is_resizing=args.if_resizing,
            output_size=(args.image_size, args.image_size),
            common_transforms=test_transform,
            edge_width=args.edge_mask_width,
            post_funcs=post_function
        )
        test_datasets['test_dataset'] = dataset_test
    else:
        with open(args.test_data_path, "r") as f:
            test_dataset_json = json.load(f)
        for dataset_name, dataset_path in test_dataset_json.items():
            #if os.path.isdir(dataset_path):
            if os.path.isfile(dataset_path):#added by xiang 2025.8.7
                dataset_test = ManiDataset(
                    dataset_path,
                    is_padding=args.if_padding,
                    is_resizing=args.if_resizing,
                    output_size=(args.image_size, args.image_size),
                    common_transforms=test_transform,
                    edge_width=args.edge_mask_width,
                    post_funcs=post_function
                )
            else:
                dataset_test = JsonDataset(
                    dataset_path,
                    is_padding=args.if_padding,
                    is_resizing=args.if_resizing,
                    output_size=(args.image_size, args.image_size),
                    common_transforms=test_transform,
                    edge_width=args.edge_mask_width,
                    post_funcs=post_function
                )
            test_datasets[dataset_name] = dataset_test
    # ------------------------------------
    print(dataset_train)
    print(test_datasets)
    test_sampler = {}
    global_rank = misc.get_rank()
    if args.distributed:
        num_tasks = misc.get_world_size()
        # global_rank = misc.get_rank()
        sampler_train = torch.utils.data.DistributedSampler(
            dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )
        for test_dataset_name, dataset_test in test_datasets.items():
            sampler_test = torch.utils.data.DistributedSampler(
                dataset_test, 
                num_replicas=num_tasks, 
                rank=global_rank, 
                shuffle=False,
                drop_last=True
            )
            test_sampler[test_dataset_name] = sampler_test
        print("Sampler_train = %s" % str(sampler_train))
        print("Sampler_test = %s" % str(sampler_test))
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        for test_dataset_name,dataset_test in test_datasets.items():
            sampler_test = torch.utils.data.RandomSampler(dataset_test)
            test_sampler[test_dataset_name] = sampler_test

    if global_rank == 0 and args.log_dir is not None:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=args.log_dir)
    else:
        log_writer = None

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )
    
    test_dataloaders = {}
    for test_dataset_name in test_sampler.keys():
        test_dataloader = torch.utils.data.DataLoader(
            test_datasets[test_dataset_name], sampler=test_sampler[test_dataset_name],
            batch_size=args.test_batch_size,
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            drop_last=True,
        )
        test_dataloaders[test_dataset_name] = test_dataloader
    
    # ========define the model directly==========
    # model = IML_ViT(
    #     vit_pretrain_path = model_args.vit_pretrain_path,
    #     predict_head_norm= model_args.predict_head_norm,
    #     edge_lambda = model_args.edge_lambda
    # )
    
    # --------------- or -------------------------
    # Init model with registry
    model = MODELS.get(args.model)
    #torch.save(model, 'objectformer-model.pth')
    # Filt usefull args
    if isinstance(model,(types.FunctionType, types.MethodType)):
        model_init_params = inspect.signature(model).parameters
    else:
        model_init_params = inspect.signature(model.__init__).parameters
    combined_args = {k: v for k, v in vars(args).items() if k in model_init_params}
    for k, v in vars(model_args).items():
        if k in model_init_params and k not in combined_args:
            combined_args[k] = v
            if args.model == 'DFFormer' and k == 'model_name':
                combined_args[k] = 'dfformer-large'
            elif args.model == 'DFFormer' and k == 'img_size':
                combined_args[k] = args.image_size
            elif args.model=='PSCC_Net' and k == 'pretrain_path':
                combined_args[k]= './pretrained/PSCC_Net/HRNet.pth'#pscc-net only
            elif args.model=='Cat_Net' and k == 'cfg_file':
                combined_args[k] = './dfformer/model_zoo/cat_net/CAT_full.yaml'
            elif args.model=='IML_ViT' and k == 'vit_pretrain_path':
                combined_args[k] = './pretrained/IML_ViT/pretrain_iml-vit.pth'
            elif args.model == 'Trufor' and k == 'config_path':
                combined_args[k] = './dfformer/model_zoo/trufor/trufor.yaml'
            elif args.model == 'Trufor' and k == 'phase':
                combined_args[k] = 2
            elif args.model == 'DFPF_Net' and k == 'vit_name':
                combined_args[k] = 'R50-ViT-B_16'
            elif args.model == 'MantraNet' and k == 'weights_path':
                combined_args = './pretrained/MantrNet/weights/IMTFEv4.pth'

    model = model(**combined_args)
    # ============================================

    """
    TODO Set the evaluator you want to use
    You can use PixelF1, ImageF1, or any other evaluator you like.
    """    
    evaluator_list = [
        PixelF1(threshold=0.5, mode="origin"),
        # ImageF1(threshold=0.5)
    ]
    
    if args.distributed:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    model.to(device)
    # for name, param in model.named_parameters():
    #     if not param.requires_grad:
    #         print('1:',name)
    model_without_ddp = model
    #model = misc.ddp_exclude_frozen(model)
    # for name, param in model.named_parameters():
    #     if not param.requires_grad:
    #         print('2:',name)
    print("Model = %s" % str(model_without_ddp))

    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()
    
    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * eff_batch_size / 256

    print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
    print("actual lr: %.2e" % args.lr)

    print("accumulate grad iterations: %d" % args.accum_iter)
    print("effective batch size: %d" % eff_batch_size)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=args.find_unused_parameters)
        model_without_ddp = model.module
        print('distributed model init done')
    
    # TODO You can set optimizer settings here
    args.opt='AdamW'
    args.betas=(0.9, 0.999)
    args.momentum=0.9
    optimizer = optim_factory.create_optimizer(args, model_without_ddp)
    print(optimizer)
    loss_scaler = misc.NativeScalerWithGradNormCount()

    misc.load_model(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    best_evaluate_metric_value = 0
    for epoch in range(args.start_epoch, args.epochs):
        args.current_epoch = epoch
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)
        # train for one epoch
        train_stats = train_one_epoch(
            model, data_loader_train,
            optimizer, device, epoch, loss_scaler,
            log_writer=log_writer,
            log_per_epoch_count=args.log_per_epoch_count,
            args=args
        )
        
        # # saving checkpoint
        if args.output_dir and (epoch % 1 == 0 or epoch + 1 == args.epochs):
            misc.save_model(
                args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                loss_scaler=loss_scaler, epoch=epoch)

        optimizer.zero_grad()
        # test for args.test_period if needed
        # if epoch % args.test_period == 0 or epoch + 1 == args.epochs:
        #     values = {} # dict of dict (dataset_name: {metric_name: metric_value})
        #     # test across all datasets in the `test_data_loaders' dict
        #     for test_dataset_name, test_dataloader in test_dataloaders.items():
        #         print(f'!!!Start Test: {test_dataset_name}',len(test_dataloader))
        #         test_stats = test_one_epoch(
        #             model,
        #             data_loader = test_dataloader,
        #             evaluator_list=evaluator_list,
        #             device = device,
        #             epoch = epoch,
        #             name = test_dataset_name,
        #             log_writer=log_writer,
        #             args = args,
        #             is_test = False,
        #         )
        #         one_metric_value = {}
        #         # Read the metric value from the test_stats dict
        #         for evaluator in evaluator_list:
        #             evaluate_metric_value = test_stats[evaluator.name]
        #             one_metric_value[evaluator.name] = evaluate_metric_value
        #         values[test_dataset_name] = one_metric_value
        #
        #     metrics_dict = {metric: {dataset: values[dataset][metric] for dataset in values} for metric in {m for d in values.values() for m in d}}
        #     # Calculate the mean of each metric across all datasets
        #     metric_means = {metric: np.mean(list(datasets.values())) for metric, datasets in metrics_dict.items()}
        #     # Calculate the mean of all metrics
        #     evaluate_metric_value = np.mean(list(metric_means.values()))
        #
        #     # Store the best metric value
        #     if evaluate_metric_value > best_evaluate_metric_value :
        #         best_evaluate_metric_value = evaluate_metric_value
        #         print(f"Best {' '.join([evaluator.name for evaluator in evaluator_list])} = {best_evaluate_metric_value}")
        #         # Save the best only after 20 epoch. TODO you can modify this.
        #         if epoch > 20:
        #             misc.save_model(
        #         args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
        #         loss_scaler=loss_scaler, epoch=epoch)
        #     else:
        #         print(f"Average {' '.join([evaluator.name for evaluator in evaluator_list])} = {evaluate_metric_value}")
            # Log the metrics to Tensorboard
        #     if log_writer is not None:
        #         for metric, datasets in metrics_dict.items():
        #             log_writer.add_scalars(f'{metric}_Metric', datasets, epoch)
        #         log_writer.add_scalar('Average', evaluate_metric_value, epoch)
        #     log_stats =  {**{f'train_{k}': v for k, v in train_stats.items()},
        #                 **{f'test_{k}': v for k, v in test_stats.items()},
        #                     'epoch': epoch,}
        # else:
        #    log_stats = {**{f'train_{k}': v for k, v in train_stats.items()}, 'epoch': epoch,}
        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()}, 'epoch': epoch, }
        
        if args.output_dir and misc.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with open(os.path.join(args.output_dir, "train_log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    print('Training time {}'.format(total_time))
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    args, model_args = get_args_parser()
    args.output_dir = os.path.join(args.output_dir, args.model)
    args.log_dir = os.path.join(args.log_dir, args.model)
    args.num_gpu = torch.cuda.device_count()
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2"
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    if 'WORLD_SIZE' not in os.environ:
        os.environ['WORLD_SIZE'] = str(args.world_size)
    if 'RANK' not in os.environ:
        os.environ['RANK'] = str(args.rank)
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args, model_args)
