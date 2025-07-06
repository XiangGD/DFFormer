import argparse
import os
import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from trainer import trainer_forensic
from networks.tcformer import TCFormer
from networks.tcformer import CONFIGS as CONFIGS_Graph

parser = argparse.ArgumentParser()
# dataset parameters
parser.add_argument('--root_path', type=str, default='/share/home/xiangyan/datasets', help='root dir for data')
parser.add_argument('--list_dir', type=str, default='/share/home/xiangyan/GraphFormer/lists', help='list dir')
parser.add_argument('--num_classes', type=int, default=1, help='output channel of network')
parser.add_argument('--tr_data', type=str, default='CASIA2',#'PS',
                    # default=['CASIA1','CASIA2','Coverage','NIST2016','Columbia','DEFACTO'],\
                    help='training dataset of network')
parser.add_argument('--val_data', type=str, default='Coverage', help='valuation dataset of network')
parser.add_argument('--test_data', type=str, default='Coverage', help='testing dataset of network')
# model parameters
parser.add_argument('--model_name', type=str, default='tc-large', help='select one vit model')
parser.add_argument('--tr_epochs', type=int, default=120, help='maximum epoch number to train')
parser.add_argument('--tr_batch', type=int, default=32, help='batch_size per gpu')
parser.add_argument('--tr_lr', type=float, default=0.00005, help='segmentation network learning rate')
parser.add_argument('--num_gpu', type=int, default=0, help='total gpu number, if only cup, num_gpu equal 0')
parser.add_argument('--local_rank', type=int, default=0)
parser.add_argument('--seed', type=int, default=1234, help='random seed')
parser.add_argument('--img_size', type=int, default=512, help='input patch size of network input')
parser.add_argument('--optimizer', type=str, default='AdamW', help='optimizier type，SGD or Adam')
parser.add_argument('--save_interval', type=int, default=5, help='frequency of perform checkpoint')
# fine_tuned parameters
parser.add_argument('--split', type=str, default='train', help='split train model or fine tune model')
parser.add_argument('--freeze', type=bool, default=False, help='Froze some layers to fine-tuned the model')
parser.add_argument('--ft_data', type=list, default=['CASIA2', 'Coverage', 'NIST2016'],#, 'Columbia', 'IMD2020'],
                    help='fine tune dataset of network')
parser.add_argument('--ft_epochs', type=int, default=200, help='max epochs for fine-tuned the model')
parser.add_argument('--ft_batch', type=int, default=4, help='batch size for fine-tuned the model')
parser.add_argument('--ft_lr', type=float, default=0.00005, help='batch size for fine-tuned the model')
parser.add_argument('--train_best_model', type=int, default=53, help='best pre_train weights')
parser.add_argument('--ft_load_path', type=str, default='', help='fine tuned init weights load path')
# checkpiont parameters
parser.add_argument('--resume', type=bool, default=False, help='resume from checkpoint')
parser.add_argument('--continue_model', type=int, default=5, help='resume from checkpoint')
parser.add_argument('--ckpt_path', type=str, default='', help='resume path')
parser.add_argument('--aux_path', type=str, default='_crpe_res_fuse1_4gpu_115k', help='aux path')
# evaluation parameters
parser.add_argument('--premask_save_path', type=str, default=None, help='save path of precision masks of evaluation')

args = parser.parse_args()

if 'LOCAL_RANK' not in os.environ:
    os.environ['LOCAL_RANK'] = str(args.local_rank)

#os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"
args.local_rank = os.environ['LOCAL_RANK']

if __name__ == "__main__":

    if torch.cuda.is_available():
        args.device = 'cuda'
        cudnn.benchmark = False
        cudnn.deterministic = True
        args.num_gpu = torch.cuda.device_count()
        #distribute initial settings
        torch.distributed.init_process_group(backend="nccl")
        local_rank = torch.distributed.get_rank()
        print('local_rank:',local_rank)
        torch.cuda.set_device(local_rank)
        args.device = torch.device("cuda", local_rank)
    else:
        args.device = 'cpu'

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    # dataset_name = args.train_data
    dataset_name =  args.tr_data
    args.is_pretrain = True
    args.exp = 'FU_' + dataset_name

    snapshot_path = "./models/{}/{}".format(args.exp,  'Graph')
    snapshot_path = snapshot_path + '_' + args.split + '_' + args.model_name + args.aux_path
    snapshot_path = snapshot_path + '_epo' + str(args.tr_epochs) if args.split == 'train' else snapshot_path + '_epo' + str(args.ft_epochs)
    snapshot_path = snapshot_path + '_bs' + str(args.tr_batch) if args.split == 'train' else snapshot_path + '_bs' + str(args.ft_batch)
    snapshot_path = snapshot_path + '_' + str(args.optimizer)
    snapshot_path = snapshot_path + '_lr' + str(args.tr_lr) if args.split == 'train' else snapshot_path + '_lr' + str(args.ft_lr)
    snapshot_path = snapshot_path + '_' + str(args.img_size)
    snapshot_path = snapshot_path + '_s' + str(args.seed) if args.seed != 1234 else snapshot_path
    print('snapshot_path:', snapshot_path)

    config_graph = CONFIGS_Graph[args.model_name]
    config_graph.num_classes = args.num_classes

    net = TCFormer(config_graph, img_size=args.img_size).to(args.device)

    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)

    assert args.split == 'train' or args.split == 'fine_tuned', 'split not train or ft !'
    if args.split == 'fine_tuned':
        ft_load_path = snapshot_path.replace( '_epo'+str(args.ft_epochs), '_epo'+str(args.tr_epochs))
        ft_load_path = ft_load_path.replace( '_bs'+str(args.ft_batch), '_bs'+str(args.tr_batch))
        ft_load_path = ft_load_path.replace( '_lr'+str(args.ft_lr), '_lr'+str(args.tr_lr))
        ft_load_path = ft_load_path.replace('_' + args.split, '_train') 
           
        args.ft_load_path = os.path.join(ft_load_path, 'epo{}.pth'.format(args.train_best_model))
        print(args.ft_load_path)
        assert os.access(args.ft_load_path, os.F_OK), 'fine tuned load path not exits!'
    

    if args.resume:
        args.ckpt_path = os.path.join(snapshot_path, 'epo{}.pth'.format(args.continue_model))
        assert os.access(args.ckpt_path, os.F_OK), 'resume path not exits!'

    trainer = {'PS': trainer_forensic,
               'DEFACTO': trainer_forensic,
               'CoCo2014': trainer_forensic,
               'Synth_COCO': trainer_forensic,
               'CASIA2': trainer_forensic,
               'IML-MUST': trainer_forensic
               }
    trainer[dataset_name](args, net, snapshot_path)
