import argparse
import logging
import os
import pickle
import random
import sys
import numpy as np
import collections
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.modules.utils import _pair
from tqdm import tqdm
from sklearn.metrics import roc_curve, roc_auc_score, auc
from datasets.dataset_forensic import Forensic_dataset
from utils import test_single_sample, metric_img_score
from trainer import trainer_forensic
from networks.tcformer import TCFormer
from networks.tcformer import CONFIGS as CONFIGS_Graph

parser = argparse.ArgumentParser()
#datiaset parameters
parser.add_argument('--root_path',type=str, default='/share/home/xiangyan/datasets', help='root dir for data')
parser.add_argument('--dataset',  type=str, default='PS', help='experiment_name')
parser.add_argument('--test_datas',  type=str, default=['Coverage','CASIA1','NIST2016'], help='experiment_name') #'Columbia','Coverage',
parser.add_argument('--num_classes', type=int, default=1, help='output channel of network')
parser.add_argument('--list_dir', type=str, default='/share/home/xiangyan/GraphFormer/lists', help='list dir') 
#model parameters
parser.add_argument('--model_name', type=str,default='dfformer-large', help='select one vit model')
parser.add_argument('--tr_epochs', type=int,default=100, help='maximum epoch number to train')
parser.add_argument('--ft_epochs', type=int,default=200, help='maximum epoch number to train')
parser.add_argument('--batch_tr', type=int,default=32, help='batch_size per gpu')
parser.add_argument('--batch_ft', type=int,default=8, help='batch_size per gpu')
parser.add_argument('--seed', type=int,default=1234, help='random seed')
parser.add_argument('--img_size', type=int,default=256, help='input patch size of network input')
parser.add_argument('--optimizer', type=str,default='AdamW', help='optimizier type, SGD or Adam')
parser.add_argument('--tr_lr', type=float,  default=0.0002,help='segmentation network learning rate')
parser.add_argument('--ft_lr', type=float,  default=0.00005,help='segmentation network learning rate')
parser.add_argument('--split', type=str,default='train',help='split train model or fine tune model')
parser.add_argument('--best_model', type=int, default=39,help='split train model or fine tune model')
parser.add_argument('--aux_path', type=str, default='_crpe_res_fuse1_2gpu_115k')
parser.add_argument('--start_epochs', type=int,default=1, help='maximum weight number ')
parser.add_argument('--end_epochs', type=int,default=100, help='maximum weight number ')
parser.add_argument('--gpu_idx', type=str,default='1', help='test gpu number ')

# test result save parameters
parser.add_argument('--is_save', default=False, action="store_true", help='whether to save results during inference')

args = parser.parse_args()

if 'LOCAL_RANK' not in os.environ:
    os.environ['LOCAL_RANK'] = str(args.local_rank)

#os.environ["CUDA_VISIBLE_DEVICES"] = "3"
args.local_rank = os.environ['LOCAL_RANK']


def loadweights(weights_path, map_location):
    new_dicts = collections.OrderedDict()
    weights = torch.load(weights_path, map_location=map_location)['model']
    keys = list(weights.keys())
    for key in keys:
        new_key = key.replace('module.','')
        new_dicts[new_key] = weights[key]
    return new_dicts
    
def inference(args, model, evaluate):
    if  evaluate == 'val':
        args.log_folder = os.path.join('./val/val_results', args.exp, snapshot_name, args.val_data)
        os.makedirs(args.log_folder, exist_ok=True)        
    
        
    db_test = Forensic_dataset(args, base_dir=args.root_path, list_dir=args.list_dir, split=args.split,
                               data_type=evaluate, img_size=_pair(args.img_size))
    
    #distributed setting
    if args.num_gpu > 1:       
        sampler = DistributedSampler(db_test, shuffle=False)  # 这个sampler会自动分配数据到各个gpu上
        testloader = DataLoader(db_test, batch_size=1, sampler=sampler)
    else:      
        testloader = DataLoader(db_test, batch_size=1, shuffle=False, num_workers=0)
    model.eval()
    
    #logging.info(args.exp+'_'+snapshot_name)    
    #logging.info("{} evaluation iterations per epoch".format(len(testloader)))

    classes = []
    scores = []
    mean_f1, mean_iou, mean_auc, mean_precision, mean_recall = 0, 0, 0, 0, 0
    num_samples = len(testloader)    
    
    for i_batch, sampled_batch in tqdm(enumerate(testloader)):
        _, _, h, w = sampled_batch["image"].size()
        assert (h,w)==_pair(args.img_size)
        image, label = sampled_batch["image"], sampled_batch["label"]
        cls, sample_name = sampled_batch['cls'][0], sampled_batch['name'][0]

        image, label = image.to(args.device), label.to(args.device)
        f1, iou, precision, recall, pixel_auc, max_score = test_single_sample(image, label, model,
                                                           num_class=args.num_classes,
                                                           test_save_path=args.premask_path,
                                                           case=sample_name,
                                                           is_save = args.is_save
                                                           )
            
        #print(max_score)
        mean_f1 += f1
        mean_iou += iou
        mean_precision += precision
        mean_recall += recall
        mean_auc += pixel_auc
        scores.append(max_score)
        classes.append(int(cls))

        # if evaluate == 'test':
        #     logging.info('%s-> single_f1: %f, single_iou: %f, single_p: %f, single_r: %f, single_auc: %f, single_score: %f' \
        #                  % (sample_name, f1, iou, precision, recall, pixel_auc, max_score)) 
   
    mean_f1        = mean_f1 / num_samples
    mean_iou       = mean_iou / num_samples
    mean_precision = mean_precision / num_samples
    mean_recall    = mean_recall / num_samples
    mean_auc       = mean_auc / num_samples
   
    return mean_f1,  mean_iou, mean_precision, mean_recall, mean_auc, scores, classes


if __name__ == "__main__":
  
    if torch.cuda.is_available():
        args.device = 'cuda'
        cudnn.benchmark = False
        cudnn.deterministic = True
        args.num_gpu = torch.cuda.device_count()
    else:
        args.device = 'cpu'
    #distribute initial settings
    if args.num_gpu > 1:
        torch.distributed.init_process_group(backend="nccl")
        local_rank = torch.distributed.get_rank()
        print('local_rank:',local_rank)
        torch.cuda.set_device(local_rank)
        args.device = torch.device("cuda", local_rank)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    
    # name the same snapshot defined in train script!
    args.exp = 'FU_' + args.dataset
    
    snapshot_path = "./models/{}/{}".format(args.exp,  'Graph')
    snapshot_path = snapshot_path + '_' + args.split + '_' + args.model_name + args.aux_path # '_crpe_res_fuse_2gpu_115k'
    
    if args.split == 'train':
        snapshot_path = snapshot_path + '_epo' + str(args.tr_epochs)
        snapshot_path = snapshot_path + '_bs' + str(args.batch_tr)
    elif args.split == 'fine_tuned':
        snapshot_path = snapshot_path + '_epo' + str(args.ft_epochs)
        snapshot_path = snapshot_path + '_bs' + str(args.batch_ft)
    else:
        snapshot_path
    snapshot_path = snapshot_path + '_' + str(args.optimizer)
    if args.split == 'train':
        snapshot_path = snapshot_path + '_lr' + str(args.tr_lr)
    elif args.split == 'fine_tuned':
        snapshot_path = snapshot_path + '_lr' + str(args.ft_lr)
    else:
        snapshot_path
    snapshot_path = snapshot_path + '_' + str(args.img_size)
    snapshot_path = snapshot_path + '_s' + str(args.seed) if args.seed != 1234 else snapshot_path
    snapshot_name = snapshot_path.split('/')[-1]
   
    config_graph = CONFIGS_Graph[args.model_name]
    config_graph.num_classes = args.num_classes

    net = TCFormer(config_graph, img_size=args.img_size).to(args.device)
    


    for data in args.test_datas:
        args.test_data = data
        args.log_folder = os.path.join('./test/test_results', args.exp, snapshot_name) 
        os.makedirs(args.log_folder, exist_ok=True)  
        logging.basicConfig(filename=args.log_folder+'/gpu'+args.gpu_idx+'_{}.txt'.format(args.test_data), level=logging.INFO)#, 
                            #format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')     
        logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    
        #logging.info(str(args))
        args.premask_path = args.log_folder.replace('test_results','visualization')
        if args.is_save:        
            args.premask_path = os.path.join(args.premask_path, 'epo{}_premask'.format(args.best_model))
            os.makedirs(args.premask_path, exist_ok=True)  
        
        for epo in range(args.start_epochs, args.end_epochs):
            snapshot = os.path.join(snapshot_path, 'epo{}.pth'.format(epo))
            assert os.path.exists(snapshot), 'test model weight donot exits.'
            #logging.info('test_model_load_path:',snapshot)

            #distributed setting
            if args.num_gpu > 1:
                net = torch.nn.parallel.DistributedDataParallel(net,find_unused_parameters=False)
                net.load_state_dict(torch.load(snapshot)['model']) 
            else:
                net.to(args.device)
                models = loadweights(snapshot, map_location=args.device)
                net.load_state_dict(models)
                #logging.info('test_model_load_path:',snapshot)
            net.eval()
    
        
            #logging.info('load pretrained mode {} sucessfully.'.format(snapshot))

            
            mean_f1,  mean_iou, mean_precision, mean_recall, mean_auc, scores, classes = inference(args, model=net, evaluate='test')  
            logging.info("data：%s, epoch: %d, pixel-mean_f1: %.4f,pixel-mean_iou: %.4f, pixel-mean_precision: %.4f, pixel-mean_recall: %.4f pixel-mean_auc: %.4f" \
                        % (data, epo, mean_f1, mean_iou, mean_precision, mean_recall, mean_auc))


