import logging
import os
import random
import sys
import torch
import torch.nn as nn
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from torch.nn.modules.utils import _pair
from tqdm import tqdm
from utils import DiceLoss
from datasets.dataset_forensic import Forensic_dataset
from torch import optim
from torch.utils.data.distributed import DistributedSampler



def trainer_forensic(args, model, snapshot_path):
    max_epoch = args.ft_epochs if args.split == 'fine_tuned' else args.tr_epochs
    logging.basicConfig(filename=snapshot_path + "/log_epo{}.txt".format(max_epoch), level=logging.INFO,
                        format='%(asctime)s.%(msecs)s 03d %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)
    
    device = args.device
    base_lr = args.tr_lr if args.split == 'train' else args.ft_lr
    batch_size = args.tr_batch * args.num_gpu if args.split == 'train' else args.ft_batch * args.num_gpu

    db_train = Forensic_dataset(args,
                                base_dir=args.root_path,
                                list_dir=args.list_dir,
                                split=args.split,
                                data_type=args.split,
                                img_size=_pair(args.img_size))
    logging.info("The number of train set is: {}".format(len(db_train)))

    #distributed setting
    if args.num_gpu > 1:
        model = torch.nn.parallel.DistributedDataParallel(model, find_unused_parameters=False)
        sampler = DistributedSampler(db_train, shuffle=True)  # 这个sampler会自动分配数据到各个gpu上
        trainloader = DataLoader(db_train, batch_size=batch_size, sampler=sampler)
    else:
        model.to(device)
        trainloader = DataLoader(db_train,
                                batch_size=batch_size,
                                shuffle=True,
                                num_workers=8,
                                pin_memory=True,
                                worker_init_fn=worker_init_fn)

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info('model params size {}:'.format(n_parameters))
    model.train()
           
    bce_loss = nn.BCEWithLogitsLoss()
    dice_loss = DiceLoss()
    optimizers = {'SGD': optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.01),
		'Adam': optim.Adam(model.parameters(), lr=base_lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.01),
		'AdamW': optim.AdamW(model.parameters(),lr=base_lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.01)
                 }
    optimizer = optimizers[args.optimizer]

    #scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,factor=0.1, patience=3, verbose=True)
    #scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.5, last_epoch=-1, verbose=True)
    #scheduler =  optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)

    writer = SummaryWriter(snapshot_path + '/log')
    iter_num = 0
    start_epoch = 0

    max_iterations = max_epoch * len(trainloader)  
    logging.info("{} iterations per epoch. {} max iterations ".format(len(trainloader), max_iterations))
    
    if args.freeze:
        for name, param in model.named_parameters():  # 带有参数名的模型的各个层包含的参数遍历
            if 'out' or 'merge' or 'before_regress' in name:  # 判断参数名字符串中是否包含某些关键字
                continue
            param.requires_grad = False   #except out&merge&before_regress layer

    if args.resume:
        model.load_state_dict(torch.load(args.ckpt_path)['model'])
        optimizer.load_state_dict(torch.load(args.ckpt_path)['optimizer'])
        start_epoch = torch.load(args.ckpt_path)['epoch_num']+1
        iter_num = torch.load(args.ckpt_path)['iter_num']
        print(start_epoch,iter_num)
        logging.info('resume succussfully! load from epo{}!'.format(args.continue_model))

    if args.split=='fine_tuned' and not args.resume:
        model.load_state_dict(torch.load(args.ft_load_path)['model'])
        logging.info('start fine_tuned! load from pre_train epo{} weights !'.format(args.train_best_model))

    epochs = tqdm(range(start_epoch, max_epoch), ncols=max_epoch)
    checkpoint = {'model': model.state_dict(),
                  'optimizer': optimizer.state_dict(),
                  'max_iterations': max_iterations
                  }
    
    torch.cuda.empty_cache() 
    epo_iter = len(trainloader)
    for epoch_num in epochs:
        epo_loss, epo_bceloss, epo_diceloss = 0.0, 0.0, 0.0
        for i_batch, sampled_batch in enumerate(trainloader):
            image_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            sample_name, cls = sampled_batch['name'], sampled_batch['cls']
            image_batch, label_batch = image_batch.to(device), label_batch.to(device)
            #print('train_image:',image_batch.shape,image_batch.min(),image_batch.max())
            #print('train_label:',label_batch.shape,label_batch.min(),label_batch.max())
            
            output = model(image_batch)
            #print('output shape:',output.shape,output.min(),output.max())
            logits = torch.squeeze(output)
            label_batch = torch.squeeze(label_batch)


            loss_bce = bce_loss(logits, label_batch)
            loss_dice = dice_loss(logits, label_batch)
            loss = 0.5 * loss_bce + 0.5 * loss_dice

            if loss.item() > 2e5:  # try to rescue the gradient explosion
                logging.info(" img{}\nLoss is abnormal, drop this batch !".format(sample_name))
                continue

            checkpoint['train_loss'] = loss
            epo_loss += loss 
            epo_bceloss += loss_bce
            epo_diceloss += loss_dice
            
            optimizer.zero_grad()    
            loss.backward()     
            optimizer.step() 
            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9

            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_
                
            iter_num += 1
            checkpoint['iter_num'] = iter_num
            
            writer.add_scalar('info/lr', lr_, iter_num)
            #writer.add_scalar('info/optimizer', optimizer, iter_num)
            writer.add_scalar('info/total_loss', loss, iter_num)
            writer.add_scalar('info/loss_bce', loss_bce, iter_num)
            writer.add_scalar('info/loss_dice', loss_dice, iter_num)
            #logging.info('iteration %d==> lr: %f,  loss : %f, loss_bce: %f, loss_dice: %f' \
            # % (iter_num, optimizer.param_groups[0]['lr'], loss.item(), loss_bce.item(),loss_dice.item()))
            
            # if iter_num % 20 == 0:
            #     image = image_batch[1, 0:1, :, :]
            #     image = (image - image.min()) / (image.max() - image.min())
            #     writer.add_image('train/Image', image, iter_num)
            #     output = torch.argmax(torch.sigmoid(output), dim=1, keepdim=True)
            #     writer.add_image('train/Prediction', output[1, ...] * 50, iter_num)
            #     labs = label_batch[1, ...].unsqueeze(0) * 50
            #     writer.add_image('train/GroundTruth', labs, iter_num)       

        logging.info('epoch %d==> lr: %f, mean_loss : %f, mean_diceloss: %f, mean_bceloss: %f'\
                     % (epoch_num, optimizer.param_groups[0]['lr'], epo_loss/epo_iter, epo_diceloss/epo_iter, epo_bceloss/epo_iter))
        #save checkpoint
        checkpoint['epoch_num'] = epoch_num
        save_mode_path = os.path.join(snapshot_path, 'epo' + str(epoch_num+1) + '.pth')
        torch.save(checkpoint, save_mode_path)
        logging.info("save model to {}".format(save_mode_path))

        """
        #evaluate model per 10 epochs
        if (epoch_num + 1) % args.save_interval == 0:           
            inference(args, model=model, evaluate='val')
            scheduler.step()
        """

    writer.close()
    return "Training Finished!"
