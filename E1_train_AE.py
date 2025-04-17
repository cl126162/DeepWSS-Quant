import argparse
import sys
import os
from datetime import datetime
from datetime import timedelta
import torch
from torch.utils.tensorboard import SummaryWriter
import torch.distributed as dist
from torch.utils.data import DataLoader
import h5py
import glob
from tqdm import tqdm
import json
import matplotlib.pyplot as plt

from E1_b_AE_utils import *

# ------------------------------------------------------------------------------------------
# ----------------------------------- define arguments    ----------------------------------
# ------------------------------------------------------------------------------------------

parser = argparse.ArgumentParser()
parser.add_argument('--trainData_dir', type=str, help='path and filename of dataset used for training')
parser.add_argument('--validData_dir', type=str, help='path and filename of dataset used for validation')
parser.add_argument('--LOGPATH', type=str, help='path to store logs')
parser.add_argument('--recover', type=str, default="no", help='yes if loading a checkpoint, no if training from scratch')
parser.add_argument('--recover_ckpt', type =str, default = "N/A", help ='full checkpoint path if recover=yes')
parser.add_argument('--RESNET', type=str, default="no", help='"yes" or "no" if using residual blocks')
parser.add_argument('--INPUT_IMAGE_HEIGHT', type=int, default=128)
parser.add_argument('--INPUT_IMAGE_WIDTH', type=int, default=128)
parser.add_argument('--channel1',type=int, default=128, help='feature maps in the first block; consecutive feature maps are multiples of that')
parser.add_argument('--layers_enc', nargs='+', type=int, default=[1,2,2,1], help='number of basic blocks in each encoding layer; 1 = 2 basic blocks, 2 = 4 basic blocks; first entry is ignored since this is related to stem')
parser.add_argument('--layers_dec', nargs='+', type=int, default=[2,2,2], help='number of basic blocks in each decoding layer; 1 = 2 basic blocks, 2 = 4 basic blocks') 
parser.add_argument('--SCALE_DECODER', type=float, default=2, help='factor by which the decoding steps make the upconv via interpolation')
parser.add_argument('--drop_ratio',type=float, default=0.3, help='dropout ratio')
parser.add_argument('--EPOCHS', default=500, type=int, help='number of total epochs')
parser.add_argument('--BATCH_SIZE', default=7, type=int, help='number of samples in a batch') 
parser.add_argument('--lr', type=float, default=0.000097, help='learning rate')
parser.add_argument('--wd',type=float, default = 0.00017, help='weight decay in optimizer')
parser.add_argument('--min_lr', type=float, default=1e-10, help='minimum learning rate for scheduler')
parser.add_argument('--reduce_factor', type=float, default=0.2, help='reduction factor in scheduler')
parser.add_argument('--patience_level', type=int, default=10, help='number of non-improvement steps in scheduler')
parser.add_argument('--DATATYPE',type=str, default="DNS", help='prefix in filename for saving')
parser.add_argument('--MODELNAME',type=str, default="AE", help='used in filename for saving') 
parser.add_argument('--add',type=str, default="N/A", help ='addition to the name used for data saving')
parser.add_argument('--is_distributed', type=bool, default=False)
parser.add_argument('--SEED', type=int, default=0)
args = parser.parse_args()

# create save name for the current run   
ID_HANDLE = datetime.now()
if args.add == "N/A":
    args.NAME = str(args.DATATYPE + ID_HANDLE.strftime('%m%d%H%M') + args.MODELNAME)
else:
    args.NAME = str(args.DATATYPE + ID_HANDLE.strftime('%m%d%H%M') + args.MODELNAME +args.add)

# ------------------------------------------------------------------------------------------
# ----------------------------------- initialization    ------------------------------------
# ------------------------------------------------------------------------------------------
    
set_random_seeds(args.SEED)
rng_gen = torch.Generator()
rng_gen.manual_seed(args.SEED)

RANK = int(os.environ.get('RANK'))
LOCAL_RANK = int(os.environ.get('LOCAL_RANK'))
WORLD_SIZE = int(os.environ.get('WORLD_SIZE'))
DEVICE_ORDINAL = 0 
device = torch.device("cuda:{}".format(DEVICE_ORDINAL))

assert RANK != None 
assert LOCAL_RANK != None 
assert WORLD_SIZE != None 
   
if WORLD_SIZE > 0:
    args.is_distributed = True

# Initializes the distributed backend which will take care of sychronizing nodes/GPUs
torch.distributed.init_process_group(backend="nccl", timeout=timedelta(seconds=1200000))
dist.barrier()

if args.RESNET == "yes": # using Resnet
    args.RESNET = True
else: # using standard AE
    args.RESNET = False

# init logging pathes, folders etc
if RANK == 0:    
    if args.RESNET == True:
        args.LOG_PATH = os.path.join(args.LOGPATH, '{}'.format(args.NAME) + '_RES')
    else:
        args.LOG_PATH = os.path.join(args.LOGPATH, '{}'.format(args.NAME))
    args.CHECKPOINT_PATH = os.path.join(args.LOG_PATH, 'checkpoints')

    if not os.path.exists(args.CHECKPOINT_PATH):
        os.makedirs(args.CHECKPOINT_PATH)

    # tensorboard logging:
    writer = SummaryWriter(log_dir=args.LOG_PATH, filename_suffix='{}'.format(args.NAME))
    log_file = open(os.path.join(args.LOG_PATH, 'log.txt'), "w")

# encoder and decoder feature maps
args.planes = [args.channel1, args.channel1*2, args.channel1*4, args.channel1*8]
args.decchannels = args.planes[::-1][:]

# dataloader
train_data = LoadTCFDataset(args.trainData_dir)
valid_data = LoadTCFDataset(args.validData_dir)

# distributed datasampler
train_sampler = torch.utils.data.DistributedSampler(train_data, shuffle=True) if args.is_distributed else None
valid_sampler = torch.utils.data.DistributedSampler(valid_data, shuffle=False) if args.is_distributed else None

# dataloader
trainLoader = DataLoader(
    dataset=train_data, 
    batch_size=args.BATCH_SIZE, 
    shuffle=False, 
    num_workers=0, 
    pin_memory=False, 
    sampler=train_sampler,
    drop_last=True, 
    worker_init_fn=seed_worker, 
	generator=rng_gen, 
    ) 

valLoader = DataLoader(
    dataset=valid_data, 
    batch_size=args.BATCH_SIZE, 
    shuffle=False, 
    num_workers=0, 
    pin_memory=False, 
    sampler=valid_sampler,
    drop_last=False, 
    worker_init_fn=seed_worker,
	generator=rng_gen,
    ) 

dist.barrier()

model = Model(
    enclayers=args.layers_enc, 
    planes=args.planes, 
    declayers=args.layers_dec, 
    decchannels=args.decchannels, 
    outputSize=[args.INPUT_IMAGE_HEIGHT,args.INPUT_IMAGE_WIDTH], 
    scaleFactor=args.SCALE_DECODER,
    resnet=args.RESNET,
    )

if RANK == 0:
    TOTAL_PARAMS = count_parameters(model)

model.to(device)
if args.add.__contains__("nodrop"):
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[DEVICE_ORDINAL], output_device=DEVICE_ORDINAL)
else:
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[DEVICE_ORDINAL], output_device=DEVICE_ORDINAL,find_unused_parameters=True)

# define optimizer
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=args.wd)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=args.reduce_factor,
                                                    patience=args.patience_level, min_lr=args.min_lr, threshold=1e-3)

# if loading a checkpoint and training with the pre-trained network
if args.recover == "yes":
    args.recover = True
    args.LOG_PATH_RECOV = os.path.join(args.LOGPATH, '{}'.format(args.recover_ckpt)) + '/checkpoints/'
    args.input_path_ckpt = glob.glob(os.path.join(args.LOG_PATH_RECOV, 'best_model_*'))
    checkpoint = torch.load(args.input_path_ckpt[-1], map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'],strict=False)
    optimizer.load_state_dict(checkpoint['optimizer_state_dict']) 
    scheduler.load_state_dict(checkpoint['scheduler_state_dict']) 
else:
    args.recover = False
    
# performance metrics
best_val_metric = 1e10
if RANK == 0:
    epoch_h5 = []
    train_lossh5 = []
    val_lossh5 = []

# save arguments
if RANK == 0:
    with open(os.path.join(args.LOG_PATH, 'args.txt'), 'w') as f: 
        json.dump(args.__dict__, f, indent=2)

# ------------------------------------------------------------------------------------------
# ----------------------------------  start training    ------------------------------------
# ------------------------------------------------------------------------------------------

for epoch in range(args.EPOCHS):
    # --------------------------------------------------------------------
    # -----  training
    # --------------------------------------------------------------------
    model.train()
    train_sampler.set_epoch(epoch) 
    total_loss = 0.
    total_batches = 0

    if RANK == 0:
        train_pbar = tqdm(enumerate(trainLoader), total=trainLoader.__len__(), desc='Epoch %s / %s TRAINING'%(epoch +1, args.EPOCHS), file=sys.stdout)
    else:
        train_pbar = tqdm(enumerate(trainLoader), total=trainLoader.__len__(), desc='Epoch %s / %s TRAINING'%(epoch +1, args.EPOCHS), disable=True)
    
    if epoch > 0:
        freeze_some_layers(model, args.drop_ratio, epoch)

    for i, train_batch in train_pbar:
        u_OL = train_batch[:,[0],0:args.INPUT_IMAGE_HEIGHT,1,0:args.INPUT_IMAGE_WIDTH].to(device) # (batchsize x channel x image dim 1 x image dim 2) = (7 x 1 x 128 x 128) default
        u_IL = train_batch[:,[0],0:args.INPUT_IMAGE_HEIGHT,0,0:args.INPUT_IMAGE_WIDTH].to(device)

        optimizer.zero_grad()        
        x_recon = model(u_OL)
        loss = lossFunc(x_recon, u_IL)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_batches += 1

        postfix_str = 'loss: %f, avg_loss: %f, lr: %f' %(loss.item(), total_loss / float(total_batches),optimizer.__getattribute__('param_groups')[0]['lr'])
        train_pbar.set_postfix_str(postfix_str)
        
        # store data in tensorboard
        if RANK == 0 and (int(epoch) % 20)==0 and i == 0:
            recon_batch = x_recon[0,:,:].squeeze().detach().cpu().numpy()
            gt_batch = u_IL[0,:,:].squeeze().detach().cpu().numpy()
            orig_batch = u_OL[0,:,:].squeeze().detach().cpu().numpy()

            plt.figure()
            plt.imshow(recon_batch)
            plt.colorbar()
            writer.add_figure('images-train/u_recon', plt.gcf(), epoch)
            plt.close()

            plt.figure()
            plt.imshow(orig_batch)
            plt.colorbar()
            writer.add_figure('images-train/u_GT_OL', plt.gcf(), epoch)
            plt.close()

            plt.figure()
            plt.imshow(gt_batch)
            plt.colorbar()
            writer.add_figure('images-train/u_GT_IL', plt.gcf(), epoch)
            plt.close()

            plt.figure()
            plt.imshow(gt_batch - recon_batch)
            plt.colorbar()
            writer.add_figure('images-train/u_recon_error', plt.gcf(), epoch)
            plt.close()    
    
    unfreeze(model)    
    avg_train_loss = total_loss / float(total_batches)

    if RANK == 0:
        writer.add_scalar("training/loss", avg_train_loss, epoch)
        writer.add_scalar("training/lr", optimizer.__getattribute__('param_groups')[0]['lr'], epoch)
        epoch_h5.append(epoch)
        train_lossh5.append(avg_train_loss)        
    
    # --------------------------------------------------------------------
    # -----  validation
    # --------------------------------------------------------------------        
    with torch.no_grad():
        model.eval()
        total_val_loss = 0.
        total_val_batches = 0
        
        if RANK == 0:
            val_pbar = tqdm(enumerate(valLoader), total=valLoader.__len__(), desc='Epoch %s / %s VALIDATING'%(epoch +1, args.EPOCHS))
        else:
            val_pbar = tqdm(enumerate(valLoader), total=valLoader.__len__(), desc='Epoch %s / %s VALIDATING'%(epoch +1, args.EPOCHS), disable=True)
    
        for i, val_batch in val_pbar:
            u_OL = val_batch[:,[0],0:args.INPUT_IMAGE_HEIGHT,1,0:args.INPUT_IMAGE_WIDTH].to(device)  # (batchsize x channel x image dim 1 x image dim 2) = (7 x 1 x 128 x 128) default
            u_IL = val_batch[:,[0],0:args.INPUT_IMAGE_HEIGHT,0,0:args.INPUT_IMAGE_WIDTH].to(device) 

            optimizer.zero_grad()            
            x_recon= model(u_OL)
            loss =  lossFunc(x_recon, u_IL)            
            total_val_loss += loss.item()
            total_val_batches += 1
            
            postfix_str = 'loss: %f, avg_loss: %f, lr: %f' %(loss.item(), total_val_loss / float(total_val_batches),optimizer.__getattribute__('param_groups')[0]['lr'])
            val_pbar.set_postfix_str(postfix_str)
            
            if RANK == 0 and (int(epoch) % 20)==0 and (i == 0):
                recon_batch = x_recon[0,:,:].squeeze().detach().cpu().numpy()
                gt_batch = u_IL[0,:,:].squeeze().detach().cpu().numpy()
                orig_batch = u_OL[0,:,:].squeeze().detach().cpu().numpy()

                plt.figure()
                plt.imshow(recon_batch)
                plt.colorbar()
                writer.add_figure('images-val/u_recon', plt.gcf(), epoch)
                plt.close()

                plt.figure()
                plt.imshow(orig_batch)
                plt.colorbar()
                writer.add_figure('images-val/u_GT_OL', plt.gcf(), epoch)
                plt.close()

                plt.figure()
                plt.imshow(gt_batch)
                plt.colorbar()
                writer.add_figure('images-val/u_GT_IL', plt.gcf(), epoch)
                plt.close()

                plt.figure()
                plt.imshow(gt_batch - recon_batch)
                plt.colorbar()
                writer.add_figure('images-val/u_recon_error', plt.gcf(), epoch)
                plt.close()               

        avg_val_loss = total_val_loss/float(total_val_batches)
        scheduler.step(avg_val_loss)
        
        if RANK == 0:    
            writer.add_scalar("val/loss", avg_val_loss, epoch)
            val_lossh5.append(avg_val_loss)
            
        output_str = 'Epoch %s avg_train_loss: %f, avg_val_loss: %f, lr: %f' % (epoch+1,
                                                                                avg_train_loss, 
                                                                                avg_val_loss,        
                                                                                optimizer.__getattribute__('param_groups')[0]['lr']
                                                                                )

        if RANK == 0:
            tqdm.write(output_str)
            log_file.write(output_str+'\n')
            log_file.flush()

        if RANK == 0:
            if avg_val_loss < best_val_metric:
                best_val_metric = min(best_val_metric, avg_val_loss)
                filelist = glob.glob(os.path.join(args.CHECKPOINT_PATH, "best_model_*.chkpt"))
                for f in filelist:
                    os.remove(f)
                torch.save({'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'scheduler_state_dict': scheduler.state_dict(),
                            'epoch': epoch},
                            os.path.join(args.CHECKPOINT_PATH, 'best_model_%s_%f.chkpt' % (epoch, best_val_metric)))

if RANK == 0:    
    with h5py.File(os.path.join(args.LOG_PATH,"loss.hdf5"), "w") as data_file:
        data_file.create_dataset("epochs",data=epoch_h5)
        data_file.create_dataset("train_loss",data=train_lossh5)
        data_file.create_dataset("val_loss",data=val_lossh5)