import argparse
import os
from datetime import timedelta
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
import h5py
import glob
from tqdm import tqdm

from E1_b_AE_utils import *

# ------------------------------------------------------------------------------------------
# ----------------------------------- define arguments    ----------------------------------
# ------------------------------------------------------------------------------------------

parser = argparse.ArgumentParser()
parser.add_argument('--testData_dir', type=str, default="sample_testdata.hdf5")
parser.add_argument('--LOGPATH', type = str, default = "AE_results/", help = 'path to store logs')
parser.add_argument('--recover_ckpt', type =str, default = "RUNx", help = 'checkpoint folder: args.LOGPATH + args.recover_ckpt') 
parser.add_argument('--RESNET', type=str, default="no", help='"yes" or "no" if using residual blocks')
parser.add_argument('--INPUT_IMAGE_HEIGHT',type=int,default=128)
parser.add_argument('--INPUT_IMAGE_WIDTH',type=int,default=128)
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
    
# ------------------------------------------------------------------------------------------
# ----------------------------------- initialization    ------------------------------------
# ------------------------------------------------------------------------------------------

xm1, xm2 = 0, args.INPUT_IMAGE_WIDTH
zm1, zm2 = 0, args.INPUT_IMAGE_HEIGHT
savename = "test_output.hdf5"

set_random_seeds(0)
rng_gen = torch.Generator()
rng_gen.manual_seed(0)

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

# init logging pathes, folders etc.
if RANK == 0:
    args.LOG_PATH = os.path.join(args.LOGPATH, '{}'.format(args.recover_ckpt))
    args.CHECKPOINT_PATH = args.LOG_PATH + '/checkpoints/'
 
if args.RESNET == "yes":
    args.RESNET = True 
else:
    args.RESNET = False

# encoder and decoder layers and channels
args.planes = [args.channel1, args.channel1*2, args.channel1*4, args.channel1*8]
args.decchannels = args.planes[::-1][:]

test_data = LoadTCFDataset(args.testData_dir)
test_sampler = torch.utils.data.DistributedSampler(test_data, shuffle=False) if args.is_distributed else None
testLoader = DataLoader(
    dataset=test_data, 
    batch_size=args.BATCH_SIZE, 
    shuffle=False, 
    num_workers=0, 
    pin_memory=False,
    sampler=test_sampler,
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

model.to(device)

if args.add.__contains__("nodrop"):
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[DEVICE_ORDINAL], output_device=DEVICE_ORDINAL)
else:
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[DEVICE_ORDINAL], output_device=DEVICE_ORDINAL,find_unused_parameters=True)

optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=args.wd)

args.input_path_ckpt = glob.glob(os.path.join(args.CHECKPOINT_PATH, 'best_model_*'))
checkpoint = torch.load(args.input_path_ckpt[-1], map_location=device)
model.load_state_dict(checkpoint['model_state_dict'],strict=False)
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

recon_all = torch.zeros((test_data.__len__(), args.INPUT_IMAGE_HEIGHT, args.INPUT_IMAGE_WIDTH)) #samples, z, x
GT_all = torch.zeros((test_data.__len__(), args.INPUT_IMAGE_HEIGHT, 2, args.INPUT_IMAGE_WIDTH))
c = 0

# ------------------------------------------------------------------------------------------
# ----------------------------------  start testing    -------------------------------------
# ------------------------------------------------------------------------------------------

with torch.no_grad():
    model.eval()
    total_test_loss = 0.
    total_test_batches = 0
    loss_indv = []
    
    if RANK == 0:
        test_pbar = tqdm(enumerate(testLoader), total=testLoader.__len__(), desc='TESTING')
    else:
        test_pbar = tqdm(enumerate(testLoader), total=testLoader.__len__(), desc='TESTING', disable=True)

    for i, test_batch in test_pbar:
        u_OL = test_batch[:,[0],:args.INPUT_IMAGE_HEIGHT,1,:args.INPUT_IMAGE_WIDTH] # (batchsize x channel x image dim 1 x image dim 2) = (7 x 1 x 128 x 128) default
        u_IL = test_batch[:,[0],:args.INPUT_IMAGE_HEIGHT,0,:args.INPUT_IMAGE_WIDTH]
        u_OL, u_IL = u_OL.to(device), u_IL.to(device)

        optimizer.zero_grad()        
        x_recon= model(u_OL)
        loss = lossFunc(x_recon[:,:,zm1:zm2,xm1:xm2], u_IL[:,:,zm1:zm2,xm1:xm2]) 

        for sn in range(u_IL.shape[0]):
            loss_indv.append(lossFunc(x_recon[sn,:,zm1:zm2,xm1:xm2], u_IL[sn,:,zm1:zm2,xm1:xm2]))

        if i < (test_data.__len__()//args.BATCH_SIZE):
            recon_all[c*args.BATCH_SIZE:(c+1)*args.BATCH_SIZE,:,:] = x_recon.squeeze()
            GT_all[c*args.BATCH_SIZE:(c+1)*args.BATCH_SIZE,:,:] = test_batch[:,0,:args.INPUT_IMAGE_HEIGHT,:,:args.INPUT_IMAGE_WIDTH].squeeze() 

        else: # last batch with potentially less samples as the batch size
            recon_all[c*args.BATCH_SIZE:(c*args.BATCH_SIZE + (test_data.__len__() % args.BATCH_SIZE)),:,:] = x_recon.squeeze()
            GT_all[c*args.BATCH_SIZE:(c*args.BATCH_SIZE + (test_data.__len__() % args.BATCH_SIZE)),:,:] = test_batch[:,0,:args.INPUT_IMAGE_HEIGHT,:,:args.INPUT_IMAGE_WIDTH].squeeze() 
        c += 1

        total_test_loss += loss.item()
        total_test_batches += 1
        
        postfix_str = 'loss: %f, avg_loss: %f, lr: %f' %(loss.item(), total_test_loss / float(total_test_batches),optimizer.__getattribute__('param_groups')[0]['lr'])
        test_pbar.set_postfix_str(postfix_str)        

    avg_test_loss = total_test_loss/float(total_test_batches)
    print('test loss averaged across ' + str(test_data.__len__()) + ' samples: ' + str(avg_test_loss))

    # get lowest validation loss for comparison
    avg_valid_loss = args.input_path_ckpt[0].split('best_model_')[-1].split('_')[-1].split('.chkpt')[0]
    print('valid loss averaged: ' + str(avg_valid_loss))

    with h5py.File(os.path.join(args.LOG_PATH, savename), "w") as data_file:
        data_file.create_dataset("test_loss",data=avg_test_loss)
        data_file.create_dataset("test_loss_indv",data=torch.tensor(loss_indv).detach().cpu().numpy())
        data_file.create_dataset("test_recon",data=recon_all)
        data_file.create_dataset("test_input",data=GT_all)
        
print('finished')