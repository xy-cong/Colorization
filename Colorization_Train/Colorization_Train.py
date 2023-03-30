import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "5"
import sys
import torch
import numpy as np
from pyhocon import ConfigFactory
from datetime import datetime
from tqdm import tqdm
from dataset.dataset import ColorDataset
from model.color_model import Colorization
from model.loss import Colorization_Loss
import cv2
from skimage import io
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
torch.autograd.set_detect_anomaly(True)

def mkdir_ifnotexists(directory):
    if not os.path.exists(directory): 
        os.mkdir(directory)

class Colorization_Train_runner():
    def __init__(self, conf):
        self.conf = ConfigFactory.parse_file(conf)
        self.train_conf = self.conf.get_config('train')
        self.nepochs = self.train_conf.nepoch
        self.start_epoch = 0 
        self.type = self.conf.get_config('type').type
        self.plot = self.train_conf.plot_freq
        self.test_plot = self.train_conf.test_plot_freq
        
        
    
    def Create_Path(self):
        mkdir_ifnotexists(os.path.join('',self.exps_folder_name))
        self.expdir = os.path.join('', self.exps_folder_name, self.expname)
        mkdir_ifnotexists(self.expdir)
        self.timestamp = '{:%Y_%m_%d_%H_%M_%S}'.format(datetime.now())
        mkdir_ifnotexists(os.path.join(self.expdir, self.timestamp))
        
        self.checkpoints_path = os.path.join(self.expdir, self.timestamp, 'checkpoints')
        mkdir_ifnotexists(self.checkpoints_path)
        self.model_params_subdir = "ModelParameters"
        self.optimizer_params_subdir = "OptimizerParameters"
        self.scheduler_params_subdir = "SchedulerParameters"
        self.plot_subdir = "Plots"
        self.conf_subdir = "Config"

        mkdir_ifnotexists(os.path.join(self.checkpoints_path, self.model_params_subdir))
        mkdir_ifnotexists(os.path.join(self.checkpoints_path, self.optimizer_params_subdir))
        mkdir_ifnotexists(os.path.join(self.checkpoints_path, self.scheduler_params_subdir))
        mkdir_ifnotexists(os.path.join(self.checkpoints_path, self.plot_subdir))
        mkdir_ifnotexists(os.path.join(self.checkpoints_path, self.conf_subdir))
        with open(os.path.join(self.checkpoints_path, self.conf_subdir, "Color.conf"), 'w') as f:
            # import ipdb; ipdb.set_trace()
            f.write(str(self.conf.as_plain_ordered_dict()))
            f.close()
        
        self.save_plot_path = os.path.join(self.checkpoints_path, self.plot_subdir)
        
        
    def Load_Data(self):
        print('... Loading Data -- self.type ...')
        dataset_conf = self.conf.get_config('dataset')
        if self.type == 'train':
            self.train_dataset = ColorDataset(dataset_conf.train)
            self.train_dataloader = torch.utils.data.DataLoader(self.train_dataset,
                                                                batch_size=self.train_conf.batch_size,
                                                                shuffle=True)
            self.test_dataset = ColorDataset(dataset_conf.test)
            self.test_dataloader = torch.utils.data.DataLoader(self.test_dataset,
                                                                batch_size=self.train_conf.batch_size,
                                                                shuffle=True)  
        else:
            self.test_dataset = ColorDataset(dataset_conf.test)
            self.test_dataloader = torch.utils.data.DataLoader(self.test_dataset,
                                                                batch_size=self.train_conf.batch_size,
                                                                shuffle=True)        
        print('... Finished ...')
        
    def Create_Model(self):
        print('Creating Model ...')
        model_conf = self.conf.get_config('model')
        self.color_model = Colorization(model_conf, image_size=self.conf.get_config('dataset').image_size)
        if torch.cuda.is_available():
            self.color_model.cuda()
        print('... Finished ...')
    
    def Create_Loss(self):
        print('Creating Loss ...')
        self.loss_conf = self.conf.get_config('loss')
        self.loss = Colorization_Loss(self.loss_conf)
        print('... Finished ...')
        
    def Create_Optimizer(self):
        print('Creating Optimizer ...')
        self.optim_conf = self.conf.get_config('optim')
        self.optimizer = torch.optim.Adam(self.color_model.parameters(), lr=self.optim_conf.lr)
        print('... Finished ...')
        
    def Create_Scheduler(self):
        print('...Creating Scheduler ...')
        self.scheduler_conf = self.conf.get_config('scheduler')
        decay_rate = self.scheduler_conf.decay_rate
        decay_steps = self.nepochs * len(self.train_dataset)
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, decay_rate ** (1./decay_steps))
        print('... Finished ...')
    
    def train(self):
        self.Load_Data()
        self.Create_Model()
        self.Create_Loss()
        self.Create_Optimizer()
        self.Create_Scheduler()
        self.load_checkpoints()
        print("... training...")
        print(f"... start_epoch: {self.start_epoch}...")
        for epoch in range(self.start_epoch, self.nepochs+1):
            print("Epoch: ", epoch)
            Loss = 0
            for data_index, imgs_input in tqdm(enumerate(self.train_dataloader)):
                model_input = imgs_input
                model_output = self.color_model(model_input)
                loss_output = self.loss(model_output) # gt
                Loss += loss_output
                self.optimizer.zero_grad()
                loss_output.backward()
                self.optimizer.step()
                self.scheduler.step()
                if (epoch+1) % self.plot == 0 and data_index % 100 == 0:
                # if True:
                    mkdir_ifnotexists(os.path.join(self.save_plot_path, f'epoch_{epoch}'))
                    img_grey = (imgs_input['img_GREY'].permute(1,2,0).cpu().detach().numpy()*255.0).astype('uint8')
                    img_ref = (model_output['Img_Ref']*255.0).astype('uint8')
                    # img_colored = (model_output['Img_Fine_RGB'][0].cpu().detach().numpy()*255.0).astype('uint8')
                    img_colored = (np.clip(model_output['Img_Fine_RGB'][0].cpu().detach().numpy(), 0, 1)*255.0).astype('uint8')
                    img_GT = (model_output['Img_RGB'][0].cpu().detach().numpy()*255.0).astype('uint8')
                    img_FAKE =  (model_output['Img_FAKE_RGB'][0].cpu().detach().numpy()*255.0).astype('uint8')
                    img_coarse =  (model_output['Img_Coarse_RGB'][0].permute(1,2,0).cpu().detach().numpy()*255.0).astype('uint8')
                    img_grey = np.concatenate((img_grey, img_grey, img_grey), -1)
                    TPS_mask = (model_output['TPS_mask']*255.0).astype('uint8')
                    TPS_img = (model_output['TPS_img'][0].permute(1,2,0).cpu().detach().numpy()*255.0).astype('uint8')
                    TPS_future_img = (model_output['TPS_future_img'][0].permute(1,2,0).cpu().detach().numpy()*255.0).astype('uint8')
                    
                    raw_size = (imgs_input['original_size'][0].item(), imgs_input['original_size'][1].item())
                    img_colored_raw = cv2.resize(img_colored, raw_size, cv2.INTER_CUBIC)
                    img_grey_raw = cv2.resize(img_grey, raw_size, cv2.INTER_CUBIC)
                    img_GT_raw = cv2.resize(img_GT, raw_size, cv2.INTER_CUBIC)
                    tmp_1 = np.concatenate((img_grey_raw, img_colored_raw, img_GT_raw), axis=1)
                    
                    img_coarse_raw = cv2.resize(img_coarse, raw_size, cv2.INTER_CUBIC)
                    img_fake_raw = cv2.resize(img_FAKE, raw_size, cv2.INTER_CUBIC)
                    img_ref_raw = cv2.resize(img_ref, raw_size, cv2.INTER_CUBIC)
                    # import ipdb; ipdb.set_trace()
                    tmp_2 = np.concatenate((img_ref_raw, img_coarse_raw, img_fake_raw), axis=1)
                    
                    TPS_mask_raw = cv2.resize(TPS_mask, raw_size, cv2.INTER_CUBIC)
                    TPS_img_raw = cv2.resize(TPS_img, raw_size, cv2.INTER_CUBIC)
                    TPS_future_img_raw = cv2.resize(TPS_future_img, raw_size, cv2.INTER_CUBIC)
                    tmp_3 = np.concatenate((TPS_img_raw, TPS_mask_raw, TPS_future_img_raw), axis=1)                  
                    
                    img_plot_raw = np.concatenate((tmp_1, tmp_2, tmp_3), axis=0)
                    
                    tmp_1 = np.concatenate((img_grey, img_colored, img_GT),axis=1)
                    tmp_2 = np.concatenate((img_ref, img_coarse, img_FAKE),axis=1)
                    tmp_3 = np.concatenate((TPS_img, TPS_mask, TPS_future_img),axis=1)
                    
                    
                    img_plot = np.concatenate((tmp_1, tmp_2, tmp_3), axis=0)
                    io.imsave(os.path.join(os.path.join(self.save_plot_path, f'epoch_{epoch}'), f"{imgs_input['img_name'][0]}"), img_plot)
                    io.imsave(os.path.join(os.path.join(self.save_plot_path, f'epoch_{epoch}'), f"_original_size_{imgs_input['img_name'][0]}"), img_plot_raw)
            
            print("Average_Loss: ", Loss / len(self.train_dataloader))
            if (epoch+1) % self.test_plot == 0:
                print(f"Epoch: {epoch}, Test plot", )
                self.color_model.eval()
                with torch.no_grad():
                    for data_index, imgs_input in tqdm(enumerate(self.test_dataloader)):
                        if data_index % 100 == 0:
                            model_input = imgs_input
                            model_output = self.color_model(model_input)
                            mkdir_ifnotexists(os.path.join(self.save_plot_path, f'epoch_{epoch}_test'))
                            img_grey = (imgs_input['img_GREY'].permute(1,2,0).cpu().detach().numpy()*255.0).astype('uint8')
                            img_ref = (model_output['Img_Ref']*255.0).astype('uint8')
                            # img_colored = (model_output['Img_Fine_RGB'][0].cpu().detach().numpy()*255.0).astype('uint8')
                            img_colored = (np.clip(model_output['Img_Fine_RGB'][0].cpu().detach().numpy(), 0, 1)*255.0).astype('uint8')
                            img_GT = (model_output['Img_RGB'][0].cpu().detach().numpy()*255.0).astype('uint8')
                            img_FAKE =  (model_output['Img_FAKE_RGB'][0].cpu().detach().numpy()*255.0).astype('uint8')
                            img_coarse =  (model_output['Img_Coarse_RGB'][0].permute(1,2,0).cpu().detach().numpy()*255.0).astype('uint8')
                            img_grey = np.concatenate((img_grey, img_grey, img_grey), -1)
                            TPS_mask = (model_output['TPS_mask']*255.0).astype('uint8')
                            TPS_img = (model_output['TPS_img'][0].permute(1,2,0).cpu().detach().numpy()*255.0).astype('uint8')
                            TPS_future_img = (model_output['TPS_future_img'][0].permute(1,2,0).cpu().detach().numpy()*255.0).astype('uint8')
                            
                            
                            raw_size = (imgs_input['original_size'][0].item(), imgs_input['original_size'][1].item())
                            img_colored_raw = cv2.resize(img_colored, raw_size, cv2.INTER_CUBIC)
                            img_grey_raw = cv2.resize(img_grey, raw_size, cv2.INTER_CUBIC)
                            img_GT_raw = cv2.resize(img_GT, raw_size, cv2.INTER_CUBIC)
                            tmp_1 = np.concatenate((img_grey_raw, img_colored_raw, img_GT_raw), axis=1)
                            
                            img_coarse_raw = cv2.resize(img_coarse, raw_size, cv2.INTER_CUBIC)
                            img_fake_raw = cv2.resize(img_FAKE, raw_size, cv2.INTER_CUBIC)
                            img_ref_raw = cv2.resize(img_ref, raw_size, cv2.INTER_CUBIC)
                            # import ipdb; ipdb.set_trace()
                            tmp_2 = np.concatenate((img_ref_raw, img_coarse_raw, img_fake_raw), axis=1)
                            
                            TPS_mask_raw = cv2.resize(TPS_mask, raw_size, cv2.INTER_CUBIC)
                            TPS_img_raw = cv2.resize(TPS_img, raw_size, cv2.INTER_CUBIC)
                            TPS_future_img_raw = cv2.resize(TPS_future_img, raw_size, cv2.INTER_CUBIC)
                            tmp_3 = np.concatenate((TPS_img_raw, TPS_mask_raw, TPS_future_img_raw), axis=1)                  
                            
                            img_plot_raw = np.concatenate((tmp_1, tmp_2, tmp_3), axis=0)
                            
                            tmp_1 = np.concatenate((img_grey, img_colored, img_GT),axis=1)
                            tmp_2 = np.concatenate((img_ref, img_coarse, img_FAKE),axis=1)
                            tmp_3 = np.concatenate((TPS_img, TPS_mask, TPS_future_img),axis=1)
                            
                            
                            img_plot = np.concatenate((tmp_1, tmp_2, tmp_3), axis=0)
                            io.imsave(os.path.join(os.path.join(self.save_plot_path, f'epoch_{epoch}_test'), f"{imgs_input['img_name'][0]}"), img_plot)
                            io.imsave(os.path.join(os.path.join(self.save_plot_path, f'epoch_{epoch}_test'), f"_original_size_{imgs_input['img_name'][0]}"), img_plot_raw)
                self.color_model.train()
            if epoch % self.save_epoch == 0:
                self.save_checkpoints(epoch)
                
    def test(self):
        self.Load_Data()
        self.Create_Model()
        self.Create_Loss()
        self.load_checkpoints()
        self.color_model.eval()
        print("... testing...")
        print(f"... testing_epoch: {self.start_epoch}...")
        mkdir_ifnotexists("eval")
        mkdir_ifnotexists(os.path.join("eval", self.expname))
        mkdir_ifnotexists(os.path.join(os.path.join("eval", self.expname), f'Epoch_{self.start_epoch}'))
        self.save_plot_path = os.path.join(os.path.join(os.path.join("eval", self.expname), f'Epoch_{self.start_epoch}'), 'save_plots')
        mkdir_ifnotexists(self.save_plot_path)
        with torch.no_grad():
            for data_index, imgs_input in tqdm(enumerate(self.test_dataloader)):
                if data_index % 50 == 0:
                    model_input = imgs_input
                    model_output = self.color_model(model_input)
                    img_grey = (imgs_input['img_GREY'].permute(1,2,0).cpu().detach().numpy()*255.0).astype('uint8')
                    img_ref = (model_output['Img_Ref']*255.0).astype('uint8')
                    # img_colored = (model_output['Img_Fine_RGB'][0].cpu().detach().numpy()*255.0).astype('uint8')
                    img_colored = (np.clip(model_output['Img_Fine_RGB'][0].cpu().detach().numpy(), 0, 1)*255.0).astype('uint8')
                    img_GT = (model_output['Img_RGB'][0].cpu().detach().numpy()*255.0).astype('uint8')
                    img_FAKE =  (model_output['Img_FAKE_RGB'][0].cpu().detach().numpy()*255.0).astype('uint8')
                    img_coarse =  (model_output['Img_Coarse_RGB'][0].permute(1,2,0).cpu().detach().numpy()*255.0).astype('uint8')
                    img_grey = np.concatenate((img_grey, img_grey, img_grey), -1)
                    TPS_mask = (model_output['TPS_mask']*255.0).astype('uint8')
                    TPS_img = (model_output['TPS_img'][0].permute(1,2,0).cpu().detach().numpy()*255.0).astype('uint8')
                    TPS_future_img = (model_output['TPS_future_img'][0].permute(1,2,0).cpu().detach().numpy()*255.0).astype('uint8')
                    
                    raw_size = (imgs_input['original_size'][0].item(), imgs_input['original_size'][1].item())
                    img_colored_raw = cv2.resize(img_colored, raw_size, cv2.INTER_CUBIC)
                    img_grey_raw = cv2.resize(img_grey, raw_size, cv2.INTER_CUBIC)
                    img_GT_raw = cv2.resize(img_GT, raw_size, cv2.INTER_CUBIC)
                    tmp_1 = np.concatenate((img_grey_raw, img_colored_raw, img_GT_raw), axis=1)
                    
                    img_coarse_raw = cv2.resize(img_coarse, raw_size, cv2.INTER_CUBIC)
                    img_fake_raw = cv2.resize(img_FAKE, raw_size, cv2.INTER_CUBIC)
                    img_ref_raw = cv2.resize(img_ref, raw_size, cv2.INTER_CUBIC)
                    # import ipdb; ipdb.set_trace()
                    tmp_2 = np.concatenate((img_ref_raw, img_coarse_raw, img_fake_raw), axis=1)
                    
                    TPS_mask_raw = cv2.resize(TPS_mask, raw_size, cv2.INTER_CUBIC)
                    TPS_img_raw = cv2.resize(TPS_img, raw_size, cv2.INTER_CUBIC)
                    TPS_future_img_raw = cv2.resize(TPS_future_img, raw_size, cv2.INTER_CUBIC)
                    tmp_3 = np.concatenate((TPS_img_raw, TPS_mask_raw, TPS_future_img_raw), axis=1)                  
                    
                    img_plot_raw = np.concatenate((tmp_1, tmp_2, tmp_3), axis=0)
                    
                    tmp_1 = np.concatenate((img_grey, img_colored, img_GT),axis=1)
                    tmp_2 = np.concatenate((img_ref, img_coarse, img_FAKE),axis=1)
                    tmp_3 = np.concatenate((TPS_img, TPS_mask, TPS_future_img),axis=1)
                    
                    
                    img_plot = np.concatenate((tmp_1, tmp_2, tmp_3), axis=0)
                    io.imsave(os.path.join(self.save_plot_path, f"{imgs_input['img_name'][0]}"), img_plot)
                    io.imsave(os.path.join(self.save_plot_path, f"_original_size_{imgs_input['img_name'][0]}"), img_plot_raw)

    def test_BLIP_ControlNet(self):
        self.Load_Data()
        self.Create_Model()
        self.Create_Loss()
        self.load_checkpoints()
        self.color_model.eval()
        print("... testing...")
        print(f"... testing_epoch: {self.start_epoch}...")
        mkdir_ifnotexists("eval")
        mkdir_ifnotexists(os.path.join("eval", self.expname))
        mkdir_ifnotexists(os.path.join(os.path.join("eval", self.expname), f'Epoch_{self.start_epoch}'))
        self.save_plot_path = os.path.join(os.path.join(os.path.join("eval", self.expname), f'Epoch_{self.start_epoch}'), 'save_plots')
        mkdir_ifnotexists(self.save_plot_path)
        with torch.no_grad():
            for data_index, imgs_input in tqdm(enumerate(self.test_dataloader)):
                if data_index % 50 == 0:
                    model_input = imgs_input
                    model_output = self.color_model(model_input)
                    img_grey = (imgs_input['img_GREY'].permute(1,2,0).cpu().detach().numpy()*255.0).astype('uint8')
                    img_ref = (model_output['Img_Ref']*255.0).astype('uint8')
                    # img_colored = (model_output['Img_Fine_RGB'][0].cpu().detach().numpy()*255.0).astype('uint8')
                    img_colored = (np.clip(model_output['Img_Fine_RGB'][0].cpu().detach().numpy(), 0, 1)*255.0).astype('uint8')
                    img_GT = (model_output['Img_RGB'][0].cpu().detach().numpy()*255.0).astype('uint8')
                    img_FAKE =  (model_output['Img_FAKE_RGB'][0].cpu().detach().numpy()*255.0).astype('uint8')
                    img_coarse =  (model_output['Img_Coarse_RGB'][0].permute(1,2,0).cpu().detach().numpy()*255.0).astype('uint8')
                    img_grey = np.concatenate((img_grey, img_grey, img_grey), -1)
                    TPS_mask = (model_output['TPS_mask']*255.0).astype('uint8')
                    TPS_img = (model_output['TPS_img'][0].permute(1,2,0).cpu().detach().numpy()*255.0).astype('uint8')
                    TPS_future_img = (model_output['TPS_future_img'][0].permute(1,2,0).cpu().detach().numpy()*255.0).astype('uint8')
                    
                    raw_size = (imgs_input['original_size'][0].item(), imgs_input['original_size'][1].item())
                    img_colored_raw = cv2.resize(img_colored, raw_size, cv2.INTER_CUBIC)
                    img_grey_raw = cv2.resize(img_grey, raw_size, cv2.INTER_CUBIC)
                    img_GT_raw = cv2.resize(img_GT, raw_size, cv2.INTER_CUBIC)
                    tmp_1 = np.concatenate((img_grey_raw, img_colored_raw, img_GT_raw), axis=1)
                    
                    img_coarse_raw = cv2.resize(img_coarse, raw_size, cv2.INTER_CUBIC)
                    img_fake_raw = cv2.resize(img_FAKE, raw_size, cv2.INTER_CUBIC)
                    img_ref_raw = cv2.resize(img_ref, raw_size, cv2.INTER_CUBIC)
                    # import ipdb; ipdb.set_trace()
                    tmp_2 = np.concatenate((img_ref_raw, img_coarse_raw, img_fake_raw), axis=1)
                    
                    TPS_mask_raw = cv2.resize(TPS_mask, raw_size, cv2.INTER_CUBIC)
                    TPS_img_raw = cv2.resize(TPS_img, raw_size, cv2.INTER_CUBIC)
                    TPS_future_img_raw = cv2.resize(TPS_future_img, raw_size, cv2.INTER_CUBIC)
                    tmp_3 = np.concatenate((TPS_img_raw, TPS_mask_raw, TPS_future_img_raw), axis=1)                  
                    
                    img_plot_raw = np.concatenate((tmp_1, tmp_2, tmp_3), axis=0)
                    
                    tmp_1 = np.concatenate((img_grey, img_colored, img_GT),axis=1)
                    tmp_2 = np.concatenate((img_ref, img_coarse, img_FAKE),axis=1)
                    tmp_3 = np.concatenate((TPS_img, TPS_mask, TPS_future_img),axis=1)
                    
                    
                    img_plot = np.concatenate((tmp_1, tmp_2, tmp_3), axis=0)
                    io.imsave(os.path.join(self.save_plot_path, f"{imgs_input['img_name'][0]}"), img_plot)
                    io.imsave(os.path.join(self.save_plot_path, f"_original_size_{imgs_input['img_name'][0]}"), img_plot_raw)

    
    def save_checkpoints(self, epoch):
        torch.save(
            {"epoch": epoch, "model_state_dict": self.color_model.state_dict()},
            os.path.join(self.checkpoints_path, self.model_params_subdir, str(epoch) + ".pth"))
        torch.save(
            {"epoch": epoch, "model_state_dict": self.color_model.state_dict()},
            os.path.join(self.checkpoints_path, self.model_params_subdir, "latest.pth"))

        torch.save(
            {"epoch": epoch, "optimizer_state_dict": self.optimizer.state_dict()},
            os.path.join(self.checkpoints_path, self.optimizer_params_subdir, str(epoch) + ".pth"))
        torch.save(
            {"epoch": epoch, "optimizer_state_dict": self.optimizer.state_dict()},
            os.path.join(self.checkpoints_path, self.optimizer_params_subdir, "latest.pth"))

        torch.save(
            {"epoch": epoch, "scheduler_state_dict": self.scheduler.state_dict()},
            os.path.join(self.checkpoints_path, self.scheduler_params_subdir, str(epoch) + ".pth"))
        torch.save(
            {"epoch": epoch, "scheduler_state_dict": self.scheduler.state_dict()},
            os.path.join(self.checkpoints_path, self.scheduler_params_subdir, "latest.pth"))
        
    def load_checkpoints(self):
        self.save_conf = self.conf.get_config('save')
        self.exps_folder_name = self.save_conf.exps_folder_name
        self.expname = self.save_conf.expname
        self.save_epoch = self.save_conf.save_epoch
        print("... Loading ...")
        if self.train_conf.is_continue and self.train_conf.timestamp == 'latest':
            if os.path.exists(os.path.join('',self.exps_folder_name,self.expname)):
                timestamps = os.listdir(os.path.join('',self.exps_folder_name,self.expname))
                if (len(timestamps)) == 0:
                    is_continue = False
                    timestamp = None
                else:
                    timestamp = sorted(timestamps)[-1]
                    is_continue = True
            else:
                is_continue = False
                timestamp = None
        else:
            timestamp = self.train_conf.timestamp
            is_continue = self.train_conf.is_continue
        if self.type == 'train':
            self.Create_Path()
            if is_continue:
                # import ipdb; ipdb.set_trace()
                old_checkpnts_dir = os.path.join(self.expdir, timestamp, 'checkpoints')

                saved_model_state = torch.load(
                    os.path.join(old_checkpnts_dir, 'ModelParameters', str(self.train_conf.timestamp) + ".pth"))
                # print(old_checkpnts_dir)
                # print(os.path.join(old_checkpnts_dir, 'ModelParameters', str(self.train_conf.timestamp) + ".pth"))
                self.color_model.load_state_dict(saved_model_state["model_state_dict"])
                self.start_epoch = saved_model_state['epoch']

                data = torch.load(
                    os.path.join(old_checkpnts_dir, 'OptimizerParameters', str(self.train_conf.timestamp) + ".pth"))
                self.optimizer.load_state_dict(data["optimizer_state_dict"])

                data = torch.load(
                    os.path.join(old_checkpnts_dir, self.scheduler_params_subdir, str(self.train_conf.timestamp) + ".pth"))
                self.scheduler.load_state_dict(data["scheduler_state_dict"])
                print('... Finished loading ...')
                return
        else:
            self.expdir = os.path.join('', self.exps_folder_name, self.expname)
            self.timestamp = '{:%Y_%m_%d_%H_%M_%S}'.format(datetime.now())
            if is_continue:
                # import ipdb; ipdb.set_trace()
                old_checkpnts_dir = os.path.join(self.expdir, timestamp, 'checkpoints')
                saved_model_state = torch.load(
                    os.path.join(old_checkpnts_dir, 'ModelParameters', str(self.train_conf.timestamp) + ".pth"))
                self.color_model.load_state_dict(saved_model_state["model_state_dict"])
                self.start_epoch = saved_model_state['epoch']
                print('... Finished loading ...')
                return
        print('... Finished no load ...')