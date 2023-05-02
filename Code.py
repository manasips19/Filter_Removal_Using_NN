import math, shutil, argparse, os, sys
import numpy as np

import torch
import torch.nn as nn
import torch.nn.init as winit
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data

from torchvision import datasets, transforms, utils
from tqdm import tqdm

from Dataset import Dataset
from multiprocessing import cpu_count


if __name__ == '__main__':
	degin = 3
	degout = 3
	patchsize = 8
	nrow = 3
	indir = 'C:/Users/19793/Documents/2023 Spring/Computational Photography/Project/places-instagram/images/'
	gtdir = 'C:/Users/19793/Documents/2023 Spring/Computational Photography/Project/places-instagram/images_orig/'
	train_list = 'C:/Users/19793/Documents/2023 Spring/Computational Photography/Project/places-instagram/train-list.txt'
	validation_list = 'C:/Users/19793/Documents/2023 Spring/Computational Photography/Project/places-instagram/smallvalidation-list.txt'
	test_list = 'C:/Users/19793/Documents/2023 Spring/Computational Photography/Project/places-instagram/test-list.txt'


	def checkpoint_save(state, is_best, filename='checkpoint.pth'):
		torch.save(state, filename)
		if is_best:
			shutil.copyfile(filename, 'model_best.pth')


	def snr(image_1, image_2):
		mse = torch.mean( torch.pow( (image_1 - image_2),2 ) )
		if mse == 0:
			return 100
		PIXEL_MAX = 1.0
		return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))


	def batch_poly(batch, degree):
		if degree > 3: raise ValueError('Degree > 2 is not yet implemented')
		if degree == 1: return batch
		r,g,b = torch.unsqueeze(batch[:,0,:,:],1), torch.unsqueeze(batch[:,1,:,:],1), torch.unsqueeze(batch[:,2,:,:],1)
		
		ris = batch

		if degree > 1:			
			ris = torch.cat((ris,r.pow(2)),1) 
			ris = torch.cat((ris,g.pow(2)),1) 
			ris = torch.cat((ris,b.pow(2)),1) 
			ris = torch.cat((ris,b*g),1) 
			ris = torch.cat((ris,b*r),1) 
			ris = torch.cat((ris,g*r),1) 

		if degree > 2:			
			ris = torch.cat((ris,r.pow(3)),1) 
			ris = torch.cat((ris,g.pow(3)),1) 
			ris = torch.cat((ris,b.pow(3)),1) 
			ris = torch.cat((ris,g*b.pow(2)),1) 
			ris = torch.cat((ris,r*b.pow(2)),1) 
			ris = torch.cat((ris,b*g.pow(2)),1) 
			ris = torch.cat((ris,r*g.pow(2)),1) 
			ris = torch.cat((ris,b*r.pow(2)),1) 
			ris = torch.cat((ris,g*r.pow(2)),1) 
			ris = torch.cat((ris,b*g*r),1) 
		return ris


	# ------------------ NEURAL NETWORK ------------------
	class Neural_Net(nn.Module):
		def __init__(self, image_dim, patch_size, n_c, n_f, degree_in, degree_output):
			super(Neural_Net, self).__init__()
			self.image_dim = image_dim
			self.patch_size = patch_size
			self.degree_in = degree_in
			self.degree_output = degree_output
			
			self.nch_in = 3
			if degree_in > 1: self.nch_in = self.nch_in + 6
			if degree_in > 2: self.nch_in = self.nch_in + 10
			if degree_in > 3: raise ValueError('Degree > 3 is not yet implemented')
			self.nch_out = 3
			if degree_output > 1: self.nch_out = self.nch_out + 6
			if degree_output > 2: self.nch_out = self.nch_out + 10
			if degree_output > 3: raise ValueError('Degree > 3 is not yet implemented')
			
			self.h_patch = int(math.floor(image_dim[0]/patch_size))
			self.w_patch = int(math.floor(image_dim[1]/patch_size))
			self.n_patch = self.h_patch *self.w_patch

			# creating network layers
			self.b1 = nn.BatchNorm2d(self.nch_in)
			self.c1 = nn.Conv2d(self.nch_in, n_c, kernel_size=3, stride=2, padding=0)
			self.b2 = nn.BatchNorm2d(n_c)
			self.c2 = nn.Conv2d(n_c, n_c, kernel_size=3, stride=2, padding=0)
			self.b3 = nn.BatchNorm2d(n_c)
			self.c3 = nn.Conv2d(n_c, n_c, kernel_size=3, stride=2, padding=0)
			self.b4 = nn.BatchNorm2d(n_c)
			self.c4 = nn.Conv2d(n_c, n_c, kernel_size=3, stride=2, padding=0)
			self.b5 = nn.BatchNorm2d(n_c)
			self.c5 = nn.Conv2d(n_c, n_c, kernel_size=3, stride=2, padding=0)

			self.l1 = nn.Linear(n_c*7*7, n_f)
			self.l2 = nn.Linear(n_f, n_f)
			self.l3 = nn.Linear(n_f, self.n_patch*(self.nch_out*3+3)) 

			for m in self.modules():
				if isinstance(m, nn.Conv2d):
					n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
					m.weight.data.normal_(0, math.sqrt(2. / n))
				elif isinstance(m, nn.BatchNorm2d):
					m.weight.data.fill_(1)
					m.bias.data.zero_()

		def forward(self,y):			
			image = batch_poly(y, self.degree_output)			
			y = batch_poly(y, self.degree_in)			
			image = image.view(-1,self.nch_out, self.image_dim[0]*self.image_dim[1],1)
			image = image.clone().permute(0,2,1,3)
			
			y = F.relu(self.c1(self.b1(y)))
			y = F.relu(self.c2(self.b2(y)))
			y = F.relu(self.c3(self.b3(y)))
			y = F.relu(self.c4(self.b4(y)))
			y = F.relu(self.c5(self.b5(y)))
			y = y.view(-1, y.size(1)*y.size(2)*y.size(3))
			y = F.relu(self.l1(y))
			y = F.relu(self.l2(y))
			y = self.l3(y)
			
			y = y.view(-1, self.nch_out*3+3, self.h_patch, self.w_patch)			
			y = F.upsample(y,scale_factor=self.patch_size,  mode='bilinear')
			
			y = y.view(-1,self.nch_out*3+3,self.image_dim[0]*self.image_dim[1]) 			
			y = y.permute(0,2,1)			
			y = y.contiguous().view(-1,y.size(1),3,self.nch_out+1) 
			
			w = Variable( torch.ones(image.size(0),image.size(1),1,image.size(3)) )
			image = torch.cat((image,w),2)			
			ris = Variable(torch.zeros(image.size(0),image.size(1),3,image.size(3)))
			
			for bn in range(y.size(0)):
				ris[bn,:,:,:] = torch.bmm(y[bn,:,:,:].clone(),image[bn,:,:,:].clone())			
			ris = ris.permute(0,2,1,3)
			ris = ris.contiguous()
			ris = ris.view(-1,3, self.image_dim[0], self.image_dim[1])
			return ris


	# ---------------- parse arguments ---------------
	parser = argparse.ArgumentParser()
	parser.add_argument("-r", "--regen", help="Regenerating images",
						default="")
	parser.add_argument("-ps", "--patchsize", help="Patch dimensions",
						default=8, type=int)
	parser.add_argument("-nrow", "--nrow", help="Batchsize given as nrow*nrow",
						default=5, type=int)
	parser.add_argument("-di", "--degin", help="Degree of input",
						default=3, type=int)
	parser.add_argument("-do", "--degout", help="Degree of output",
						default=3, type=int)
	parser.add_argument("-indir", "--indir", help="Path to the folder containing filtered images",
						default='C:/Users/19793/Documents/2023 Spring/Computational Photography/Project/places-instagram/images/')
	parser.add_argument("-gtdir", "--gtdir", help="Path to the folder containing original images",
						default='C:/Users/19793/Documents/2023 Spring/Computational Photography/Project/places-instagram/images_orig/')
	parser.add_argument("-trl", "--train_list", help="Training Set",
						default='C:/Users/19793/Documents/2023 Spring/Computational Photography/Project/places-instagram/train-list.txt')
	parser.add_argument("-val", "--validation_list", help="Validation Set",
						default='C:/Users/19793/Documents/2023 Spring/Computational Photography/Project/places-instagram/smallvalidation-list.txt')
	parser.add_argument("-tsl", "--test_list", help="Testing Set",
						default='C:/Users/19793/Documents/2023 Spring/Computational Photography/Project/places-instagram/test-list.txt')
	args = parser.parse_args()

	
	conf_txt = ''
	for arg in vars(args):
		conf_txt = conf_txt + '{:>10} = '.format(arg) + str(getattr(args, arg)) + '\n'
	print(conf_txt)
	
	ouput_file = open("config.txt","w")
	ouput_file.write(conf_txt)
	ouput_file.close()

	# ------------------ TRAINING THE NN ------------------

	image_dim = [256,256]
	patch_size = args.patchsize
	nRow = args.nrow
	batchSize = nRow*nRow
	batchSizeVal = 50
	n_epochs = 4
	saveImagesEvery = 1
	saveModelEvery = 20
	n_c = 200
	n_f = 2000
	lr = 0.0001 
	degree_in = args.degin
	degree_output = args.degout
	
	net = Neural_Net(image_dim, patch_size, n_c, n_f, degree_in, degree_output)	
	Loss = nn.MSELoss()	
	optimizer = optim.Adam(net.parameters(), lr=lr)
	
	image_dir = [args.gtdir, args.indir]
	get_train = args.train_list
	get_validation = args.validation_list
	get_test = args.test_list
	
	train_loader = data.DataLoader(
			Dataset(image_dir, get_train, sep=','),
			batch_size = batchSize,
			shuffle = True,
			num_workers = cpu_count(),
	)
	validation_loader = data.DataLoader(
			Dataset(image_dir, get_validation, sep=','),
			batch_size = batchSize,
			shuffle = True,
			num_workers = cpu_count(),
	)
	test_loader = data.DataLoader(
			Dataset(image_dir, get_test, sep=',', include_filenames=True),
			batch_size = batchSize,
			shuffle = True,
			num_workers = cpu_count(),
	)

	best_psnr = np.inf
	if not args.regen:		
		for epoch in range(n_epochs):			
			for bn, (data, target) in enumerate(train_loader):				
				original, filtered = data				
				original, filtered, target = Variable(original), Variable(filtered), Variable(target)				
				original, filtered, target = original, filtered, target
				
				optimizer.zero_grad()				
				output = net(filtered)
				
				loss = Loss(output, original)				
				loss.backward()				
				optimizer.step()
				
				if bn%saveImagesEvery == 0:
					utils.save_image(original.data, './original.png', nrow=nRow)
					utils.save_image(filtered.data, './input.png', nrow=nRow)
					utils.save_image(output.data, './output.png', nrow=nRow)
				
				if bn%saveModelEvery == 0:
					checkpoint_save({
						'epoch': epoch + 1,
						'state_dict': net.state_dict(),
						'optimizer' : optimizer.state_dict(),
						'best_psnr' : best_psnr
					}, False)
				
				col = '\033[92m'
				endCol = '\033[0m'
				print('Epoch: [' + str(epoch+1) + '][' + str(bn+1) + '/' + str(len(train_loader)) + '] Loss = ' + col + str(round(loss.item(),4)) + endCol)
	else:
		
		print('Regenerating the images')		
		net.load_state_dict(torch.load(args.regen)['state_dict'])		
		net.train(False)

		for bn, (data, target, fns) in enumerate(tqdm(test_loader)):			
			original, filtered = data			
			original, filtered, target = Variable(original, requires_grad=False), Variable(filtered, requires_grad=False), Variable(target, requires_grad=False)			
			original, filtered, target = original, filtered, target			
			output = net(filtered)
			
			for i in range(output.size(0)):
				current_image = output[i,:,:,:].data
				current_filename = fns[i]
				current_filename = os.path.splitext(current_filename)[0] + '.png'
				if not os.path.isdir('./Test_Output/'): os.makedirs('./Test_Output/')
				utils.save_image(current_image, os.path.join('./Test_Output/', current_filename), nrow=1, padding=0)