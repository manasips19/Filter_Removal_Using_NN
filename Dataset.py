import os,sys
import torch.utils.data as data

from torchvision import transforms
from PIL import Image

class Dataset(data.Dataset):
	def __init__(self,dirs,listfn, include_filenames=False, sep=','):		
		if not isinstance(dirs, list): dirs = [dirs]
		self.dirs = dirs
		self.include_filenames = include_filenames
		self.sep = sep		
		in_file = open(listfn,"r")
		self.lines = in_file.read().split('\n')
		in_file.close()
		self.fns = [l for l in self.lines if l]

	def __getitem__(self, index):		
		parts = self.fns[index].split(self.sep)
		fn, lbl = parts[0], int(parts[1]) if len(parts)>1 else 0
		output = []		
		for d in self.dirs:			
			current_image = Image.open(os.path.join(d,fn))			
			if not current_image.mode == 'RGB':
				current_image = current_image.convert('RGB')			
			w, h = current_image.size
			if not (w==256 and h==256):
				current_image.resize((256,256), Image.ANTIALIAS)
			
			current_image = transforms.ToTensor()(current_image)			
			output.append(current_image)
		
		if self.include_filenames:
			return output, lbl, fn
		else:
			return output, lbl

	def __len__(self):
		return len(self.fns)
