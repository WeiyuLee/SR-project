import utils as ut
import cv2
import numpy as np
import os
import tensorflow as tf

class becnchmark:

	def __init__(self, input, label):

		self.input = input
		self.label = label

	def psnr(self):
		return
	def ssim(self):
		return

	def ifc(self):
		return


def split_img(imgname,img, padding_size, subimg_size):

	"""
	split a image into sub-images
	img: image for splitting
	padding_size: Size of padding
	subimg_size: Size of each subimage
	"""

	ori_size = img.shape
	assert len(ori_size) == 3, "the dimensions of image shall be (height, width, channel)! " 

	#Calculate image size without padding
	padded_size = [	ori_size[0] - 2*padding_size[0],
					ori_size[1] - 2*padding_size[1],
					ori_size[2]]

	strides = subimg_size
	sub_imgs = {}
	
	for r in range(padded_size[0]//subimg_size[0]):
		for c in range(padded_size[1]//subimg_size[1]):

			grid_r = padding_size[0] + r*strides[0] 
			grid_c = padding_size[1] + c*strides[1] 

			sub_img = img[	grid_r - padding_size[0] : grid_r + strides[0] + padding_size[0],
							grid_c - padding_size[1] : grid_c + strides[1] + padding_size[1],
							:]

			# insert sub image to dictionary with key = [imagename]_[row_index]_[col_index]
			sub_imgs[imgname + "_"+ str(grid_r) + "_" + str(grid_c)] = sub_img


	return sub_imgs



def merge_img(img_size, sub_images, padding_size,subimg_size):

	# Create an empty array for merging image

	padding_size = [padding_size[0]*2, padding_size[1]*2]
	merged_image = np.zeros([img_size[0]-2*padding_size[0],
							img_size[1]-2*padding_size[1],
							img_size[2]])

	for k in sub_images:



		key = k.split("_")

		#if int(key[1]) == padding_size[0]: grid_r = 0
		grid_r = int(key[1])*2 - padding_size[0]

		#if int(key[2]) == padding_size[1]: grid_c = 0
		grid_c = int(key[2])*2 - padding_size[1]

		print(k, grid_r, grid_c,  sub_images[k].shape)
		
		merged_image[grid_r:grid_r+subimg_size[0],
						grid_c:grid_c+subimg_size[1],
						:] = sub_images[k][padding_size[0]:padding_size[0]+subimg_size[0],
											padding_size[1]:padding_size[1]+subimg_size[1],
											:]

	return merged_image
'''
def evaluation(model_ticket, checkpoint_dir, eval_dataset = ["Set5", "Set14"], scale = [2,4]):

	eval_img = tf.placeholder(tf.float2, [None, None, None, 3], name="eval_imput")
	target = tf.placeholder(tf.float2, [None, None, None, 3], name="target")
	dropout = tf.placeholder(tf.float32, name='dropout')

	mz = model_zoo.model_zoo(eval_img, dropout, False, model_ticket)    
    pred = mz.build_model()

    with tf.Session() as tf:
    	saver.restore(sess, tf.train.latest_checkpoint(checkpoint_dir))
	
	return
'''
def load_dataset(root, dataset, scale):

	input_list = os.path.listdir(path)

	return input_list

def input_setup(inputs, targets, imgname, padding_size, subimg_size, scale):

	small_out = {}

	sout = split_img("input",inputs, padding_size, subimg_size)
	for k in sout:

		small_out[k] = cv2.resize(sout[k], (subimg_size[0]*scale+padding_size[0]*scale*2,subimg_size[1]*scale+padding_size[0]*scale*2))


	target_out = split_img("target",targets, padding_size, [subimg_size[0]*scale,subimg_size[1]*scale])
	
	ms = merge_img([512,512,3], small_out, padding_size,[subimg_size[0]*scale,subimg_size[1]*scale])
	#ts = merge_img([512,512,3], target_out, padding_size, [subimg_size[0]*scale,subimg_size[1]*scale])

	cv2.imwrite("small.jpg", ms)
	#cv2.imwrite("taget.jpg", ts)



img_path = '../Test/Set5/baby_GT.bmp'
tmp_img = cv2.imread(img_path)
small_img = cv2.resize(tmp_img,(256,256))


input_setup(small_img, tmp_img, "nn", [3,3], [30,30],2)

#sub = split_img("test",tmp_img, [3,3], [30,30])
#mout = merge_img(tmp_img.shape,sub, [3,3], [30,30])


"""
for k in sub:
	print(k, sub[k].shape)
	cv2.imshow("sub", sub[k])
	cv2.waitKey()
"""
#print(mout.dtype)
#cv2.imwrite("raw.jpg", mout)
#cv2.waitKey()