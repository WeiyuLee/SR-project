import sys
import tensorflow as tf
sys.path.append('./utility')
from utility import model_zoo as mz

train_set = []
test_set = []

def load_dataset(data_dir, isRandom = True):
    img_files = os.listdir(data_dir)
    test_size = int(len(img_files)*0.2)
    
    if isRandom == True: 
        test_indices = random.sample(range(len(img_files)),test_size)
    else:
        test_indices = list(range(test_size))
     
    for i in range(len(img_files)):
	#img = scipy.misc.imread(data_dir+img_files[i])
        if i in test_indices:
            test_set.append(data_dir+"/"+img_files[i])
        else:
            train_set.append(data_dir+"/"+img_files[i])

    test_set.append(data_dir+"/butterfly_GT.bmp")
    return

def get_batch(batch_size,original_size,shrunk_size):
	x =[]
	y =[]
	img_indices = random.sample(range(len(train_set)),batch_size)
	for i in range(len(img_indices)):
		index = img_indices[i]
		img = scipy.misc.imread(train_set[index])
		img = crop_center(img,original_size,original_size)
		x_img = scipy.misc.imresize(img,(shrunk_size,shrunk_size))
		x.append(x_img)
		y.append(img)
	return x,y

def crop_center(img,cropx,cropy):
	y,x,_ = img.shape
	startx = x//2-(cropx//2)
	starty = y//2-(cropy//2)    
	return img[starty:starty+cropy,startx:startx+cropx]




output_channels = 3
scale = 2
img_size = 50

x = tf.placeholder(tf.float32,[None,img_size,img_size,output_channels])
y = tf.placeholder(tf.float32,[None,img_size*scale,img_size*scale,output_channels])
dropout = tf.placeholder(tf.float32, name='dropout')
    

mean_x = tf.reduce_mean(x)
image_input = x- mean_x
mean_y = tf.reduce_mean(y)
image_target = y- mean_y

edsr = mz.model_zoo(x, dropout, True, "EDSR_v1")
netout = edsr.build_model()
l1_loss = tf.reduce_mean(tf.losses.absolute_difference(image_target,netout))
optimizer = tf.train.AdamOptimizer()
#This is the train operation for our objective
train_op = optimizer.minimize(l1_loss)    

mse = tf.reduce_mean(tf.squared_difference(image_target,output))    
PSNR = tf.constant(255**2,dtype=tf.float32)/mse

tf.summary.scalar("loss",l1_loss)
tf.summary.scalar("psnr",PSNR)
tf.summary.image("input_image",image_input+mean_x)
tf.summary.image("target_image",image_target+mean_y)
tf.summary.image("output_image",netout+mean_x)

merged = tf.summary.merge_all()

data_path = '/home/ubuntu/dataset/SR_set/General-100'
save_dir = '/home/ubuntu/model/model/SR_project/dirtytest'

if os.path.exists(save_dir):
            shutil.rmtree(save_dir)
os.mkdir(save_dir)
os.mkdir(os.path.join(save_dir, 'model'))
os.mkdir(os.path.join(save_dir, 'log'))

self.sess = tf.Session()
self.saver = tf.train.Saver(3)


load_dataset(data_path)
batch_size = 32
original_size = img_size*scale
shrunk_size = img_size

for i in range(100000):

	train_x, train_y = get_batch(batch_size, original_size, shrunk_size)




