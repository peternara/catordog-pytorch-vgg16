import matplotlib.pyplot as plt
import numpy as np
import os
from IPython.display import display, Image, HTML
import cv2
import tensorflow as tf


#最终放到模型中训练的图像的大小为96*96
IMAGE_SIZE = 96
#图像的厚度，rgb为3层，灰度图像为1层
CHANNELS = 3
#像素的深度，最大为255，在预处理中会用到
pixel_depth = 255.0

def dataset_init():
	
	#训练数据与测试数据所在文件夹
	TRAIN_DIR = '/home/wumingjie/Dataset/catvsdog2/train_data/'
	#TEST_DIR = '/home/wumingjie/Dataset/catvsdog2/test_data/'
	TEST_DIR = '/home/wumingjie/Dataset/catvsdog/train_data/'

	#输出文件目录，保存当前的配置
	OUTFILE = './smalltest.npsave.bin'
	#训练文件中狗的数量
	TRAINING_AND_VALIDATION_SIZE_DOGS = 12500
	#训练文件中猫的数量
	TRAINING_AND_VALIDATION_SIZE_CATS = 12500
	#训练文件中所有图片的数量
	TRAINING_AND_VALIDATION_SIZE_ALL = 25000
	#训练数据的大小，也就是所有图片的数量
	TRAINING_SIZE = 25000

	#测试文件中狗的数量
	TEST_SIZE_DOGS = 1000
	#测试文件中猫的数量
	TEST_SIZE_CATS = 1000
	#测试数据的大小，也就是测试所有图片的数量
	TEST_SIZE_ALL = 2000

	#从训练文件夹中获得训练图片的路径列表
	train_images = [TRAIN_DIR + i for i in os.listdir(TRAIN_DIR)]
	#从训练文件夹中获得训练数据的狗的图片的路径列表
	train_dogs = [TRAIN_DIR + i for i in os.listdir(TRAIN_DIR) if 'dog' in i]
	#从训练文件夹中获得训练数据中猫的图片的路径列表
	train_cats = [TRAIN_DIR + i for i in os.listdir(TRAIN_DIR) if 'cat' in i]

	#从测试文件夹中获的测试图片的路径列表
	test_images = [TEST_DIR + i for i in os.listdir(TEST_DIR)]
	#从测试文件中获得训练数据的狗的图片的路径列表
	test_dogs = [TEST_DIR + i for i in os.listdir(TEST_DIR) if 'dog' in i]
	#从测试文件中获的训练数据的猫的图片的路径列表
	test_cats = [TEST_DIR + i for i in os.listdir(TEST_DIR) if 'cat' in i]

	#将训练数据中狗和猫的图片数组拼接起来
	train_images = train_dogs[:TRAINING_AND_VALIDATION_SIZE_DOGS] + train_cats[:TRAINING_AND_VALIDATION_SIZE_CATS]
	#将构建训练数据标签数组，因为前面的全是狗，后面的全是猫
	train_labels = np.array((['dogs'] * TRAINING_AND_VALIDATION_SIZE_DOGS) + ['cats'] * TRAINING_AND_VALIDATION_SIZE_CATS)

	#将测试数据中狗和猫的图片数组拼接起来
	test_images = test_dogs[:TEST_SIZE_DOGS] + test_cats[:TEST_SIZE_CATS]
	#构建测试数据标签数组，同样是狗在前，猫在后面
	test_labels = np.array((['dogs'] * TEST_SIZE_DOGS + ['cats'] * TEST_SIZE_CATS))
	#正则化训练数据和测试数据
	train_normalized, train_image_name = prep_data(train_images)
	test_normalized, test_image_name = prep_data(test_images)
	#输出训练数据和测试数据的维度
	print("Train shape: {}".format(train_normalized.shape))
	print("Test shape: {}".format(test_normalized.shape))
	#初始化随机种子
	np.random.seed(133)
	#将训练数据和训练标签随机
	train_dataset_rand, train_labels_rand = randomize(train_normalized, train_labels)
	#将测试数据和标签随机
	test_dataset, test_labels = randomize(test_normalized, test_labels)
	#获得随机化后的训练数据
	train_dataset = train_dataset_rand[:TRAINING_SIZE, :, :, :]
	train_labels = train_labels_rand[:TRAINING_SIZE]
	#获的随机化后的测试数据
	test_dataset = train_dataset_rand[:TEST_SIZE_ALL, :, :, :]
	test_labels = train_labels_rand[:TEST_SIZE_ALL]

	#输出训练和测试数据的维度
	print('Training', train_dataset.shape, train_labels.shape)
	print('Test', test_dataset.shape, test_labels.shape)
	
	#显示训练数据正则化后的第一个图片
	plt.imshow (train_normalized[0,:,:,:], interpolation='nearest')
	plt.figure ()
	#显示测试数据正则化的第一个图片
	plt.imshow (test_normalized[0, :, :, :], interpolation='nearest')
	plt.figure()
	#更改图片的大小变成规定的大小函数

	#调动上述函数，对数据转换成模型要求的格式，即标签为[0,1]类型，
	#数据类型为[-1,imagesize, imageszie, channels]
	train_dataset, train_labels = reformat(train_dataset, train_labels)
	test_dataset, test_labels = reformat(test_dataset, test_labels)
	print ('Training set', train_dataset.shape, train_labels.shape)
	print ('Test set', test_dataset.shape, test_labels.shape)

	return train_dataset, train_labels, test_dataset, test_labels

def read_image(file_path):
	img = cv2.imread(file_path, cv2.IMREAD_COLOR)
	img2 = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_CUBIC)
	img3 = cv2.copyMakeBorder(img2, 0, IMAGE_SIZE - img2.shape[0], 0, IMAGE_SIZE - img2.shape[1], cv2.BORDER_CONSTANT, 0)

	return img3[:,:,::-1]

#正则化图片数据函数
def prep_data(images):
	count = len(images)
	data = np.ndarray((count, IMAGE_SIZE, IMAGE_SIZE, CHANNELS), dtype=np.float32)
	image_name = []
	for i, image_file in enumerate(images):
		#调用上面的read_image函数对图片的大小进行修改
		img = read_image(image_file)
		image_name.append(image_file)
		#将图片数据转换成浮点型数组
		image_data = np.array(img, dtype=np.float32)
		#讲图片的每层都进行正则化
		image_data[:, :, 0] = (image_data[:, :, 0].astype(float) - pixel_depth / 2) / pixel_depth
		image_data[:, :, 1] = (image_data[:, :, 1].astype(float) - pixel_depth / 2) / pixel_depth
		image_data[:, :, 2] = (image_data[:, :, 2].astype(float) - pixel_depth / 2) / pixel_depth

		data[i] = image_data
		if i % 1000 == 0:
			print("Processed {} of {}".format(i, count))

	return data, image_name

#随机函数，对标签和图片按照同样的随机顺序，进行随机
def randomize(dataset, labels):
	#获的随机顺序
	permutation = np.random.permutation(labels.shape[0])
	#将数据按照顺序随机
	shuffled_dataset = dataset[permutation, :, :, :]
	#将标签按照顺序随机
	shuffled_labels = labels[permutation]
	return shuffled_dataset, shuffled_labels

#将训练数据转化成为模型需要的数据类型
def reformat(dataset, labels):
	image_size = IMAGE_SIZE
	num_labels = 2
	num_channels = 3 # rgb
	dataset = dataset.reshape((-1, image_size, image_size, num_channels)).astype(np.float32)
	#狗是0，猫是1
	labels = (labels=='cats').astype(np.float32);
	labels = (np.arange(num_labels) == labels[:,None]).astype(np.float32)
	return dataset, labels

#权值初始化计算函数
def Weight_variable(shape):
	initial = tf.truncated_normal(shape, stddev=0.1)
	return tf.Variable(initial)

#偏置初始化函数
def bias_variable(shape):
	initial = tf.constant(0.1, shape=shape)
	return tf.Variable(initial)

#卷积计算函数
def con2d(inputs, weight):
	return tf.nn.conv2d(inputs, weight, strides=[1,1,1,1], padding='SAME')

#池化层计算函数
def max_pooling_2x2(inputs):
	return tf.nn.max_pool(inputs, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')



def load_test2():
	TEST_DIR2 = '/home/wumingjie/Dataset/catvsdog/test_data/'
	#TEST_DIR2 = '/home/wumingjie/Dataset/catvsdog2/test_data/'
	test_images2 = [TEST_DIR2 + i for i in os.listdir(TEST_DIR2)]
	test_normalized2, image_name = prep_data(test_images2)
	print("Test shape2: {}".format(test_normalized2.shape))
	test_dataset2 = test_normalized2
	
	return test_dataset2, image_name

#准确度计算函数
def computer_accuracy(v_xs, v_ys, sess):
	#计算得到预测值
	y_re = sess.run(prediction, feed_dict={xs: v_xs, keep_prob: 1})
	return (np.sum(np.argmax(y_re, 1) == np.argmax(v_ys, 1)) / 2000.)

def computer_accuracy2(v_xs, sess, image_name):
	#从测试文件夹中获的测试图片的路径列表
	
	#计算得到预测值
	y_re = sess.run(prediction, feed_dict={xs: v_xs, keep_prob: 1})
	#print (y_re)
	i = 0
	f = open("result.txt", "wt")
	for num in y_re :
		if num[0] > num[1]:
			f.write('Dog' + ' ' + image_name[i] + '\n')
		else :
			f.write('Cat' + ' ' + image_name[i] + '\n')
		i = i + 1
	f.close()

#定义变量
xs = tf.placeholder(tf.float32, [None, 96, 96, 3])
ys = tf.placeholder(tf.float32, [None, 2])
#定义保存的概率
x_image = tf.reshape(xs, [-1, 96, 96, 3])

#第一层卷积层
W_conv1 = Weight_variable([5, 5, 3, 32])
b_conv1 = bias_variable([32])
h_conv1 = tf.nn.relu(con2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pooling_2x2(h_conv1)

#第二层卷积层
W_conv2 = Weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(con2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pooling_2x2(h_conv2)

#第三层卷积层
W_conv3 = Weight_variable([5, 5, 64, 128])
b_conv3 = bias_variable([128])
h_conv3 = tf.nn.relu(con2d(h_pool2, W_conv3) + b_conv3)
h_pool3 = max_pooling_2x2(h_conv3)

print("The shape of the h_pool3 is {}".format(h_pool3.shape))
print("The shape of the h_conv3 is {}".format(h_conv3.shape))

#第四层卷积层
W_conv4 = Weight_variable([5, 5, 128, 256])
b_conv4 = bias_variable([256])
h_conv4 = tf.nn.relu(con2d(h_pool3, W_conv4) + b_conv4)
h_pool4 = max_pooling_2x2(h_conv4)
print("The shape of the h_pool4 is {}".format(h_pool4.shape))
print("The shape of the h_conv4 is {}".format(h_conv4.shape))

#全连接层
w_f1 = Weight_variable([6*6*256, 2048])
b_f1 = bias_variable([2048])
h_pool4_flat = tf.reshape(h_pool4, [-1, 6*6*256])
h_f1 = tf.nn.relu(tf.matmul(h_pool4_flat, w_f1) + b_f1)
	
#Dropout
keep_prob = tf.placeholder(tf.float32)
h_f1_drop = tf.nn.dropout(h_f1, keep_prob)

#输出层
w_f2 = Weight_variable([2048, 2])
b_f2 = bias_variable([2])
y = tf.matmul(h_f1_drop, w_f2) + b_f2 
prediction = tf.nn.softmax(tf.matmul(h_f1_drop, w_f2) + b_f2)#y_conv
y_conv = tf.nn.softmax(tf.matmul(h_f1_drop, w_f2) + b_f2)

#误差计算
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = y,labels=ys))
#cross_entropy = -tf.reduce_sum(ys*tf.log(y_conv))

#采用梯度下降法进行训练
train_step = tf.train.RMSPropOptimizer(0.0001).minimize(cross_entropy)
#train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
#train_step = tf.train.GradientDescentOptimizer(0.0001).minimize(cross_entropy)#效果71%
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(ys,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))


def train():

	#prediction, cross_entropy, train_step = network_init()
	#sess = tf.InteractiveSession()
	sess = tf.Session()
	sess.run(tf.global_variables_initializer())

	NUM = 1800
	train_dataset, train_labels, test_dataset, test_labels = dataset_init()
	for i in range(NUM):
		#每次训练500个，计算的是每次训练完的最后标志位
		offset = (i * 100) % (train_labels.shape[0] - 100)
		batch_xs = train_dataset[offset:(offset + 100), :, :, :]
		batch_ys = train_labels[offset:(offset + 100), :]
		a ,b = sess.run((train_step, cross_entropy),feed_dict={xs: batch_xs, ys: batch_ys, keep_prob: 1})
		print(b)
		#每五十步计算一次准确率
		if i % 50 == 0:
			val = computer_accuracy(test_dataset, test_labels, sess)
			print("Train step %d:" %i ,val)
		if i == NUM-1 :
			test_dataset2,image_name = load_test2()
			computer_accuracy2(test_dataset2, sess, image_name)

	saver = tf.train.Saver()
	saver.save(sess, './my_test_model')
	sess.close()

def test():
	sess = tf.Session()
	sess.run(tf.global_variables_initializer())
	saver = tf.train.Saver()
	saver.restore(sess, './my_test_model')
	test_dataset2, image_name = load_test2()
	computer_accuracy2(test_dataset2, sess, image_name)
	sess.close()

if __name__ == '__main__':
	train()
	test()



