import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import tensorflow.compat.v1 as tf

tf.compat.v1.disable_eager_execution()
LEARNING_RATE = 1e-4
TRAINING_ITERATIONS = 2500
DROPOUT = 0.5
BATCH_SIZE = 50
VALIDATION_SIZE = 2000
IMAGE_TO_DISPLAY = 10
data = pd.read_csv(filepath_or_buffer='DigitalRecognizerData/train.csv')
# 切片得到像素，不包括标签
images = data.iloc[:, 1:].values
images = images.astype(np.float64)

# 归一化
images = np.multiply(images, 1.0 / 255.0)
image_size = images.shape[1]
# 获取图片长和宽
image_width = image_height = np.ceil(np.sqrt(image_size)).astype(np.uint8)


# 输出图片
def display(img):
    one_image = img.reshape(image_width, image_height)
    plt.axis('off')
    plt.imshow(one_image, cmap=cm.binary)
    plt.show()


# display(images[IMAGE_TO_DISPLAY])
# 获取标签
labels_flat = data.iloc[:, 0].values.ravel()
labels_count = np.unique(labels_flat).shape[0]


# 转化为one-hot vectors
# 0 => [1 0 0 0 0 0 0 0 0 0 ]
def dense_to_one_hot(labels_dense, num_classes):
    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes))
    # flat就是相当于变成一维数组,再读取
    # ravel将多维数组转化为一维，返回一个连续的平整的数组。
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
    return labels_one_hot


labels = dense_to_one_hot(labels_flat, labels_count)
labels = labels.astype(np.uint8)
# print('labels[{0}]=>{1}'.format(IMAGE_TO_DISPLAY, labels[IMAGE_TO_DISPLAY]))
# 把训练数据划分出一个验证集，来证明模型是否有泛化能力。
validation_images = images[:VALIDATION_SIZE]
validation_labels = labels[:VALIDATION_SIZE]

train_images = images[VALIDATION_SIZE:]
train_labels = labels[VALIDATION_SIZE:]


# 权重初始化，并把他们放到图中，作为变量。
# 变量不同于张量，张量是图中的数据流图的边，是可以流动的。
# 变量在图中有固定的位置。
def weight_variable(shape):
    # tf.truncated_normal(shape, mean, stddev)
    # 释义：截断的产生正态分布的随机数，即随机数与均值的差值若大于两倍的标准差，则重新生成。

    # shape，生成张量的维度
    # mean，均值
    # stddev，标准差
    initial = tf.truncated_normal(shape, stddev=0.1)
    # 函数用于创建变量(Variable), 变量是一个特殊的张量()，其可以是任意的形状和类型的张量。
    # tf.Variable.init(initial_value, trainable=True, collections=None, validate_shape=True, name=None)
    #
    # 参数名称
    # 参数类型
    # 含义
    # initial_value
    # 所有可以转换为Tensor的类型
    # 变量的初始值
    # trainable
    # bool
    # 如果为True，会把它加入到GraphKeys.TRAINABLE_VARIABLES，才能对它使用Optimizer
    # collections
    # list
    # 指定该图变量的类型、默认为[GraphKeys.GLOBAL_VARIABLES]
    # validate_shape
    # bool
    # 如果为False，则不进行类型和维度检查
    # name
    # string
    # 变量的名称，如果没有指定则系统会自动分配一个唯一的值

    return tf.Variable(initial)


def bias_variable(shape):
    # tf.constant(value, shape, dtype=None, name=None)
    # 释义：生成常量
    #
    # value，值
    # shape，数据形状
    # dtype，数据类型
    # name，名称
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


# convolution 卷积层
def conv2d(x, W):
    # tf.nn.conv2d(input, filter, strides, padding, use_cudnn_on_gpu=None, data_format=None, name=None)
    #
    # 参数：
    # input: 输入的要做卷积的图片，要求为一个张量，
    # shape为[batch, in_height, in_width, in_channel]，
    # 其中batch为图片的数量，
    # in_height为图片高度，
    # in_width为图片宽度，
    # in_channel为图片的通道数，
    # 灰度图该值为1，彩色图为3。
    # filter： 卷积核，要求也是一个张量，shape为[filter_height, filter_width, in_channel, out_channels]，
    # 其中filter_height为卷积核高度，
    # filter_width为卷积核宽度，
    # in_channel是图像通道数 ，和input的n_channel要保持一致，
    # out_channel是卷积核数量。
    # strides： 卷积时在图像每一维的步长，这是一个一维的向量，[1, strides, strides, 1]，第一位和最后一位固定必须是1
    # padding： string类型，值为“SAME” 和 “VALID”，表示的是卷积的形式，是否考虑边界。
    # "SAME" 是考虑边界，不足的时候用0去填充周围，"VALID" 则不考虑
    # use_cudnn_on_gpu： bool类型，是否使用cudnn加速，默认为true

    # 选择0补充层来防止数据宽高减小。步长选择为1
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


# 池化层
# 池化层的作用：1.保持不变性，比如两张图有平移，旋转，尺度时，
# 通过取最大值（ maxpooling ）可以使标签相同但是图像略微不同的图具有相同的特征。
# 2.保留主要特征同时减少参数和计算量，防止过拟合，提高模型的泛化能力。
def max_pool_2x2(x):
    # tf.nn.max_pool(value, ksize, strides, padding, name=None)
    # 参数是四个，和卷积很类似：
    # 第一个参数value：需要池化的输入，一般池化层接在卷积层后面，所以输入通常是feature map，
    # 依然是[batch, height, width, channels]这样的shape
    # 第二个参数ksize：池化窗口的大小，取一个四维向量，
    # 一般是[1, height, width, 1]，因为我们不想在batch和channels上做池化，所以这两个维度设为了1
    # 第三个参数strides：和卷积类似，窗口在每一个维度上滑动的步长，一般也是[1, stride, stride, 1]
    # 第四个参数padding：和卷积类似，可以取 'VALID' 或者 'SAME'
    # 返回一个Tensor，类型不变，shape仍然是[batch, height, width, channels]这种形式

    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


# 占位符
# images
x = tf.placeholder('float', shape=[None, image_size])
# labels
y_ = tf.placeholder('float', shape=[None, labels_count])
# first convolutional layer
# 第一层卷积层，用了5*5的过滤器，并且卷积层想要预估出32个特征值，
# 我们可以得出权重的shape[5, 5, 1, 32].这里第三个数字是输入的通道要与上一层的输出通道值相一致。
W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])

# 函数接口：
#
# tf.reshape(
#     tensor, shape, name=None
# )
# 参数
# tensor	Tensor张量
# shape	Tensor张量，用于定义输出张量的shape，组成元素类型为 int32或int64.
# name	可选参数，用于定义操作名称.
# 返回
# A Tensor. 输出张量和输入张量的元素类型相同。
# tf.reshape函数用于对输入tensor进行维度调整，但是这种调整方式并不会修改内部元素的数量以及元素之间的顺序，
# 换句话说，reshape函数不能实现类似于矩阵转置的操作。比如，对于矩阵[[1,2,3],[4,5,6]],如果使用reshape,将维度变为[3,2],
# 其输出结果为：[[1,2],[3,4],[5,6]], 元素之间的顺序并没有改变：1之后还是2，如果是矩阵转置操作，1之后应该为4。
# 其内部实现可以理解为: tf.reshape(a, shape) -> tf.reshape(a, [-1]) -> tf.reshape(a, shape)
# 现将输入tensor，flatten铺平，然后再进行维度调整（元素的顺序没有变化）
# tf.reshape不会更改张量中元素的顺序或总数，因此可以重用基础数据缓冲区。这使得它快速运行，而与要运行的张量有多大无关。
# 如果需要修改张量的维度来实现对元素重新排序，需要使用tf.transpose。

image = tf.reshape(x, [-1, image_width, image_height, 1])

# relu函数处理
# 因tf.nn.relu()函数的目的是，将输入小于0的值幅值为0，输入大于0的值不变。
h_conv1 = tf.nn.relu(conv2d(image, W_conv1) + b_conv1)

h_pool1 = max_pool_2x2(h_conv1)

# second convolutional layer
# 第二层卷积层想要预估出64个特征值，得到权重的shape[5, 5, 32, 64]。
# 因为经过池化层，图像已经变成了14*14的大小，
# 第二层卷几层想要得到更一般的特征，过滤器覆盖了图像的更多空间，所以我们调整选择使用更多的特征。
W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

# densely connected layer
# 一个1024个神经元的全连接层
W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])

h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

# dropout
# 防止过拟合，加入了随机失活。
keep_prob = tf.placeholder('float')
# tf.nn.dropout函数说明

# tf.nn.dropout(x,keep_prob,noise_shape=None,seed=None,name=None)
# 参数说明：
# x：指输入，输入tensor

# keep_prob: float类型，每个元素被保留下来的概率，设置神经元被选中的概率, 在初始化时keep_prob是一个占位符,
# keep_prob = tf.placeholder(tf.float32) 。
# tensorflow在run时设置keep_prob具体的值，例如keep_prob: 0.5

# noise_shape: 一个1维的int32张量，代表了随机产生“保留 / 丢弃”标志的shape。

# seed: 整形变量，随机数种子。

# name：指定该操作的名字

# dropout必须设置概率keep_prob，并且keep_prob也是一个占位符，跟输入是一样的
# keep_prob = tf.placeholder(tf.float32)
# train的时候才是dropout起作用的时候，test的时候不应该让dropout起作用

h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
# readout layer for deep net
# 最后使用softmax来得到各分类的预测分数。
W_fc2 = weight_variable([1024, labels_count])
b_fc2 = bias_variable([labels_count])
# tf.nn.softmax(logits,axis=None,name=None,dim=None)
#
# logits：一个非空的Tensor。必须是下列类型之一：half， float32，float64
# axis：将在其上执行维度softmax。默认值为-1，表示最后一个维度
# name：操作的名称(可选)
# dim：axis的已弃用的别名
# 输入: 全连接层（往往是模型的最后一层）的值
# 输出: 归一化的值，含义是属于该位置的概率;

# tf.matmul 矩阵乘法
y = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

# cost function
# 定义损失函数，使用了交叉熵。
# 还需要定义一个优化方法，选择了Adam算法。
# tf.reduce_sum 求和
cross_entropy = -tf.reduce_sum(y_ * tf.log(y))

# optimisation function
# tf.train.AdamOptimizer Adam优化算法
train_step = tf.train.AdamOptimizer(LEARNING_RATE).minimize(cross_entropy)

# evaluation
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))

# tf.cast bool格式转换为float
# tf.reduce_mean概率值相加
accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))
# prediction function
# 预测概率
predict = tf.argmax(y, 1)

epochs_completed = 0
index_in_epoch = 0
num_examples = train_images.shape[0]


# 批次训练
# serve data by batches
def next_batch(batch_size):
    global train_images
    global train_labels
    global index_in_epoch
    global epochs_completed

    start = index_in_epoch
    index_in_epoch += batch_size

    # when all trainig data have been already used, it is reorder randomly
    if index_in_epoch > num_examples:
        # finished epoch
        epochs_completed += 1
        # shuffle the data
        perm = np.arange(num_examples)
        np.random.shuffle(perm)
        train_images = train_images[perm]
        train_labels = train_labels[perm]
        # start next epoch
        start = 0
        index_in_epoch = batch_size
        assert batch_size <= num_examples
    end = index_in_epoch
    return train_images[start:end], train_labels[start:end]


# 变量初始化
init = tf.global_variables_initializer()
sess = tf.InteractiveSession()

sess.run(init)
# 检查其中的几个点的精度。
# visualisation variables
train_accuracies = []
validation_accuracies = []
x_range = []

display_step = 1

# 因为会迭代很多次，所以应用了next_batch。在检查其中几个点精度的同时，也训练好了整个网络
for i in range(TRAINING_ITERATIONS):

    # get new batch
    batch_xs, batch_ys = next_batch(BATCH_SIZE)

    # check progress on every 1st,2nd,...,10th,20th,...,100th... step
    if i % display_step == 0 or (i + 1) == TRAINING_ITERATIONS:

        train_accuracy = accuracy.eval(feed_dict={x: batch_xs,
                                                  y_: batch_ys,
                                                  keep_prob: 1.0})
        if VALIDATION_SIZE:
            validation_accuracy = accuracy.eval(feed_dict={x: validation_images[0:BATCH_SIZE],
                                                           y_: validation_labels[0:BATCH_SIZE],
                                                           keep_prob: 1.0})
            print('training_accuracy / validation_accuracy => %.2f / %.2f for step %d' % (
                train_accuracy, validation_accuracy, i))

            validation_accuracies.append(validation_accuracy)

        else:
            print('training_accuracy => %.4f for step %d' % (train_accuracy, i))
        train_accuracies.append(train_accuracy)
        x_range.append(i)

        # increase display_step
        if i % (display_step * 10) == 0 and i:
            display_step *= 10
    # train on batch
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys, keep_prob: DROPOUT})
# 用matplotlib画出train和validation的准确度。
if VALIDATION_SIZE:
    validation_accuracy = accuracy.eval(feed_dict={x: validation_images,
                                                   y_: validation_labels,
                                                   keep_prob: 1.0})
    plt.plot(x_range, train_accuracies, '-b', label='Training')
    plt.plot(x_range, validation_accuracies, '-g', label='Validation')
    plt.legend(loc='lower right', frameon=False)
    plt.ylim(ymax=1.1, ymin=0.7)
    plt.ylabel('accuracy')
    plt.xlabel('step')
    plt.show()
# 对测试数据进行预测
# read test data from CSV file
test_images = pd.read_csv('DigitalRecognizerData/test.csv').values
test_images = test_images.astype('float')

# convert from [0:255] => [0.0:1.0]
test_images = np.multiply(test_images, 1.0 / 255.0)

# print('test_images({0[0]},{0[1]})'.format(test_images.shape))

# predict test set
# predicted_lables = predict.eval(feed_dict={x: test_images, keep_prob: 1.0})

# using batches is more resource efficient
predicted_lables = np.zeros(test_images.shape[0])
for i in range(0, test_images.shape[0] // BATCH_SIZE):
    predicted_lables[i * BATCH_SIZE: (i + 1) * BATCH_SIZE] = predict.eval(
        feed_dict={x: test_images[i * BATCH_SIZE: (i + 1) * BATCH_SIZE],
                   keep_prob: 1.0})

# print('predicted_lables({0})'.format(len(predicted_lables)))

# output test image and prediction
# display(test_images[IMAGE_TO_DISPLAY])
# print('predicted_lables[{0}] => {1}'.format(IMAGE_TO_DISPLAY, predicted_lables[IMAGE_TO_DISPLAY]))

# save results
np.savetxt('test_labels.csv',
           np.c_[range(1, len(test_images) + 1), predicted_lables],
           delimiter=',',
           header='ImageId,Label',
           comments='',
           fmt='%d')

sess.close()
