
# CRNN网络模型
#################### 自己写的代码这里开始 ################### 
class CRNN(object):
    def __init__(self,
                 num_classes,  # 类别数量
                 label_dict):  # 标签字典
        self.outputs = None  # 输出
        self.label_dict = label_dict  # 标签字典
        self.num_classes = num_classes  # 类别数量
 
    def name(self):
        return "crnn"
 
    def conv_bn_pool(self, input, group,  # 输入,组
                     out_ch,  # 输入通道数
                     act="relu",  # 激活函数
                     param=None, bias=None,  # 参数、权重初始值
                     param_0=None, is_test=False,
                     pooling=True,  # 是否执行池化
                     use_cudnn=False):  # 是否对cuda加速
        tmp = input
 
        for i in six.moves.xrange(group):
            # for i in range(group): # 也可以
            # 卷积层
            tmp = fluid.layers.conv2d(
                input=tmp,  # 输入
                num_filters=out_ch[i],  # num_filters (int) - 滤波器（卷积核）的个数。和输出图像通道相同。
                filter_size=3,##滤波器大小
                padding=1,##填充大小
                param_attr=param if param_0 is None else param_0,##指定权重参数属性的对象
                act=None,
                use_cudnn=use_cudnn)
            # 批量归一化
            tmp = fluid.layers.batch_norm(
                input=tmp,  # 前面卷基层输出作为输入
                act=act,  # 激活函数
                param_attr=param,  # 参数初始值
                bias_attr=bias,  # 偏置初始值
                is_test=is_test)  # 测试模型
        # 根据传入的参数决定是否做池化操作
        if pooling:
            tmp = fluid.layers.pool2d(
                input=tmp,  # 前一层的输出作为输入
                pool_size=2,  # 池化区域
                pool_type="max",  # 池化类型
                pool_stride=2,  # 步长
                use_cudnn=use_cudnn,
                ceil_mode=True)  # 输出高度计算公式
        return tmp
 
    # 包含4个卷积层操作
    def ocr_convs(self, input,
                  regularizer=None,  # 正则化
                  gradient_clip=None,  # 梯度裁剪，防止梯度过大
                  is_test=False, use_cudnn=False):
 ###创建一个参数属性对象，用户可设置参数的名称、初始化方式、学习率、正则化规则、是否需要训练、梯度裁剪方式、是否做模型平均等属性。
        b = fluid.ParamAttr(
            regularizer=regularizer,
            gradient_clip=gradient_clip,
            initializer=fluid.initializer.Normal(0.0, 0.0))
        w0 = fluid.ParamAttr(
            regularizer=regularizer,
            gradient_clip=gradient_clip,
            initializer=fluid.initializer.Normal(0.0, 0.0005))
        w1 = fluid.ParamAttr(
            regularizer=regularizer,
            gradient_clip=gradient_clip,
            initializer=fluid.initializer.Normal(0.0, 0.01))
 
        tmp = input
 
        # 第一组卷积池化
        tmp = self.conv_bn_pool(tmp,
                                2, [16, 16],  # 组数量及卷积核数量
                                param=w1,
                                bias=b,
                                param_0=w0,
                                is_test=is_test,
                                use_cudnn=use_cudnn)
        # 第二组卷积池化
        tmp = self.conv_bn_pool(tmp,
                                2, [32, 32],  # 组数量及卷积核数量
                                param=w1,
                                bias=b,
                                is_test=is_test,
                                use_cudnn=use_cudnn)
        # 第三组卷积池化
        tmp = self.conv_bn_pool(tmp,
                                2, [64, 64],  # 组数量及卷积核数量
                                param=w1,
                                bias=b,
                                is_test=is_test,
                                use_cudnn=use_cudnn)
        # 第四组卷积池化
        tmp = self.conv_bn_pool(tmp,
                                2, [128, 128],  # 组数量及卷积核数量
                                param=w1,
                                bias=b,
                                is_test=is_test,
                                pooling=False,  # 不做池化
                                use_cudnn=use_cudnn)
        return tmp
 
    # 组网
    def net(self, images,
            rnn_hidden_size=200,  # 隐藏层输出值数量
            regularizer=None,  # 正则化
            gradient_clip=None,  # 梯度裁剪，防止梯度过大
            is_test=False,
            use_cudnn=True):
        # 卷积池化
        conv_features = self.ocr_convs(
            images,
            regularizer=regularizer,
            gradient_clip=gradient_clip,
            is_test=is_test,
            use_cudnn=use_cudnn)
        # 将特征图转为序列
        sliced_feature = fluid.layers.im2sequence(
            input=conv_features,  # 卷积得到的特征图作为输入
            stride=[1, 1],
            # 卷积核大小(高度等于原高度，宽度1)
            filter_size=[conv_features.shape[2], 1])
        # 两个全连接层
        para_attr = fluid.ParamAttr(
            regularizer=regularizer,  # 正则化
            gradient_clip=gradient_clip,
            initializer=fluid.initializer.Normal(0.0, 0.02))
        bias_attr = fluid.ParamAttr(
            regularizer=regularizer,  # 正则化
            gradient_clip=gradient_clip,
            initializer=fluid.initializer.Normal(0.0, 0.02))
        bias_attr_nobias = fluid.ParamAttr(
            regularizer=regularizer,  # 正则化
            gradient_clip=gradient_clip,
            initializer=fluid.initializer.Normal(0.0, 0.02))
 
        fc_1 = fluid.layers.fc(
            input=sliced_feature,  # 序列化处理的特征图
            size=rnn_hidden_size * 3,
            param_attr=para_attr,
            bias_attr=bias_attr_nobias)
        fc_2 = fluid.layers.fc(
            input=sliced_feature,  # 序列化处理的特征图
            size=rnn_hidden_size * 3,
            param_attr=para_attr,
            bias_attr=bias_attr_nobias)
 
        # 双向GRU(门控循环单元，LSTM变种, LSTM是RNN变种)
        gru_foward = fluid.layers.dynamic_gru(
            input=fc_1,
            size=rnn_hidden_size,
            param_attr=para_attr,
            bias_attr=bias_attr,
            candidate_activation="relu")
        gru_backward = fluid.layers.dynamic_gru(
            input=fc_2,
            size=rnn_hidden_size,
            is_reverse=True,  # 反向循环神经网络
            param_attr=para_attr,
            bias_attr=bias_attr,
            candidate_activation="relu")
        # 输出层
        w_attr = fluid.ParamAttr(
            regularizer=regularizer,
            gradient_clip=gradient_clip,
            initializer=fluid.initializer.Normal(0.0, 0.02))
        b_attr = fluid.ParamAttr(
            regularizer=regularizer,
            gradient_clip=gradient_clip,
            initializer=fluid.initializer.Normal(0.0, 0.0))
 
        fc_out = fluid.layers.fc(
            input=[gru_foward, gru_backward],  # 双向RNN输出作为输入
            size=self.num_classes + 1,  # 输出类别
            param_attr=w_attr,
            bias_attr=b_attr)
 
        self.outputs = fc_out
        return fc_out
 
    def get_infer(self):
        # 将CRNN网络输出交给CTC层转录(纠错、去重)
        return fluid.layers.ctc_greedy_decoder(
            input=self.outputs, # 输入为CRNN网络输出
            blank=self.num_classes)

