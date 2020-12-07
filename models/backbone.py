'''
This code is to build backbone model by pretrained ResNet from ImageNet.
'''
import tensorflow as tf

__all__ = ['resnet50','resnet101']

class resnet50(tf.keras.Model):                                                               
    def __init__(self, pretrained=False):
        super(resnet50, self).__init__()
        if pretrained: 
            self.model = tf.keras.applications.ResNet50(include_top=False, weights='imagenet')
        else:
            self.model = tf.keras.applications.ResNet50(include_top=False, weights=None)
        # extract layer 1, 2, 3, 4
        layer1_name = 'conv2_block3_out'
        layer2_name = 'conv3_block4_out'
        layer3_name = 'conv4_block6_out'
        layer4_name = 'conv5_block3_out'
        self.model1 = tf.keras.Model(inputs=self.model.input, outputs=self.model.get_layer(layer1_name).output)
        self.model2 = tf.keras.Model(inputs=self.model.input, outputs=self.model.get_layer(layer2_name).output)
        self.model3 = tf.keras.Model(inputs=self.model.input, outputs=self.model.get_layer(layer3_name).output)
        self.model4 = tf.keras.Model(inputs=self.model.input, outputs=self.model.get_layer(layer4_name).output)

    def call(self, x):
        f = []
        x1 = self.model1(x)
        x2 = self.model2(x)
        x3 = self.model3(x)
        x4 = self.model4(x)
        f.append(x1)
        f.append(x2)
        f.append(x3)
        f.append(x4)

        return f

class resnet101(tf.keras.Model):
    def __init__(self, pretrained=False):
        super(resnet101, self).__init__()
        if pretrained: 
            self.model = tf.keras.applications.ResNet101(include_top=False, weights='imagenet')
        else:
            self.model = tf.keras.applications.ResNet101(include_top=False, weights=None)
        # extract layer 1, 2, 3, 4
        layer1_name = 'conv2_block3_out'
        layer2_name = 'conv3_block4_out'
        layer3_name = 'conv4_block23_out'
        layer4_name = 'conv5_block3_out'
        self.model1 = tf.keras.Model(inputs=self.model.input, outputs=self.model.get_layer(layer1_name).output)
        self.model2 = tf.keras.Model(inputs=self.model.input, outputs=self.model.get_layer(layer2_name).output)
        self.model3 = tf.keras.Model(inputs=self.model.input, outputs=self.model.get_layer(layer3_name).output)
        self.model4 = tf.keras.Model(inputs=self.model.input, outputs=self.model.get_layer(layer4_name).output)

    def call(self, x):
        f = []
        x1 = self.model1(x)
        x2 = self.model2(x)
        x3 = self.model3(x)
        x4 = self.model4(x)
        f.append(x1)
        f.append(x2)
        f.append(x3)
        f.append(x4)

        return f

# unit test
if __name__ == '__main__':
    batch_size = 32
    Height = 48
    Width = 160
    Channel = 3
    tf.random.set_seed(0)
    input_images = tf.random.uniform(shape=[batch_size,Height,Width,Channel])
    model = resnet50(pretrained=False)
    output_features = model(input_images)

    print("Input size is:",input_images.shape)
    print("Output feature map size is:", len(output_features))
    for layer in range(len(output_features)):
        print("Shape of layer {} is {}".format(layer, output_features[layer].shape))