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

    def call(self, inputs):
        outputs = self.model(inputs)

        return outputs

class resnet101(tf.keras.Model):                                                               
    def __init__(self, pretrained=False):
        super(resnet101, self).__init__()
        if pretrained: 
            self.model = tf.keras.applications.ResNet101(include_top=False, weights='imagenet')
        else:
            self.model = tf.keras.applications.ResNet101(include_top=False, weights=None)

    def call(self, inputs):
        outputs = self.model(inputs)

        return outputs

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
    print("Output feature map size is:", output_features.shape)
    print(model.layers[0].weights)
    print(model.layers[0].trainable)
    print(output_features)