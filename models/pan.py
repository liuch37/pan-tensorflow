'''
This code is the integrted model for PAN.
'''

import tensorflow as tf

from .backbone import resnet50
from .fpem import FPEM, Conv_BN_ReLU
from .ffm import FFM
from .head import PA_Head

__all__ = ['PAN']

class PAN(tf.keras.Model):
    def __init__(self, pretrained, hidden_dim, num_classes):
        super(PAN, self).__init__()
        self.backbone = resnet50(pretrained=pretrained)
        self.reduce_layer1 = Conv_BN_ReLU(128)
        self.reduce_layer2 = Conv_BN_ReLU(128)
        self.reduce_layer3 = Conv_BN_ReLU(128)
        self.reduce_layer4 = Conv_BN_ReLU(128)

        self.fpem1 = FPEM(128)
        self.fpem2 = FPEM(128)

        self.ffm = FFM()

        self.det_head = PA_Head(hidden_dim, num_classes)

    def call(self, imgs):
        # backbone
        f = self.backbone(imgs)

        # reduce channel
        f1 = self.reduce_layer1(f[0])
        f2 = self.reduce_layer2(f[1])
        f3 = self.reduce_layer3(f[2])
        f4 = self.reduce_layer4(f[3])

        # FPEM
        f1_1, f2_1, f3_1, f4_1 = self.fpem1(f1, f2, f3, f4)
        f1_2, f2_2, f3_2, f4_2 = self.fpem2(f1_1, f2_1, f3_1, f4_1)

        # FFM
        f = self.ffm(f1_1, f2_1, f3_1, f4_1, f1_2, f2_2, f3_2, f4_2)

        # detection
        det_out = self.det_head(f)

        return det_out

# unit testing
if __name__ == '__main__':

    batch_size = 32
    Height = 32
    Width = 64
    hidden_dim = 128
    Channel = 3

    input_images = tf.random.uniform(shape=[batch_size,Height,Width,Channel])
    
    model = PAN(pretrained=False, hidden_dim=hidden_dim, num_classes=6)

    det_out = model(input_images)
    print("PAN output size is:", det_out.shape)