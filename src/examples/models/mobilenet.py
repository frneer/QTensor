from tensorflow import keras
model = keras.applications.MobileNet(
    include_top=True,
    weights="imagenet",
    classes=1000,
    classifier_activation="softmax",
)

from quantizers.flex_quantizer import FlexQuantizer
from quantizers.uniform_quantizer import UniformQuantizer

#simple_folded_merged_qconfig = {
#        #"conv2d": {
#        #    "weights": {"kernel": FlexQuantizer(bits=4, n_levels=10 , signed=True)},
#        #    "activations": {"activation": UniformQuantizer(bits=4, signed=False)},
#        #},
#        #"dense_2": {
#        #    "weights": {"kernel": FlexQuantizer(bits=4, n_levels=10 , signed=True)},
#        #    "activations": {"activation": UniformQuantizer(bits=4, signed=False)},
#        #},
#    }

simple_folded_merged_qconfig = {
        "conv1"      : { "weights": {"kernel"          : UniformQuantizer(bits=10, signed=True)}, "activations": {"activation": UniformQuantizer(bits=10, signed=False)}, },
        "conv_dw_1"  : { "weights": {"depthwise_kernel": UniformQuantizer(bits=10, signed=True)}, "activations": {"activation": UniformQuantizer(bits=10, signed=False)}, },
        "conv_pw_1"  : { "weights": {"kernel"          : UniformQuantizer(bits=10, signed=True)}, "activations": {"activation": UniformQuantizer(bits=10, signed=False)}, },
        "conv_dw_2"  : { "weights": {"depthwise_kernel": UniformQuantizer(bits=10, signed=True)}, "activations": {"activation": UniformQuantizer(bits=10, signed=False)}, },
        "conv_pw_2"  : { "weights": {"kernel"          : UniformQuantizer(bits=10, signed=True)}, "activations": {"activation": UniformQuantizer(bits=10, signed=False)}, },
        "conv_dw_3"  : { "weights": {"depthwise_kernel": UniformQuantizer(bits=10, signed=True)}, "activations": {"activation": UniformQuantizer(bits=10, signed=False)}, },
        "conv_pw_3"  : { "weights": {"kernel"          : UniformQuantizer(bits=10, signed=True)}, "activations": {"activation": UniformQuantizer(bits=10, signed=False)}, },
        "conv_dw_4"  : { "weights": {"depthwise_kernel": UniformQuantizer(bits=10, signed=True)}, "activations": {"activation": UniformQuantizer(bits=10, signed=False)}, },
        "conv_pw_4"  : { "weights": {"kernel"          : UniformQuantizer(bits=10, signed=True)}, "activations": {"activation": UniformQuantizer(bits=10, signed=False)}, },
        "conv_dw_5"  : { "weights": {"depthwise_kernel": UniformQuantizer(bits=10, signed=True)}, "activations": {"activation": UniformQuantizer(bits=10, signed=False)}, },
        "conv_pw_5"  : { "weights": {"kernel"          : UniformQuantizer(bits=10, signed=True)}, "activations": {"activation": UniformQuantizer(bits=10, signed=False)}, },
        "conv_dw_6"  : { "weights": {"depthwise_kernel": UniformQuantizer(bits=10, signed=True)}, "activations": {"activation": UniformQuantizer(bits=10, signed=False)}, },
        "conv_pw_6"  : { "weights": {"kernel"          : UniformQuantizer(bits=10, signed=True)}, "activations": {"activation": UniformQuantizer(bits=10, signed=False)}, },
        "conv_dw_7"  : { "weights": {"depthwise_kernel": UniformQuantizer(bits=10, signed=True)}, "activations": {"activation": UniformQuantizer(bits=10, signed=False)}, },
        "conv_pw_7"  : { "weights": {"kernel"          : UniformQuantizer(bits=10, signed=True)}, "activations": {"activation": UniformQuantizer(bits=10, signed=False)}, },
        "conv_dw_8"  : { "weights": {"depthwise_kernel": UniformQuantizer(bits=10, signed=True)}, "activations": {"activation": UniformQuantizer(bits=10, signed=False)}, },
        "conv_pw_8"  : { "weights": {"kernel"          : UniformQuantizer(bits=10, signed=True)}, "activations": {"activation": UniformQuantizer(bits=10, signed=False)}, },
        "conv_dw_9"  : { "weights": {"depthwise_kernel": UniformQuantizer(bits=10, signed=True)}, "activations": {"activation": UniformQuantizer(bits=10, signed=False)}, },
        "conv_pw_9"  : { "weights": {"kernel"          : UniformQuantizer(bits=10, signed=True)}, "activations": {"activation": UniformQuantizer(bits=10, signed=False)}, },
        "conv_dw_10" : { "weights": {"depthwise_kernel": UniformQuantizer(bits=10, signed=True)}, "activations": {"activation": UniformQuantizer(bits=10, signed=False)}, },
        "conv_pw_10" : { "weights": {"kernel"          : UniformQuantizer(bits=10, signed=True)}, "activations": {"activation": UniformQuantizer(bits=10, signed=False)}, },
        "conv_dw_11" : { "weights": {"depthwise_kernel": UniformQuantizer(bits=10, signed=True)}, "activations": {"activation": UniformQuantizer(bits=10, signed=False)}, },
        "conv_pw_11" : { "weights": {"kernel"          : UniformQuantizer(bits=10, signed=True)}, "activations": {"activation": UniformQuantizer(bits=10, signed=False)}, },
        "conv_dw_12" : { "weights": {"depthwise_kernel": UniformQuantizer(bits=10, signed=True)}, "activations": {"activation": UniformQuantizer(bits=10, signed=False)}, },
        "conv_pw_12" : { "weights": {"kernel"          : UniformQuantizer(bits=10, signed=True)}, "activations": {"activation": UniformQuantizer(bits=10, signed=False)}, },
        "conv_dw_13" : { "weights": {"depthwise_kernel": UniformQuantizer(bits=10, signed=True)}, "activations": {"activation": UniformQuantizer(bits=10, signed=False)}, },
        "conv_pw_13" : { "weights": {"kernel"          : UniformQuantizer(bits=10, signed=True)}, "activations": {"activation": UniformQuantizer(bits=10, signed=False)}, },
        "conv_preds" : { "weights": {"kernel"          : UniformQuantizer(bits=10, signed=True)}, "activations": {"activation": UniformQuantizer(bits=10, signed=False)}, },
        #"conv_pad_2" : { "weights": {"kernel"          : UniformQuantizer(bits=10, signed=True)}, "activations": {"activation": UniformQuantizer(bits=10, signed=False)}, },
        #"conv_pad_4" : { "weights": {"kernel"          : UniformQuantizer(bits=10, signed=True)}, "activations": {"activation": UniformQuantizer(bits=10, signed=False)}, },
        #"conv_pad_6" : { "weights": {"kernel"          : UniformQuantizer(bits=10, signed=True)}, "activations": {"activation": UniformQuantizer(bits=10, signed=False)}, },
        #"conv_pad_12": { "weights": {"kernel"          : UniformQuantizer(bits=10, signed=True)}, "activations": {"activation": UniformQuantizer(bits=10, signed=False)}, },
        #"dropout"    : { "weights": {"kernel"          : FlexQuantizer(bits=4, n_levels=10 , signed=True)}, "activations": {"activation": UniformQuantizer(bits=4, signed=False)}, },
        #"reshape_2"  : { "weights": {"kernel"          : FlexQuantizer(bits=6, n_levels=10 , signed=True)}, "activations": {"activation": UniformQuantizer(bits=6, signed=False)}, },
        #"prediction" : { "weights": {"kernel"          : FlexQuantizer(bits=8, n_levels=20 , signed=True)}, "activations": {"activation": UniformQuantizer(bits=8, signed=False)}, },
        #"global_average_pooling2d" : { "weights": {"kernel"          : FlexQuantizer(bits=4, n_levels=10 , signed=True)}, "activations": {"activation": UniformQuantizer(bits=4, signed=False)}, },
        }
simple_folded_merged_qconfig_2 = {
        #"conv1"      : { "weights": {"kernel"          : FlexQuantizer(bits=8, n_levels=20 , signed=True)}, "activations": {"activation": UniformQuantizer(bits=8, signed=False)}, },
        "conv_dw_1"  : { "weights": {"depthwise_kernel": UniformQuantizer(bits=10, signed=True)}, "activations": {"activation": UniformQuantizer(bits=10, signed=False)}, },
        #"conv_pw_1"  : { "weights": {"kernel"          : FlexQuantizer(bits=6, n_levels=10 , signed=True)}, "activations": {"activation": UniformQuantizer(bits=6, signed=False)}, },
        #"conv_dw_2"  : { "weights": {"depthwise_kernel": FlexQuantizer(bits=6, n_levels=10 , signed=True)}, "activations": {"activation": UniformQuantizer(bits=6, signed=False)}, },
        #"conv_pw_2"  : { "weights": {"kernel"          : FlexQuantizer(bits=6, n_levels=10 , signed=True)}, "activations": {"activation": UniformQuantizer(bits=6, signed=False)}, },
        #"conv_dw_3"  : { "weights": {"depthwise_kernel": FlexQuantizer(bits=6, n_levels=10 , signed=True)}, "activations": {"activation": UniformQuantizer(bits=6, signed=False)}, },
        #"conv_pw_3"  : { "weights": {"kernel"          : FlexQuantizer(bits=6, n_levels=10 , signed=True)}, "activations": {"activation": UniformQuantizer(bits=6, signed=False)}, },
        #"conv_dw_4"  : { "weights": {"depthwise_kernel": FlexQuantizer(bits=6, n_levels=10 , signed=True)}, "activations": {"activation": UniformQuantizer(bits=6, signed=False)}, },
        #"conv_pw_4"  : { "weights": {"kernel"          : FlexQuantizer(bits=6, n_levels=10 , signed=True)}, "activations": {"activation": UniformQuantizer(bits=6, signed=False)}, },
        #"conv_dw_5"  : { "weights": {"depthwise_kernel": FlexQuantizer(bits=6, n_levels=10 , signed=True)}, "activations": {"activation": UniformQuantizer(bits=6, signed=False)}, },
        #"conv_pw_5"  : { "weights": {"kernel"          : FlexQuantizer(bits=6, n_levels=10 , signed=True)}, "activations": {"activation": UniformQuantizer(bits=6, signed=False)}, },
        #"conv_dw_6"  : { "weights": {"depthwise_kernel": FlexQuantizer(bits=6, n_levels=10 , signed=True)}, "activations": {"activation": UniformQuantizer(bits=6, signed=False)}, },
        #"conv_pw_6"  : { "weights": {"kernel"          : FlexQuantizer(bits=6, n_levels=10 , signed=True)}, "activations": {"activation": UniformQuantizer(bits=6, signed=False)}, },
        #"conv_dw_7"  : { "weights": {"depthwise_kernel": FlexQuantizer(bits=6, n_levels=10 , signed=True)}, "activations": {"activation": UniformQuantizer(bits=6, signed=False)}, },
        #"conv_pw_7"  : { "weights": {"kernel"          : FlexQuantizer(bits=6, n_levels=10 , signed=True)}, "activations": {"activation": UniformQuantizer(bits=6, signed=False)}, },
        #"conv_dw_8"  : { "weights": {"depthwise_kernel": FlexQuantizer(bits=6, n_levels=10 , signed=True)}, "activations": {"activation": UniformQuantizer(bits=6, signed=False)}, },
        #"conv_pw_8"  : { "weights": {"kernel"          : FlexQuantizer(bits=6, n_levels=10 , signed=True)}, "activations": {"activation": UniformQuantizer(bits=6, signed=False)}, },
        #"conv_dw_9"  : { "weights": {"depthwise_kernel": FlexQuantizer(bits=6, n_levels=10 , signed=True)}, "activations": {"activation": UniformQuantizer(bits=6, signed=False)}, },
        #"conv_pw_9"  : { "weights": {"kernel"          : FlexQuantizer(bits=6, n_levels=10 , signed=True)}, "activations": {"activation": UniformQuantizer(bits=6, signed=False)}, },
        #"conv_dw_10" : { "weights": {"depthwise_kernel": FlexQuantizer(bits=6, n_levels=10 , signed=True)}, "activations": {"activation": UniformQuantizer(bits=6, signed=False)}, },
        #"conv_pw_10" : { "weights": {"kernel"          : FlexQuantizer(bits=6, n_levels=10 , signed=True)}, "activations": {"activation": UniformQuantizer(bits=6, signed=False)}, },
        #"conv_dw_11" : { "weights": {"depthwise_kernel": FlexQuantizer(bits=6, n_levels=10 , signed=True)}, "activations": {"activation": UniformQuantizer(bits=6, signed=False)}, },
        #"conv_pw_11" : { "weights": {"kernel"          : FlexQuantizer(bits=6, n_levels=10 , signed=True)}, "activations": {"activation": UniformQuantizer(bits=6, signed=False)}, },
        #"conv_dw_12" : { "weights": {"depthwise_kernel": FlexQuantizer(bits=6, n_levels=10 , signed=True)}, "activations": {"activation": UniformQuantizer(bits=6, signed=False)}, },
        #"conv_pw_12" : { "weights": {"kernel"          : FlexQuantizer(bits=6, n_levels=10 , signed=True)}, "activations": {"activation": UniformQuantizer(bits=6, signed=False)}, },
        #"conv_dw_13" : { "weights": {"depthwise_kernel": FlexQuantizer(bits=6, n_levels=10 , signed=True)}, "activations": {"activation": UniformQuantizer(bits=6, signed=False)}, },
        #"conv_pw_13" : { "weights": {"kernel"          : FlexQuantizer(bits=6, n_levels=10 , signed=True)}, "activations": {"activation": UniformQuantizer(bits=6, signed=False)}, },
        #"conv_preds" : { "weights": {"kernel"          : FlexQuantizer(bits=8, n_levels=20 , signed=True)}, "activations": {"activation": UniformQuantizer(bits=6, signed=False)}, },
        #"conv_pad_2" : { "weights": {"kernel"          : FlexQuantizer(bits=6, n_levels=10 , signed=True)}, "activations": {"activation": UniformQuantizer(bits=6, signed=False)}, },
        #"conv_pad_4" : { "weights": {"kernel"          : FlexQuantizer(bits=6, n_levels=10 , signed=True)}, "activations": {"activation": UniformQuantizer(bits=6, signed=False)}, },
        #"conv_pad_6" : { "weights": {"kernel"          : FlexQuantizer(bits=6, n_levels=10 , signed=True)}, "activations": {"activation": UniformQuantizer(bits=6, signed=False)}, },
        #"conv_pad_12": { "weights": {"kernel"          : FlexQuantizer(bits=6, n_levels=10 , signed=True)}, "activations": {"activation": UniformQuantizer(bits=6, signed=False)}, },
        #"dropout"    : { "weights": {"kernel"          : FlexQuantizer(bits=4, n_levels=10 , signed=True)}, "activations": {"activation": UniformQuantizer(bits=4, signed=False)}, },
        #"reshape_2"  : { "weights": {"kernel"          : FlexQuantizer(bits=6, n_levels=10 , signed=True)}, "activations": {"activation": UniformQuantizer(bits=6, signed=False)}, },
        #"prediction" : { "weights": {"kernel"          : FlexQuantizer(bits=8, n_levels=20 , signed=True)}, "activations": {"activation": UniformQuantizer(bits=8, signed=False)}, },
        #"global_average_pooling2d" : { "weights": {"kernel"          : FlexQuantizer(bits=4, n_levels=10 , signed=True)}, "activations": {"activation": UniformQuantizer(bits=4, signed=False)}, },
        }

qconfigs = {
    "folded_merged_qconfig": simple_folded_merged_qconfig
}

