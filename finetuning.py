"""
QUANTIZATION
Conversion from higher memory format to a lower memory format.
Value of weights is stored as (for example) 32 bits (full precision/single precision), we can convert it to int8 (or FP16 - half precision, or any other) and then use the model
cost and resources go down
very useful for inference (faster and cheaper)
useful for less powerful devices like cell phone and other edge devices
we can fine tune quantized models, but there is loss of info and hence loss of accuracy
 

CALIBRATION - squeezing higher range to lower range
Symmetric Quantization 
e.g. batch norm - all weights are 0 centred
symmetric unsigned int8 quantization - we have floating point numbers [0.0, 1000.0] (for larger model, maybe 32 bits single precision floating point) - 1 sign, 7 exponent, 23 fraction (mantissa)
unsigned int8 would be [0, 255].
we can use min max scaler
scale_factor = x_max - x_min / q_max - q_min [1000 - 0 / 255 - 0]
x_new = round(x_old/scale_factor)

Asymmetric Quantization
symmetric uint8  [-20.0, 1000.0] -> values can be left or right skewed 
if we convert it to [0, 255]
scale facto = 4.0
x_min_new = -20/4 = -5 (it is signed) -> this number is called zero point
x_new = round(x_old/scale_factor) + zero_point


Post Training quantization (PTQ) - already have pretrained model -> apply calibration to weights and save it as quantized model
    - loss of info
Quantization aware training (QAT) - take the trained model, perform calibration, perform fine tuning on new training data.
    - trained on new precision values
"""