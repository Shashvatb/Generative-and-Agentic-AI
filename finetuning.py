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


We get weights of a LLM trained on huge data. fine tune on new data on all params. major challenge -> expensive
then we can perform domain specific fine tuning (for medical data etc) 
or we can perform specific task fine tuning (for Q&A chatbot, document retrieval etc)

LoRA (Low Rank Adaptation of LLM) - instead of updating all weight, it will track the changes in weights based on fine tuning (same size matrix) and based on that it will update weights
the tracked weights are stored as 2 vectors which is created with matrix decomp (the vectors can be cross product-ed to reform the matrix)
there is loss in precision but the resources needed are much lower.
W0 + W_delta = W0 + BxA
the parameters increase linearly with the rank increase (instead of polynomial)
high ranks are used - if we want more complex model. if we need a simple LLM, we can use smaller ranks

QLora (Quantized LoRA) - All params stored in W_delta, we store it in lower precision (saves even more resources)
"""