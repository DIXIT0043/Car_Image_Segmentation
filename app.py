import streamlit as st
import tensorflow as tf
import numpy as np
import cv2

def dice_coef(y_true, y_pred, smooth=1):
    intersection = tf.reduce_sum(y_true * y_pred, axis=[1,2,3])
    union = tf.reduce_sum(y_true, axis=[1,2,3]) + tf.reduce_sum(y_pred, axis=[1,2,3])
    return tf.reduce_mean((2. * intersection + smooth) / (union + smooth), axis=0)

def dice_loss(in_gt, in_pred):
    return 1-dice_coef(in_gt, in_pred)

# Model Architecture
base_model = tf.keras.applications.MobileNetV2(input_shape=[256, 256, 3], include_top=False)

layer_names = [
    'block_1_expand_relu',
    'block_3_expand_relu',
    'block_6_expand_relu',
    'block_13_expand_relu',
    'block_16_project',
]
base_model_outputs = [base_model.get_layer(name).output for name in layer_names]
down_stack = tf.keras.Model(inputs=base_model.input, outputs=base_model_outputs)
down_stack.trainable = False

def upsample(filters, size, norm_type='batchnorm', apply_dropout=False):
    initializer = tf.random_normal_initializer(0., 0.02)
    
    result = tf.keras.Sequential()
    result.add(
        tf.keras.layers.Conv2DTranspose(filters, size, strides=2,
                                    padding='same',
                                    kernel_initializer=initializer,
                                    use_bias=False))

    if norm_type.lower() == 'batchnorm':
        result.add(tf.keras.layers.BatchNormalization())
    elif norm_type.lower() == 'instancenorm':
        result.add(tf.keras.layers.LayerNormalization())

    if apply_dropout:
        result.add(tf.keras.layers.Dropout(0.5))

    result.add(tf.keras.layers.ReLU())
    return result

up_stack = [
    upsample(512, 3),
    upsample(256, 3),
    upsample(128, 3),
    upsample(64, 3),
]

def unet_model(output_channels):
    inputs = tf.keras.layers.Input(shape=[256, 256, 3])
    
    skips = down_stack(inputs)
    x = skips[-1]
    skips = reversed(skips[:-1])

    for up, skip in zip(up_stack, skips):
        x = up(x)
        concat = tf.keras.layers.Concatenate()
        x = concat([x, skip])

    last = tf.keras.layers.Conv2DTranspose(
        output_channels, 3, strides=2, activation='sigmoid',
        padding='same')

    x = last(x)
    return tf.keras.Model(inputs=inputs, outputs=x)

# Create and compile model
model = unet_model(output_channels=1)
model.compile(optimizer='adam',
             loss=dice_loss,
             metrics=[dice_coef, 'binary_accuracy'])

# Load the trained weights
model.load_weights('unet_model_weights.weights.h5')

# After model.load_weights()
print("Model summary:")
model.summary()

# Test if weights are loaded
test_weights = model.get_weights()
print("Number of weight arrays:", len(test_weights))
print("First layer weights shape:", test_weights[0].shape)

def preprocess_image(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (256, 256))
    image = image.astype('float32') / 255.0
    return np.expand_dims(image, axis=0)

def predict_mask(image):
    # Store original dimensions
    original_height, original_width = image.shape[:2]
    
    processed_image = preprocess_image(image)
    prediction = model.predict(processed_image)
    
    # Convert prediction to binary mask (0 or 1)
    mask = (prediction[0] > 0.5).astype(np.uint8)
    
    # Create visualization array with purple background
    visualization = np.zeros((*mask.shape[0:2], 3), dtype=np.uint8)
    visualization[..., 0] = 59  # R channel
    visualization[..., 2] = 89  # B channel
    
    # Set yellow color for car pixels
    car_pixels = mask[..., 0] == 1
    visualization[car_pixels] = [255, 255, 0]  # Yellow color
    
    # Resize visualization to match original image dimensions
    visualization = cv2.resize(visualization, (original_width, original_height), 
                             interpolation=cv2.INTER_NEAREST)
    
    return visualization

# Streamlit UI
st.title("Car Image Segmentation")
st.write("Upload a car image to generate its segmentation mask")

# Create a sidebar for file upload
with st.sidebar:
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    
    # Create two columns for displaying images
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Original Image")
        st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), use_column_width=True)
    
    with col2:
        st.subheader("Predicted Mask")
        mask = predict_mask(image)
        st.image(mask, use_column_width=True)
