<h1 align="center">VGG16 & VGG19 Deep Learning Projects</h1>

<p align="center">
  Binary and Multi-Class Image Classification using Transfer Learning
</p>

<hr>

<h2>📌 Project Overview</h2>

<p>
This repository contains two Deep Learning projects implemented using TensorFlow and Keras:
</p>

<ul>
  <li><b>VGG16 Binary Classification</b> – Cat vs Dog Classification</li>
  <li><b>VGG19 Multi-Class Classification</b> – COVID-19, Normal, Viral Pneumonia Classification</li>
</ul>

<p>
Both projects use <b>Transfer Learning</b> with pretrained CNN architectures.
</p>

<hr>

<h2>📚 What is Transfer Learning?</h2>

<p>
Transfer Learning means using a pretrained model that was already trained on a huge dataset like <b>ImageNet</b> and adapting it for a new task.
</p>

<p>
Instead of training a CNN from scratch:
</p>

<ul>
  <li>We use pretrained weights</li>
  <li>The model already learned edges, textures, shapes, and patterns</li>
  <li>Reduces training time</li>
  <li>Improves accuracy</li>
  <li>Requires less dataset</li>
</ul>

<hr>

<h2>🧠 About VGG Networks</h2>

<p>
VGG models were developed by the <b>Visual Geometry Group (VGG)</b> at the University of Oxford.
</p>

<p>
Main idea of VGG architecture:
</p>

<ul>
  <li>Use very small convolution filters (3×3)</li>
  <li>Increase depth of network</li>
  <li>Improve feature extraction capability</li>
</ul>

<hr>

<h1>🔥 VGG16 Architecture</h1>

<h2>Why Called VGG16?</h2>

<ul>
  <li>13 Convolution Layers</li>
  <li>3 Fully Connected Layers</li>
</ul>

<p><b>Total = 16 Learnable Layers</b></p>

<hr>

<h2>📷 VGG16 Architecture Diagram</h2>

<p align="center">
  <img src="https://miro.medium.com/v2/resize:fit:1400/1*HsLUJQy2fGHcN-o8SoKBQg.png" width="900">
</p>

<hr>

<h2>🏗️ VGG16 Architecture Structure</h2>

<pre>
Input Image (224 × 224 × 3)

Block 1
→ Conv3-64
→ Conv3-64
→ MaxPooling

Block 2
→ Conv3-128
→ Conv3-128
→ MaxPooling

Block 3
→ Conv3-256
→ Conv3-256
→ Conv3-256
→ MaxPooling

Block 4
→ Conv3-512
→ Conv3-512
→ Conv3-512
→ MaxPooling

Block 5
→ Conv3-512
→ Conv3-512
→ Conv3-512
→ MaxPooling

Flatten

FC-4096
FC-4096
FC-1000
</pre>

<hr>

<h1>🧾 VGG16 Code Explanation</h1>

<h2>1️⃣ Import Libraries</h2>

<pre>
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
</pre>

<h3>Explanation</h3>

<ul>
  <li><b>tensorflow</b> → Deep Learning framework</li>
  <li><b>keras</b> → High-level API for neural networks</li>
  <li><b>ImageDataGenerator</b> → Used for preprocessing and augmentation</li>
</ul>

<hr>

<h2>2️⃣ Dataset Paths</h2>

<pre>
training_data_path = "/content/train"
validation_data_path = "/content/val"
</pre>

<h3>Dataset Structure</h3>

<pre>
train/
   cats/
   dogs/

val/
   cats/
   dogs/
</pre>

<hr>

<h2>3️⃣ Data Augmentation</h2>

<pre>
train_data_preprocessing = ImageDataGenerator(
    1 / 255,
    rotation_range=0.2,
    shear_range=0.2,
    horizontal_flip=True
)
</pre>

<h3>Explanation</h3>

<table border="1" cellpadding="10">
<tr>
<th>Parameter</th>
<th>Purpose</th>
</tr>

<tr>
<td>1/255</td>
<td>Normalize pixel values</td>
</tr>

<tr>
<td>rotation_range</td>
<td>Rotate images</td>
</tr>

<tr>
<td>shear_range</td>
<td>Shearing transformation</td>
</tr>

<tr>
<td>horizontal_flip</td>
<td>Flip image horizontally</td>
</tr>

</table>

<hr>

<h2>4️⃣ Load Dataset</h2>

<pre>
final_train_data = train_data_preprocessing.flow_from_directory(
    training_data_path,
    target_size=(224, 224),
    class_mode='binary',
    classes=labels,
    batch_size=20
)
</pre>

<h3>Explanation</h3>

<ul>
  <li><b>target_size=(224,224)</b> → Required VGG input size</li>
  <li><b>class_mode='binary'</b> → Used for 2 classes</li>
  <li><b>batch_size=20</b> → Train 20 images at a time</li>
</ul>

<hr>

<h2>5️⃣ Import VGG16</h2>

<pre>
from tensorflow.keras.applications import VGG16
</pre>

<hr>

<h2>6️⃣ Load Pretrained Model</h2>

<pre>
vgg16_obj = VGG16(
    input_shape=(224,224,3),
    weights="imagenet",
    include_top=False
)
</pre>

<h3>Explanation</h3>

<ul>
  <li><b>weights="imagenet"</b> → Load pretrained weights</li>
  <li><b>include_top=False</b> → Remove original classifier layer</li>
</ul>

<hr>

<h2>7️⃣ Freeze Layers</h2>

<pre>
for i in vgg16_obj.layers:
    i.trainable = False
</pre>

<h3>Why Freeze Layers?</h3>

<ul>
  <li>Prevents destroying learned features</li>
  <li>Reduces training time</li>
  <li>Avoids overfitting</li>
</ul>

<hr>

<h2>8️⃣ Add Custom ANN Layers</h2>

<pre>
one_d_values = Flatten()(vgg16_obj.output)

h1_out = Dense(
    units=128,
    kernel_initializer='he_uniform',
    activation='relu'
)(one_d_values)

h2_out = Dense(
    units=64,
    kernel_initializer='he_uniform',
    activation='relu'
)(h1_out)

h3_out = Dense(
    units=32,
    kernel_initializer='he_uniform',
    activation='relu'
)(h2_out)

output = Dense(
    units=1,
    kernel_initializer='glorot_uniform',
    activation='sigmoid'
)(h3_out)
</pre>

<hr>

<h2>📌 Flatten Layer</h2>

<pre>
Flatten()
</pre>

<p>
Converts multidimensional feature maps into 1D vector.
</p>

<pre>
7 × 7 × 512
↓
25088
</pre>

<hr>

<h2>📌 ReLU Activation Function</h2>

<pre>
activation='relu'
</pre>

<p><b>Formula:</b></p>

<pre>
f(x) = max(0, x)
</pre>

<h3>Advantages</h3>

<ul>
  <li>Removes negative values</li>
  <li>Faster training</li>
  <li>Avoids vanishing gradient</li>
</ul>

<hr>

<h2>📌 Sigmoid Activation Function</h2>

<pre>
activation='sigmoid'
</pre>

<p><b>Formula:</b></p>

<pre>
σ(x) = 1 / (1 + e^-x)
</pre>

<p>
Output Range:
</p>

<pre>
0 → 1
</pre>

<hr>

<h2>9️⃣ Create Final Model</h2>

<pre>
model = Model(
    inputs=vgg16_obj.input,
    outputs=output
)
</pre>

<hr>

<h2>🔟 Compile Model</h2>

<pre>
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)
</pre>

<hr>

<h2>📌 Binary Crossentropy</h2>

<pre>
L = -( y log(y^) + (1-y) log(1-y^) )
</pre>

<hr>

<h2>1️⃣1️⃣ Train Model</h2>

<pre>
model.fit(
    final_train_data,
    validation_data=final_val_data,
    epochs=5
)
</pre>

<hr>

<h1>🔥 VGG19 Architecture</h1>

<h2>Why Called VGG19?</h2>

<ul>
  <li>16 Convolution Layers</li>
  <li>3 Fully Connected Layers</li>
</ul>

<p><b>Total = 19 Learnable Layers</b></p>

<hr>

<h2>📷 VGG19 Architecture Diagram</h2>

<p align="center">
  <img src="https://miro.medium.com/v2/resize:fit:1400/1*1Tsw26K-rtxqqBP8v6_6gQ.png" width="900">
</p>

<hr>

<h2>🏗️ VGG19 Architecture Structure</h2>

<pre>
Input Image (224 × 224 × 3)

Block 1
→ Conv64
→ Conv64
→ MaxPooling

Block 2
→ Conv128
→ Conv128
→ MaxPooling

Block 3
→ Conv256
→ Conv256
→ Conv256
→ Conv256
→ MaxPooling

Block 4
→ Conv512
→ Conv512
→ Conv512
→ Conv512
→ MaxPooling

Block 5
→ Conv512
→ Conv512
→ Conv512
→ Conv512
→ MaxPooling

Flatten

FC4096
FC4096
FC1000
</pre>

<hr>

<h1>🧾 VGG19 Code Explanation</h1>

<h2>Dataset Classes</h2>

<pre>
labels = ['Covid','Normal','Viral Pneumonia']
</pre>

<p>
This is a <b>Multi-Class Classification</b> problem.
</p>

<hr>

<h2>Load Dataset</h2>

<pre>
class_mode='categorical'
</pre>

<hr>

<h2>Import VGG19</h2>

<pre>
from tensorflow.keras.applications import VGG19
</pre>

<hr>

<h2>Load Pretrained Model</h2>

<pre>
vgg_obj = VGG19(
    input_shape=(224,224,3),
    weights="imagenet",
    include_top=False
)
</pre>

<hr>

<h2>Freeze Layers</h2>

<pre>
for i in vgg_obj.layers:
    i.trainable = False
</pre>

<hr>

<h2>Add Custom Output Layer</h2>

<pre>
output = Dense(
    units=3,
    kernel_initializer='glorot_uniform',
    activation='softmax'
)(h3_out)
</pre>

<hr>

<h2>📌 Softmax Activation Function</h2>

<pre>
Softmax(zᵢ) = e^zᵢ / Σ e^z
</pre>

<p>
Output probabilities always sum to 1.
</p>

<pre>
Covid → 0.80
Normal → 0.10
Pneumonia → 0.10
</pre>

<hr>

<h2>📌 Categorical Crossentropy</h2>

<pre>
L = - Σ y log(y^)
</pre>

<hr>

<h2>🖼️ Prediction Function</h2>

<pre>
def pred_image(path_of_image):
</pre>

<hr>

<h2>Read Image</h2>

<pre>
image = cv2.imread(path_of_image,1)
</pre>

<hr>

<h2>Resize Image</h2>

<pre>
resized_image = cv2.resize(image , (224,224))
</pre>

<hr>

<h2>Normalize Pixels</h2>

<pre>
scaled_pixel_values = resized_image / 255
</pre>

<hr>

<h2>Add Batch Dimension</h2>

<pre>
input_image = np.expand_dims(
    scaled_pixel_values,
    axis=0
)
</pre>

<hr>

<h2>Prediction</h2>

<pre>
result = model.predict(input_image)
</pre>

<hr>

<h2>Final Output</h2>

<pre>
print(labels[np.argmax(result)])
</pre>

<hr>

<h1>📊 Difference Between VGG16 and VGG19</h1>

<table border="1" cellpadding="10">

<tr>
<th>Feature</th>
<th>VGG16</th>
<th>VGG19</th>
</tr>

<tr>
<td>Total Layers</td>
<td>16</td>
<td>19</td>
</tr>

<tr>
<td>Conv Layers</td>
<td>13</td>
<td>16</td>
</tr>

<tr>
<td>Parameters</td>
<td>Less</td>
<td>More</td>
</tr>

<tr>
<td>Speed</td>
<td>Faster</td>
<td>Slower</td>
</tr>

<tr>
<td>Accuracy</td>
<td>Slightly Lower</td>
<td>Slightly Higher</td>
</tr>

</table>

<hr>

<h1>✅ Advantages of VGG Models</h1>

<ul>
  <li>High Accuracy</li>
  <li>Strong Feature Extraction</li>
  <li>Easy Architecture</li>
  <li>Transfer Learning Support</li>
  <li>Pretrained Weights Available</li>
</ul>

<hr>

<h1>❌ Limitations</h1>

<ul>
  <li>Large Model Size</li>
  <li>High Memory Usage</li>
  <li>Slow Training</li>
  <li>Computationally Expensive</li>
</ul>

<hr>

<h1>🎯 Conclusion</h1>

<p>
In these projects:
</p>

<ul>
  <li>VGG16 was used for Binary Classification</li>
  <li>VGG19 was used for Multi-Class Medical Image Classification</li>
  <li>Transfer Learning improved accuracy and reduced training time</li>
  <li>Custom ANN layers were added on top of pretrained CNN models</li>
</ul>

<p>
These projects demonstrate the power of Deep Learning and Transfer Learning for image classification tasks.
</p>

<hr>

<h2 align="center">⭐ If you like this project, give it a star on GitHub ⭐</h2>
