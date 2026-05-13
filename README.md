<h1 align="center">📘 Detailed Explanation of VGG16 and VGG19 Architectures</h1>

<p align="center">
Deep Learning | CNN | Transfer Learning | TensorFlow | Keras
</p>

<hr>

<h1>📌 Introduction to CNN</h1>

<p>
CNN (Convolutional Neural Network) is a Deep Learning architecture mainly used for:
</p>

<ul>
  <li>Image Classification</li>
  <li>Object Detection</li>
  <li>Face Recognition</li>
  <li>Medical Image Analysis</li>
</ul>

<p>
CNN automatically learns important image features such as:
</p>

<ul>
  <li>Edges</li>
  <li>Textures</li>
  <li>Shapes</li>
  <li>Patterns</li>
</ul>

<hr>

<h1>📌 What is Transfer Learning?</h1>

<p>
Transfer Learning means using a pretrained model that was already trained on a huge dataset like <b>ImageNet</b>.
</p>

<p>
Instead of training from scratch:
</p>

<ul>
  <li>We use pretrained weights</li>
  <li>Save training time</li>
  <li>Improve accuracy</li>
  <li>Require smaller datasets</li>
</ul>

<hr>

<h1>📌 About VGG Networks</h1>

<p>
VGG models were developed by the <b>Visual Geometry Group (VGG)</b> at the University of Oxford.
</p>

<p>
Main idea of VGG:
</p>

<ul>
  <li>Use small 3×3 filters</li>
  <li>Increase network depth</li>
  <li>Improve feature extraction</li>
</ul>

<hr>

<h1>🔥 VGG16 Architecture</h1>

<h2>📌 Why Called VGG16?</h2>

<ul>
  <li>13 Convolution Layers</li>
  <li>3 Fully Connected Layers</li>
</ul>

<p>
<b>Total = 16 Learnable Layers</b>
</p>

<hr>

<h2>📷 VGG16 Architecture Diagram</h2>

<p align="center">
<img src="https://miro.medium.com/v2/resize:fit:1400/1*HsLUJQy2fGHcN-o8SoKBQg.png" width="1000">
</p>

<hr>

<h2>🏗️ VGG16 Complete Architecture</h2>

<pre>
Input : 224 × 224 × 3 RGB Image

Block 1
--------------------------------
Conv3-64
Conv3-64
MaxPooling

Block 2
--------------------------------
Conv3-128
Conv3-128
MaxPooling

Block 3
--------------------------------
Conv3-256
Conv3-256
Conv3-256
MaxPooling

Block 4
--------------------------------
Conv3-512
Conv3-512
Conv3-512
MaxPooling

Block 5
--------------------------------
Conv3-512
Conv3-512
Conv3-512
MaxPooling

Flatten

Fully Connected Layer - 4096
Fully Connected Layer - 4096
Output Layer - 1000 Classes
</pre>

<hr>

<h1>📌 VGG16 Architecture Explanation</h1>

<h2>🔹 Input Layer</h2>

<pre>
Input Shape = (224, 224, 3)
</pre>

<ul>
  <li>224 → Image Width</li>
  <li>224 → Image Height</li>
  <li>3 → RGB Channels</li>
</ul>

<hr>

<h2>🔹 Convolution Layer</h2>

<p>
Convolution layer extracts important features from image.
</p>

<h3>Example Features:</h3>

<ul>
  <li>Edges</li>
  <li>Corners</li>
  <li>Textures</li>
  <li>Shapes</li>
</ul>

<h3>Code Example</h3>

<pre>
Conv2D(
    filters=64,
    kernel_size=(3,3),
    activation='relu'
)
</pre>

<hr>

<h2>🔹 Why 3×3 Filters?</h2>

<ul>
  <li>Less parameters</li>
  <li>More efficient</li>
  <li>Better feature extraction</li>
  <li>Improves deep learning capability</li>
</ul>

<hr>

<h2>🔹 ReLU Activation Function</h2>

<pre>
f(x) = max(0, x)
</pre>

<p>
ReLU removes negative values and speeds up training.
</p>

<h3>Advantages</h3>

<ul>
  <li>Fast training</li>
  <li>Avoids vanishing gradient problem</li>
  <li>Improves performance</li>
</ul>

<hr>

<h2>🔹 MaxPooling Layer</h2>

<p>
Pooling reduces image dimensions.
</p>

<h3>Code Example</h3>

<pre>
MaxPooling2D(pool_size=(2,2))
</pre>

<h3>Benefits</h3>

<ul>
  <li>Reduces computation</li>
  <li>Prevents overfitting</li>
  <li>Extracts dominant features</li>
</ul>

<hr>

<h2>🔹 Flatten Layer</h2>

<p>
Flatten converts multidimensional feature maps into 1D vector.
</p>

<pre>
7 × 7 × 512
↓
25088
</pre>

<hr>

<h2>🔹 Fully Connected Layers</h2>

<p>
Fully connected layers perform final classification.
</p>

<pre>
Dense(4096, activation='relu')
</pre>

<hr>

<h2>🔹 Output Layer</h2>

<p>
For binary classification:
</p>

<pre>
Dense(1, activation='sigmoid')
</pre>

<p>
Sigmoid output range:
</p>

<pre>
0 → 1
</pre>

<hr>

<h1>📌 VGG16 Transfer Learning Code</h1>

<h2>Import Libraries</h2>

<pre>
import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten
</pre>

<hr>

<h2>Load Pretrained VGG16 Model</h2>

<pre>
vgg16_obj = VGG16(
    input_shape=(224,224,3),
    weights='imagenet',
    include_top=False
)
</pre>

<h3>Explanation</h3>

<ul>
  <li><b>weights='imagenet'</b> → Load pretrained weights</li>
  <li><b>include_top=False</b> → Remove original classifier layer</li>
</ul>

<hr>

<h2>Freeze Layers</h2>

<pre>
for layer in vgg16_obj.layers:
    layer.trainable = False
</pre>

<p>
Freezing layers prevents modification of pretrained weights.
</p>

<hr>

<h2>Add Custom ANN Layers</h2>

<pre>
x = Flatten()(vgg16_obj.output)

x = Dense(128, activation='relu')(x)

x = Dense(64, activation='relu')(x)

output = Dense(1, activation='sigmoid')(x)
</pre>

<hr>

<h2>Create Final Model</h2>

<pre>
model = Model(
    inputs=vgg16_obj.input,
    outputs=output
)
</pre>

<hr>

<h2>Compile Model</h2>

<pre>
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)
</pre>

<hr>

<h2>Train Model</h2>

<pre>
model.fit(
    train_data,
    validation_data=val_data,
    epochs=5
)
</pre>

<hr>

<h1>🔥 VGG19 Architecture</h1>

<h2>📌 Why Called VGG19?</h2>

<ul>
  <li>16 Convolution Layers</li>
  <li>3 Fully Connected Layers</li>
</ul>

<p>
<b>Total = 19 Learnable Layers</b>
</p>

<hr>

<h2>📷 VGG19 Architecture Diagram</h2>

<p align="center">
<img src="https://miro.medium.com/v2/resize:fit:1400/1*1Tsw26K-rtxqqBP8v6_6gQ.png" width="1000">
</p>

<hr>

<h2>🏗️ VGG19 Complete Architecture</h2>

<pre>
Input : 224 × 224 × 3 RGB Image

Block 1
--------------------------------
Conv3-64
Conv3-64
MaxPooling

Block 2
--------------------------------
Conv3-128
Conv3-128
MaxPooling

Block 3
--------------------------------
Conv3-256
Conv3-256
Conv3-256
Conv3-256
MaxPooling

Block 4
--------------------------------
Conv3-512
Conv3-512
Conv3-512
Conv3-512
MaxPooling

Block 5
--------------------------------
Conv3-512
Conv3-512
Conv3-512
Conv3-512
MaxPooling

Flatten

FC-4096
FC-4096
FC-1000
</pre>

<hr>

<h1>📌 VGG19 Architecture Explanation</h1>

<p>
VGG19 is deeper than VGG16.
</p>

<h3>Advantages of More Layers</h3>

<ul>
  <li>Better feature extraction</li>
  <li>Improved learning capability</li>
  <li>Higher accuracy</li>
</ul>

<h3>Disadvantages</h3>

<ul>
  <li>More parameters</li>
  <li>Slow training</li>
  <li>High memory usage</li>
</ul>

<hr>

<h1>📌 VGG19 Transfer Learning Code</h1>

<h2>Import Libraries</h2>

<pre>
from tensorflow.keras.applications import VGG19
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten
</pre>

<hr>

<h2>Load Pretrained VGG19</h2>

<pre>
vgg19_obj = VGG19(
    input_shape=(224,224,3),
    weights='imagenet',
    include_top=False
)
</pre>

<hr>

<h2>Freeze Layers</h2>

<pre>
for layer in vgg19_obj.layers:
    layer.trainable = False
</pre>

<hr>

<h2>Add Custom Layers</h2>

<pre>
x = Flatten()(vgg19_obj.output)

x = Dense(128, activation='relu')(x)

x = Dense(64, activation='relu')(x)

output = Dense(3, activation='softmax')(x)
</pre>

<hr>

<h2>Softmax Activation Function</h2>

<pre>
Softmax(z) = e^z / Σe^z
</pre>

<p>
Softmax is used for multi-class classification.
</p>

<hr>

<h2>Compile Model</h2>

<pre>
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
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
<td>Convolution Layers</td>
<td>13</td>
<td>16</td>
</tr>

<tr>
<td>Training Speed</td>
<td>Faster</td>
<td>Slower</td>
</tr>

<tr>
<td>Parameters</td>
<td>Less</td>
<td>More</td>
</tr>

<tr>
<td>Memory Usage</td>
<td>Lower</td>
<td>Higher</td>
</tr>

<tr>
<td>Accuracy</td>
<td>High</td>
<td>Slightly Higher</td>
</tr>

</table>

<hr>

<h1>✅ Advantages of VGG Models</h1>

<ul>
  <li>Easy architecture</li>
  <li>Powerful feature extraction</li>
  <li>High accuracy</li>
  <li>Transfer learning support</li>
  <li>Pretrained weights available</li>
</ul>

<hr>

<h1>❌ Limitations of VGG Models</h1>

<ul>
  <li>Very large model size</li>
  <li>High memory usage</li>
  <li>Slow training</li>
  <li>Computationally expensive</li>
</ul>

<hr>

<h1>🎯 Conclusion</h1>

<p>
VGG16 and VGG19 are powerful Deep Learning architectures widely used in computer vision tasks.
</p>

<p>
Both models use:
</p>

<ul>
  <li>Small 3×3 convolution filters</li>
  <li>Deep neural networks</li>
  <li>Transfer learning capability</li>
</ul>

<p>
VGG16 is faster and lighter, while VGG19 is deeper and slightly more accurate.
</p>

<hr>

<h2 align="center">
⭐ Thank You ⭐
</h2>
