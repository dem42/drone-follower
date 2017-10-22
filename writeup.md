## Project: Follow Me

[//]: # (Image References)
[image1]: ./images/network.png
[image2]: ./images/overfitting.png
[image3]: ./images/train_img1.jpeg
[image4]: ./images/train_img2.jpeg
[image5]: ./images/train_img3.jpeg
[image6]: ./images/following1.PNG
[image7]: ./images/following2.PNG
[image8]: ./images/pred_viz1.PNG
[image9]: ./images/pred_viz2.PNG

### Intro
The purpose of the Follow Me project is to build a semantic segmentation filter which when applied to an input image will classify each pixel of that image into one of three categories. In this project the categories are `[hero, crowd, background]`. The semantic segmentation filter therefore gives us scene understanding of an image at pixel resolution. The purpose is to use the classifier as the decision mechanism in a follower drone, which locates the hero in its camera input and follows the hero.

### Network architecture
My network architecture choice in this project was inspired by the architecture described in by Long, Shelhamer and Darrell [^1] where multiple convolutional layers of same filter depth are interspersed with max pooling layers to achieve input size reduction. 

My chosen architecture is that of a fully convolutional nework which consists of an encoder and a decoder. Here is the architecture in visual format:

![Network Architecture][image1]

###### Encoder
The encoder consists of four blocks, each of which contains two separable convolutions with `kernel=3, stride=1` and a max pooling layer with `stride=2`. The first block of convolutions learn parameters for 32 filters, the second block for 64, the third block for 128 filters and the final block learns 256 filters. This group of four blocks of increasing filter depth is then connected to a 1x1 convolutional layer with filter depth 256. This layer collects all the pixel information into a deep output. The purpose of the encoder is to extract features into a thin but deep output tensor.

A 1x1 convolutional layer is necessary because it preseves spatial information since it doesn't flatten the input tensor. In classical convolutional networks a fully connected layer (dense layer) is normally used as the final layer which collects all information, but such a layer flattens the tensor and so loses spatial information. 

All of the seperable convolutional layers use a rectified linear unit as an activation function to add more non-linearity. Additionally, each separable convolutional layer has a batch normalization layer attached. This normalizes the tensor passed into the next layer and thus improves regularization (avoids overfitting). I use separable convolutional layers which have the advantage of needing fewer parameters than normal convolutional layers. This makes training faster. 

I used a stride of 1 in the convolutional layers because the max pooling layers take care of downscaling in my network. I tried using just stride without using max pooling, but the results were much worse (max 40% accuracy).

I used four a total of 8 convolutional layers because I found that adding layers improved the IOU metric because it adds more non-linearity. This however slows down the training process and the improvements decrease the more layers are added.

I also experimented using `layers.DropOut` inside the encoder to improve regularization, but I didn't see any improvements. This coincides with what Long et al [^1] mention in their paper where patch subsampling using dropout did not yield improvements.

###### Decoder
The decoder consists of four upsampling layers which use bilinear interpolation to perform upsampling. Since I  used max pooling with `stride=2` in the encoder I ended up reducing the input size by `2x2x2x2` along the width and height, so I used four layers of bilinear interpolation each of which upsamples by a factor of two. 

Additionally, I used layer concatenation in the decoder as in [^1]. The idea behind layer concatenation is to combine the coarsely upscaled input with data from a higher resolution. This is done by concatenating the output the upsampled input with a separable convolutional layer of the same size from the encoder.

Finally, the concatenated upscaled outputs are converted into classes using a normal convolutional layer. This layer has a filter depth three (for our three classes), a `kernel=3` and a softmax activation function to convert logits into probabilities at each pixel. The purpose of the decoder is to use the features from the encoder to decide per pixel classes.

### Hyperparameters
The hyperparameters to the network are the `learning_rate` (how large a step to take during stochastic gradient descent), `batch_num` (number of batches per SGD step), `epochs` (number of iterations of SGD training), `steps_per_epoch` (number of batches per SGD step), `validation_steps` (number of batches during validation step) and `workers` (processes used by keras).

I tuned the hyperparameters manually, using a gradient descent approach. I started with one hyperparameter and increased/decreased it until I could no longer see an improvement in the metric. Then I proceeded to the next parameter. I repeated this until I couldn't see any more noticable improvements. 

In the end I stettled on using `batch_num = 60` and I ran multiple trainings without recreating the model with the values:
`learning_rate = 0.01` and `epochs = 8`
`learning_rate = 0.001` and `epochs = 5`
`learning_rate = 0.0001` and `epochs = 5`
`learning_rate = 0.005` and `epochs = 5`
I did it this way because I noticed that if I run too many epochs with the same learning rate my networks end up overfitting as shown in the image below where the validation loss fluctuates heavily even though the training set loss is strictly decreasing. So instead I manually increased/decreased the learning rate, experimenting every few epochs based on how quickly the loss is decreasing or if a plateau has been reached.

![Overfitting][image2]

I found that a big improvement was to set `steps_per_epoch=300` and `validation_steps=100`. This seemed a bit counterintuitive to me because these parameters should be set to number of images divided by batch num. I assume it helped because my network architecture has many parameters due to many layers so it worked better when trained with a larger input per SGD step. However, setting increasing values caused each epoch to take 350 seconds even on a p2.xlarge in AWS. 

For `workers` I didn't see much speed up in training when increasing it above 10.

### Data collection
I first used the provided dataset to perform hyperparameter optimization. Then I collected around 5000 training and 2000 validation images using the drone simulator to build a good dataset with many different angles of the crowd, hero and the background. I focused on creating large crowds using short spawning and on taking images of the hero from all possible angles. Here are some example of the image I collected:

![Angle 1][image3]
![Angle 2][image4]
![Grass][image5]

The images are stored as .jpg to keep their size smaller and additionally a mask is stored which is used during training as a label for which pixels are which class.

### Results
After the training I achieved an mean IOU metric of `0.5762` (57.6% accuracy) with the model trained on my own dataset.

The final model is saved in the file h5 file `data_follower_model_57.h5` which is stored in the `code/` directory as well as the `data/weights/` directory.

The final numbers from the training were
`time per epoch: 371s`
`final epoch loss: 0.0150`
`final epoch val_loss: 0.0161`

Here are some images from running the drone in follow me mode:

![Following target][image6]
![Following target][image7]

And here is the view of the FCN output when running the follow me drone with `--pred_viz`

![Following target][image8]
![Following target][image9]

### Limitations and future improvements
The biggest limitation is of course that the network has been training with images of 3D models from the simulator. Therefore, it can only follow those models and cannot be used outside of the simulator. To use the drone in the real world it would have to be retrained with real images. Obviously, the 3d models in the simulator are very simple and thus easily learnable, for example the hero model has very distinct red colors which make it easy to recognize. 

However, in general the network could be used to follow any object as long as we could collect enough images of it and train the network with these images. So the network is very general, but needs a lot of data to achieve good accuracy.

The performance of my network I think can still be improved. I stopped adding layers after reaching a very good accuracy and also because adding more layers slowed down the network. I also didn't experiment too much with the kernel size, but in the paper from Long et al [^1] their kernel sizes are much larger in the initial layers, so perhaps this could improve the network performance.

### Bibliography
[^1]: Jonathan Long, Evan Shelhamer, Trevor Darrell, "Fully Convolutional Networks for Semantic Segmentation"
    https://people.eecs.berkeley.edu/~jonlong/long_shelhamer_fcn.pdf
