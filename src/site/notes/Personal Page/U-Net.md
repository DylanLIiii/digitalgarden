---
{"dg-publish":true,"permalink":"/personal-page/u-net/"}
---

The U-Net is a type of convolutional neural network (CNN) that is specifically designed for image segmentation tasks. It consists of an encoder network that maps the input image to a lower-dimensional representation, and a decoder network that reconstructs the image from the lower-dimensional representation and produces a segmentation map.

The encoder network of a U-Net can be represented mathematically as a series of convolutional and pooling layers, which can be written as:

$x_1 = Conv_1(x_0) + b_1$

$x_2 = Conv_2(x_1) + b_2$

$\cdots$

$x_n = Conv_n(x_{n-1}) + b_n$

$p_n = MaxPool_n(x_n)$

Where:

-   $x_0$ is the input image
-   $x_1, x_2, \dots, x_n$ are the output feature maps at each convolutional layer
-   $b_1, b_2, \dots, b_n$ are the bias terms at each convolutional layer
-   $Conv_1(\cdot), Conv_2(\cdot), \dots, Conv_n(\cdot)$ are the convolutional operations at each layer
-   $MaxPool_n(\cdot)$ is the max pooling operation at the n-th layer

The pooling layers downsample the feature maps by taking the maximum value in each pooling window.

The decoder network of a U-Net can be represented mathematically as a series of upsampling and convolutional layers, which can be written as:

$u_n = UpSample_n(p_n)$

$u_{n-1} = Conv_{n+1}(u_n) + b_{n+1}$

$\cdots$

$u_1 = Conv_{n+m-1}(u_2) + b_{n+m-1}$

$\hat{x} = Conv_{n+m}(u_1) + b_{n+m}$

Where:

-   $u_n, u_{n-1}, \dots, u_1$ are the output feature maps at each upsampling layer
-   $b_{n+1}, b_{n+2}, \dots, b_{n+m}$ are the bias terms at each upsampling layer
-   $UpSample_n(\cdot)$ is the upsampling operation at the n-th layer
-   $Conv_{n+1}(\cdot), Conv_{n+2}(\cdot), \dots, Conv_{n+m}(\cdot)$ are the convolutional operations at each layer
-   $\hat{x}$ is the reconstructed image

The upsampling layers increase the spatial resolution of the feature maps by using interpolation or transposed convolution operations. The final layer of the decoder network produces a segmentation map by applying a sigmoid activation function to the output feature map.