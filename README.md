# Facial Recognition Using Principal Component Analysis

An image recognition system using Matlab:
* Performed Principal Component Analysis on a set of 32x32 8-bit facial images.
* Projected a set of training and test images into the Eigenspace.
* Calculated Euclidean distance and displayed 3 nearest neighbors among trained images that best match the test image.


Steps followed:
==============
* Collection of training images
* Vectorization of each image to a column vector and combining them
* Computation of a mean(average) image
* Calculate the covariant matrix
* Get eigenvectors and sort in decreasing order of corresponding eigenvalues
* Selection of top K eigenvectors resulting in a face model
* Take input image from user, show best matches based on distance
