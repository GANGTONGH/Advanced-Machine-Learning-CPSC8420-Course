clear all;

original = imread("Lenna_(test_image).png");
pixels = reshape(double(original),512*512,3);

[U, S, V] = svd(pixels*pixels', "econ")

