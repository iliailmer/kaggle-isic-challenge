# My Solution for the Skin Cancer classification competition

Here is the link to the competition: https://www.kaggle.com/c/siim-isic-melanoma-classification/

The problem is as follows. Given a heavily imbalanced dataset of images with 0 class being >98% of the data and 1 class constituting all remaining images, we want to construct a model with the best AUC score on the test set.

To overcome the imbalance, first idea was to use a class rebalancing via over/undersampling. 
Using additional datasets did not work properly for me and did not improve my scores.

The models I used here were Efficient Net and ResNeSt-50 neural networks. With a 5-fold cross validation, the best score I was able to get with naive upsampling and hard augmentations was ~0.9 which is good but not great.

This is a work in progress.
<!-- After multiple attempts at that, I discovered this discussion: https://www.kaggle.com/c/siim-isic-melanoma-classification/discussion/165526 -->
