# Human-Shape-Prediction
A dense deep neural network for human shape prediction from provided height, weight and 3d keypoints locations.

1) The 3d keypoints locations are initially predicted by using https://github.com/glinsun/hmr.
2) The human-shape-prediciton net (HSPN) takes the approximate 3d keypoints locations, and it will predict shape parametes so reletive locations of 3d keypoints are still the same, but the weight and the height will correspond to the provided weight and height.


