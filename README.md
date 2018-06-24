# Human-Shape-Prediction
A dense deep neural network for human shape prediction from provided height, weight and approximate 3d keypoints locations.

1) The 3d keypoints locations are initially predicted by using https://github.com/glinsun/hmr.
2) The human-shape-prediciton net (HSPN) takes the approximate 3d keypoints locations, weight and height, and it predicts the shape parametes so that the relative positions of 3d keypoints are still the same, but the weight and the height correspond to the provided weight and height.

### Demo


1. Download the pre-trained models (hmr module)
```
wget https://people.eecs.berkeley.edu/~kanazawa/cachedir/hmr/models.tar.gz && tar -xf models.tar.gz
```

2. Run the demo
```
python -m main --img_path external/hmr/data/coco1.png
python -m main --img_path external/hmr/data/im1954.jpg
