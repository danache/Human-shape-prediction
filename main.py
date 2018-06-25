"""
Demo of HMR.

Note that HMR requires the bounding box of the person in the image. The best performance is obtained when max length of the person in the image is roughly 150px. 

When only the image path is supplied, it assumes that the image is centered on a person whose length is roughly 150px.
Alternatively, you can supply output of the openpose to figure out the bbox and the right scale factor.

Sample usage:

# On images on a tightly cropped image around the person
python -m demo --img_path data/im1963.jpg
python -m demo --img_path data/coco1.png

# On images, with openpose output
python -m demo --img_path data/random.jpg --json_path data/random_keypoints.json
"""



from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
sys.path.insert(0, "./external/hmr/")
sys.path.insert(0, "./src/")


from absl import flags
import numpy as np

import skimage.io as io
import tensorflow as tf

from src.util import renderer as vis_util
from src.util import image as img_util
from src.util import openpose as op_util
import src.config
from src.RunModel import RunModel

from nets import DenseNet
import torch
from pytorch_smpl.smpl import SMPL
import pytorch_smpl.measure as measure
from train import debug_display_cloud


flags.DEFINE_string('img_path', 'data/im1963.jpg', 'Image to run')
flags.DEFINE_string(
    'json_path', None,
    'If specified, uses the openpose output to crop the image.')


def visualize(img, proc_param, joints, verts, cam):
    """
    Renders the result in original image coordinate frame.
    """
    cam_for_render, vert_shifted, joints_orig = vis_util.get_original(
        proc_param, verts, cam, joints, img_size=img.shape[:2])

    # Render results
    skel_img = vis_util.draw_skeleton(img, joints_orig)
    rend_img_overlay = renderer(
        vert_shifted, cam=cam_for_render, img=img, do_alpha=True)
    rend_img = renderer(
        vert_shifted, cam=cam_for_render, img_size=img.shape[:2])
    rend_img_vp1 = renderer.rotated(
        vert_shifted, 60, cam=cam_for_render, img_size=img.shape[:2])
    rend_img_vp2 = renderer.rotated(
        vert_shifted, -60, cam=cam_for_render, img_size=img.shape[:2])

    import matplotlib.pyplot as plt
    # plt.ion()
    plt.figure(1)
    plt.clf()
    plt.subplot(231)
    plt.imshow(img)
    plt.title('input')
    plt.axis('off')
    plt.subplot(232)
    plt.imshow(skel_img)
    plt.title('joint projection')
    plt.axis('off')
    plt.subplot(233)
    plt.imshow(rend_img_overlay)
    plt.title('3D Mesh overlay')
    plt.axis('off')
    plt.subplot(234)
    plt.imshow(rend_img)
    plt.title('3D mesh')
    plt.axis('off')
    plt.subplot(235)
    plt.imshow(rend_img_vp1)
    plt.title('diff vp')
    plt.axis('off')
    plt.subplot(236)
    plt.imshow(rend_img_vp2)
    plt.title('diff vp')
    plt.axis('off')
    plt.draw()
    plt.show()
    # import ipdb
    # ipdb.set_trace()


def preprocess_image(img_path, json_path=None):
    img = io.imread(img_path)
    if img.shape[2] == 4:
        img = img[:, :, :3]

    if json_path is None:
        if np.max(img.shape[:2]) != config.img_size:
            print('Resizing so the max image size is %d..' % config.img_size)
            scale = (float(config.img_size) / np.max(img.shape[:2]))
        else:
            scale = 1.
        center = np.round(np.array(img.shape[:2]) / 2).astype(int)
        # image center in (x,y)
        center = center[::-1]
    else:
        scale, center = op_util.get_bbox(json_path)

    crop, proc_param = img_util.scale_and_crop(img, scale, center,
                                               config.img_size)

    # Normalize image to [-1, 1]
    crop = 2 * ((crop / 255.) - 0.5)

    return crop, proc_param, img




def preprocess_image_V2(img):
    if img.shape[2] == 4:
        img = img[:, :, :3]


    if np.max(img.shape[:2]) != config.img_size:
        print('Resizing so the max image size is %d..' % config.img_size)
        scale = (float(config.img_size) / np.max(img.shape[:2]))
    else:
        scale = 1.
    center = np.round(np.array(img.shape[:2]) / 2).astype(int)
    # image center in (x,y)
    center = center[::-1]


    crop, proc_param = img_util.scale_and_crop(img, scale, center,
                                               config.img_size)

    # Normalize image to [-1, 1]
    crop = 2 * ((crop / 255.) - 0.5)

    return crop, proc_param, img


def predict(image, weight, height):
    global config, renderer


    config = flags.FLAGS
    config(sys.argv)
    # Using pre-trained model, change this to use your own.
    config.load_path = src.config.PRETRAINED_MODEL
    config.batch_size = 1
    renderer = vis_util.SMPLRenderer(face_path=config.smpl_face_path)


    tf.reset_default_graph()
    sess = tf.Session()
    model = RunModel(config, sess=sess)

    input_img, proc_param, img = preprocess_image_V2(image)
    # Add batch dimension: 1 x D x D x 3
    input_img = np.expand_dims(input_img, 0)

    joints, verts, cams, joints3d, theta = model.predict(
        input_img, get_theta=True)
    sess.close()

    cams = theta[:, :model.num_cam]
    poses = theta[:, model.num_cam:(model.num_cam + model.num_theta)]
    shapes = theta[:, (model.num_cam + model.num_theta):]


  #  visualize(img, proc_param, joints[0], verts[0], cams[0])

    '''
    Start adjusting the shape
    '''
    shape_adjuster = torch.load("./trained/model_release_1.5961363467")
    smpl = SMPL("./models/neutral_smpl_with_cocoplus_reg.pkl")

    beta = torch.from_numpy(shapes).float().cuda()
    theta = torch.zeros((1, 72)).float().cuda()
    heights = torch.from_numpy(np.asarray([height]))
    volume = torch.from_numpy(np.asarray([weight]))

    verts, joints3d, Rs = smpl.forward(beta, theta, True)
    flatten_joints3d = joints3d.view(1, -1)
    heights = torch.unsqueeze(heights, -1).float().cuda()
    volumes = torch.unsqueeze(volume, -1).float().cuda()
    input_to_net = torch.cat((flatten_joints3d, heights, volumes), 1)

    adjusted_betas = shape_adjuster.forward(input_to_net)

    adjusted_verts, adjusted_joints3d, Rs = smpl.forward(adjusted_betas, theta, True)
    adjusted_heights = measure.compute_height(adjusted_verts)
    adjusted_volumes = measure.compute_volume(adjusted_verts, smpl.f)

    print (adjusted_heights, adjusted_volumes)

  #  debug_display_cloud(verts[0], joints3d[0], adjusted_verts[0], adjusted_joints3d[0])

    # Change the posture for measurement
    from measurement import POSE1
    theta = torch.from_numpy(np.expand_dims(POSE1, 0)).float().cuda()
    adjusted_verts, adjusted_joints3d, Rs = smpl.forward(adjusted_betas, theta, True)
    return torch.squeeze(adjusted_verts).detach().cpu().numpy(), \
           torch.squeeze(adjusted_volumes).detach().cpu().numpy(),\
           torch.squeeze(adjusted_heights).detach().cpu().numpy(),



def main(img_path, json_path=None):
    sess = tf.Session()
    model = RunModel(config, sess=sess)

    input_img, proc_param, img = preprocess_image(img_path, json_path)
    # Add batch dimension: 1 x D x D x 3
    input_img = np.expand_dims(input_img, 0)

    joints, verts, cams, joints3d, theta = model.predict(
        input_img, get_theta=True)

    cams = theta[:, :model.num_cam]
    poses = theta[:, model.num_cam:(model.num_cam + model.num_theta)]
    shapes = theta[:, (model.num_cam + model.num_theta):]


    visualize(img, proc_param, joints[0], verts[0], cams[0])

    '''
    Start adjusting the shape
    '''
    shape_adjuster = torch.load("./trained/model_save1.57873062595")
    smpl = SMPL("./models/neutral_smpl_with_cocoplus_reg.pkl")

    beta = torch.from_numpy(shapes).float().cuda()
    theta = torch.zeros((1, 72)).float().cuda()
    heights = torch.from_numpy(np.asarray([1.9]))
    volume = torch.from_numpy(np.asarray([90 / 1000.]))

    verts, joints3d, Rs = smpl.forward(beta, theta, True)
    flatten_joints3d = joints3d.view(1, -1)
    heights = torch.unsqueeze(heights, -1).float().cuda()
    volumes = torch.unsqueeze(volume, -1).float().cuda()
    input_to_net = torch.cat((flatten_joints3d, heights, volumes), 1)

    adjusted_betas = shape_adjuster.forward(input_to_net)

    adjusted_verts, adjusted_joints3d, Rs = smpl.forward(adjusted_betas, theta, True)
    adjusted_heights = measure.compute_height(adjusted_verts)
    adjusted_volumes = measure.compute_volume(adjusted_verts, smpl.f)

    print (adjusted_heights, adjusted_volumes)

    debug_display_cloud(verts[0], joints3d[0], adjusted_verts[0], adjusted_joints3d[0])






if __name__ == '__main__':
    config = flags.FLAGS
    config(sys.argv)
    # Using pre-trained model, change this to use your own.
    config.load_path = src.config.PRETRAINED_MODEL

    config.batch_size = 1

    renderer = vis_util.SMPLRenderer(face_path=config.smpl_face_path)

    main("/home/sparky/Documents/Projects/Human-Shape-Prediction/external/hmr/data/coco3.png", config.json_path)
