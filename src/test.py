import glob

import ray
import ray.tune as tune
from pylab import *
from ray.tune import TrainingResult
from ray.tune.trainable import Trainable
from torch.nn.modules.normalization import *

import pytorch_smpl.measure as measure
from nets import DenseNet
from pytorch_camera.camera import Camera
from pytorch_smpl.smpl import SMPL


def debug_display_joints(joints2d, true_joints2d):
    joints2d = joints2d.view(2, -1).cpu().detach().numpy()
    joints2d = np.squeeze(joints2d)

    true_joints2d = true_joints2d.view(2, -1).cpu().detach().numpy()
    true_joints2d = np.squeeze(true_joints2d)


    plt.figure(2)
    plt.clf()
    plt.plot(joints2d[0,:], joints2d[1,:], 'ko')
    plt.plot(true_joints2d[0,:], true_joints2d[1,:], 'go')
    plt.gca().set_aspect('equal', adjustable='box')
    plt.draw()
    plt.pause(1e-6)


def debug_display_cloud(verts, joints, true_verts, true_joints):
    verts = verts.cpu().detach().numpy()
    verts = np.squeeze(verts)

    joints = joints.cpu().detach().numpy()
    joints = np.squeeze(joints)

    true_verts = true_verts.cpu().detach().numpy()
    true_verts = np.squeeze(true_verts)

    true_joints = true_joints.cpu().detach().numpy()
    true_joints = np.squeeze(true_joints)

    plt.clf()

    fig = plt.figure(1)
    ax3d = fig.gca(projection='3d')
    ax3d.clear()
    ax3d.set_aspect("equal")
    ax3d.set_xlim3d(-1, 1)
    ax3d.set_ylim3d(-1, 1)
    ax3d.set_zlim3d(-1, 1)
    ax3d.plot(verts[:,0], verts[:,1], verts[:,2], 'k,')
    ax3d.plot(joints[:,0], joints[:,1], joints[:,2], 'ko')
    ax3d.plot(true_verts[:,0], true_verts[:,1], true_verts[:,2], 'g,')
    ax3d.plot(true_joints[:,0], true_joints[:,1], true_joints[:,2], 'go')
    plt.draw()
    plt.pause(1e-6)




# http://ray.readthedocs.io/en/latest/tune.html


class Trainer(Trainable):

    def _setup(self):
        # 1) Initialize all needed parameters
        self.smpl = SMPL('/home/king/Projects/GITHUB/hmr/models/neutral_smpl_with_cocoplus_reg.pkl')
        self.camera = Camera()
        self.batch_size = int(self.config["batch_size"])
        self.net = DenseNet(self.config)
        self.net.cuda()
        # optimizer setup
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=self.config["lr"], weight_decay=self.config["weight_decay"])
        self.posesV0  = glob.glob('/home/king/Documents/measurement/dataset/up-3d/*.pkl')

    def _sample_random_theta(self):
        root = '/home/king/Documents/measurement/dataset/up-3d/'
        path = self.posesV0[np.random.random_integers(0, len(self.posesV0) - 1)]
        with open(path, 'rb') as f:
            # print path
            stored_parameters = pickle.load(f)
            orig_pose = np.array(stored_parameters['pose']).copy()
            orig_rt = np.array(stored_parameters['rt']).copy()
            orig_trans = np.array(stored_parameters['trans']).copy()
            orig_t = np.array(stored_parameters['t']).copy()
            thetas = stored_parameters['pose']
            thetas[:3] = np.random.uniform(0, 2*np.pi, 3)

        return np.asarray(thetas)

    # The inner function for batch generation
    def _get_batch(self, N):
        beta =  4 * torch.randn((N, 10)).float().cuda() #torch.zeros((N, 10)).float().cuda()
        theta = np.ones((N, 72)) * np.expand_dims(self._sample_random_theta(), 0) #np.zeros((N, 72)) #
        theta = torch.from_numpy(theta).float().cuda()
        # Without pose but with shape for measuring
        verts, joints3d, Rs = self.smpl.forward(beta, theta, True)
        heights = measure.compute_height(verts)
        volumes = measure.compute_volume(verts, self.smpl.f)
        joints2d, camera_parameters = self.camera.forward(joints3d)
        return joints2d, verts, joints3d, camera_parameters, beta, theta, heights, volumes

    def _save(self, checkpoint_dir):
        file_path = checkpoint_dir + "/model_save"
        torch.save(self.net, file_path)
        return file_path

    def _restore(self, path):
        self.net = torch.load(path)

    def _stop(self):
        # If need, save your model when exit.
         saved_path = self.logdir + "/model_stop"
         torch.save(self.net, saved_path)
         print("save model at: ", saved_path)


    def _train(self):
        # 2) After the initialization has been done, we can start training the model
        # The training loop


        # Get the training samples
        joints2d, verts, joints3d, camera_parameters, beta, theta, heights, volumes = self._get_batch(
            self.batch_size)
        joints2d = joints2d.view((self.batch_size, -1))
        heights = torch.unsqueeze(heights, -1)
        volumes = torch.unsqueeze(volumes, -1)
        input_to_net = torch.cat((joints2d, heights, volumes), 1)
        input_to_net = torch.unsqueeze(input_to_net, -1)

        eval_loss = inf
        while eval_loss > (1.3 / 32.):
            self.net.train()

            # clean the gradients
            self.net.zero_grad()
            self.camera.zero_grad()
            self.smpl.zero_grad()

            # Prediction ===============================================================================================
            # predicted_theta, predicted_beta, predicted_camera_parameters = net.forward(joints2d)
            predicted_theta, predicted_beta = self.net.forward(input_to_net)
            verts, predicted_joints3d, Rs = self.smpl.forward(predicted_beta, predicted_theta, True)
            predicted_joints2d, _ = self.camera.forward(predicted_joints3d, camera_parameters)
            # make sure they are references to the same object
            assert (_ is camera_parameters)

            verts, _, Rs = self.smpl.forward(predicted_beta, torch.zeros_like(predicted_theta), True)
            predicted_volumes = measure.compute_height(verts)
            predicted_heights = measure.compute_volume(verts, self.smpl.f)

            beta_loss = F.mse_loss(torch.squeeze(predicted_beta), torch.squeeze(beta))
            theta_loss =  F.mse_loss(torch.squeeze(predicted_theta), torch.squeeze(theta))

            total_loss = theta_loss + beta_loss
            total_loss.backward()
            self.optimizer.step()


            # Evaluation ===============================================================================================
            self.net.eval()
            predicted_theta, predicted_beta = self.net.forward(input_to_net)
            verts, predicted_joints3d, Rs = self.smpl.forward(predicted_beta, predicted_theta, True)
            predicted_joints2d, _ = self.camera.forward(predicted_joints3d, camera_parameters)

            beta_loss = F.mse_loss(torch.squeeze(predicted_beta), torch.squeeze(beta))
            theta_loss =  F.mse_loss(torch.squeeze(predicted_theta), torch.squeeze(theta))
            eval_loss = (theta_loss + beta_loss) / self.batch_size

            if float(np.random.random_sample()) > 0.99:
                true_verts, true_joints3d, Rs = self.smpl.forward(beta, theta, True)
                debug_display_cloud(verts[0], joints3d[0], true_verts[0], true_joints3d[0])
                debug_display_joints(predicted_joints2d[0], joints2d[0])
                print eval_loss

        return TrainingResult(timesteps_this_iter=1, mean_loss=eval_loss)


if __name__ == "__main__":
    # We start from hyperparameter optimization
    # https://github.com/ray-project/ray/tree/master/python/ray/tune/examples
    ray.init(num_workers=8, num_cpus=4, num_gpus=2, driver_mode=ray.SILENT_MODE)
    tune.register_trainable("train_model", Trainer)

    with torch.cuda.device(1):
        debug = Trainer(config= {
            "lr":  0.01,
            "weight_decay":  0.00,
            "batch_size":  32,
            "num_layers":  24,
            "num_blocks": 1,
            "k":  128,
            'activation':  "leaky_relu",
            })

        while True:
            debug._train()



    exp = tune.Experiment(
        name="measurements",
        local_dir="/home/king/ray_results/",
        run="train_model",
        repeat=32,
        checkpoint_freq=5,
        stop={"timesteps_total": 1e7}, # "timesteps_total": 1000,
        config={
            "lr": lambda spec: np.random.uniform(1e-6, 1e-2),
            "weight_decay": lambda spec: np.random.uniform(0, 0.2),
            "batch_size": lambda spec: np.random.choice([16, 32, 64]),
            "layers_num": lambda spec: np.random.choice([25, 50, 75]),
            "k": lambda spec: int(np.random.uniform(16, 320)),
            'activation': lambda spec: np.random.choice(["relu", "tanh", "leaky_relu"]),
        },
        trial_resources= {"cpu": 1, "gpu": 1})

    pbt = PopulationBasedTraining(
        time_attr="timesteps_total",
        reward_attr="neg_mean_loss",
        perturbation_interval=60, hyperparam_mutations=
        {
            "lr": lambda: np.random.uniform(1e-6, 1e-2),
            "weight_decay": lambda: np.random.uniform(0, 0.2),
            "batch_size": lambda: int(np.random.uniform(16, 64)),
            "layers_num": lambda: int(np.random.uniform(25, 75)),
            "k": lambda: int(np.random.uniform(16, 320)),
        })

    tune.run_experiments(exp, pbt)




   # joints2d, verts, joints3d, camera_parameters,  beta, theta = trainer.get_batch(3)


#    debug_display_joints(source_and_target)
#    for vert, joint3d in zip(verts, joints3d):
#        debug_display_cloud(vert, joint3d)







