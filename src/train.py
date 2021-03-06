import ray
import ray.tune as tune
from mpl_toolkits.mplot3d import Axes3D  # <-- Note the capitalization!
from pylab import *
from ray.tune import TrainingResult
from ray.tune.trainable import Trainable
from torch.nn.modules.normalization import *
from torch.optim.lr_scheduler import ReduceLROnPlateau

import pytorch_smpl.measure as measure
from nets import DenseNet
from pytorch_camera.camera import Camera
from pytorch_smpl.smpl import SMPL
from scipy import misc
'''
2 experiments must be done

1) Point cloud predition

2) Betas and thetas prediction
'''


def debug_display_joints(joints3d, noisy_joints2d):
    joints3d = joints3d.cpu().detach().numpy()
    joints3d = np.squeeze(joints3d)

    noisy_joints2d = noisy_joints2d.cpu().detach().numpy()
    noisy_joints2d = np.squeeze(noisy_joints2d)



    plt.clf()

    fig = plt.figure(1)
    ax3d = Axes3D(fig)
    ax3d.clear()
    ax3d.set_aspect("equal")
    ax3d.set_xlim3d(-1, 1)
    ax3d.set_ylim3d(-1, 1)
    ax3d.set_zlim3d(-1, 1)
    ax3d.plot(joints3d[:,0], joints3d[:,1], joints3d[:,2], 'go')
    ax3d.plot(noisy_joints2d[:,0], noisy_joints2d[:,1], noisy_joints2d[:,2], 'ko')

    plt.draw()
    plt.show()



def debug_display_cloud(verts, joints, true_verts, true_joints, min_loss=0, save=False):
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
    ax3d = Axes3D(fig)
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

    if save:
        plt.pause(1e-6)
        plt.savefig(str(min_loss) + ".png")
    else:
        plt.show()

# http://ray.readthedocs.io/en/latest/tune.html
class Trainer(Trainable):

    def _setup(self):
        # 1) Initialize all needed parameters
        self.smpl = SMPL('../models/neutral_smpl_with_cocoplus_reg.pkl')
        self.camera = Camera()
        self.batch_size = int(self.config["batch_size"])
        self.net = DenseNet(self.config)
        self.net.cuda()
        # optimizer setup
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=self.config["lr"], weight_decay=self.config["weight_decay"])
        self.scheduler = ReduceLROnPlateau(self.optimizer, 'min', verbose=True)
        self.min_loss = inf
        self.num_iter = 5000

    def _sample_random_theta(self):
        # The pose can be not natural, since the task is to predict shape.
        return np.asarray(np.random.uniform(-2*np.pi, 2*np.pi, 72))

    # The inner function for batch generation
    def _get_batch(self):
        beta = 3 * torch.randn((self.batch_size, 10)).float().cuda()
        theta = torch.zeros((self.batch_size, 72)).float().cuda()

        # Without pose but with shape for measuring
        verts, joints3d, Rs = self.smpl.forward(beta, theta, True)
        heights = measure.compute_height(verts)
        volumes = measure.compute_volume(verts, self.smpl.f)


        # Must add artificial noise to the joints since joints detection algorithms are not perfect
        # -+ 2 cm
        noise = torch.normal(torch.zeros_like(joints3d), 0.04).float().cuda()
        noisy_joints3d = joints3d + noise

        # Visualize noisy joints
        #debug_display_joints(noisy_joints3d[0], joints3d[0])

        scale = torch.rand((self.batch_size, 1, 1)) * 3.75 + 0.25
        noisy_joints3d /= scale.float().cuda()

        # Must add artificial noise to the height and volume.
        # Volume must represent a weight of a person, so -+ 3 kg
        # Height is -+ 2 cm
        noise = torch.normal(torch.zeros_like(heights), 0.03).float().cuda()
        heights += noise

        noise = torch.normal(torch.zeros_like(volumes), volumes / 100.0).float().cuda()
        volumes += noise

        noisy_joints3d = noisy_joints3d.view(self.batch_size, -1)
        heights = torch.unsqueeze(heights, -1)
        volumes = torch.unsqueeze(volumes, -1)

        input_to_net = torch.cat((noisy_joints3d, heights, volumes), 1)
        return input_to_net.detach(), torch.squeeze(beta).detach()

    def _save(self, checkpoint_dir, postfix=None):
        file_path = checkpoint_dir + "/model_save_new" + postfix
        torch.save(self.net, file_path)
        return file_path

    def _restore(self, path):
        self.net = torch.load(path)

    def _stop(self):
        # If need, save your model when exit.
         saved_path = self.logdir + "/model_stop"
         torch.save(self.net, saved_path)
         print("save model at: ", saved_path)


    def _eval(self):
        total_loss = 0
        for i in range(self.num_iter):
            inputs, beta = self._get_batch()
            # Evaluation ===============================================================================================
            self.net.eval()
            predicted_beta = self.net.forward(inputs)
            beta_loss = F.mse_loss(predicted_beta, beta)
            total_loss += float(beta_loss.cpu().detach().numpy())

        total_loss = total_loss / float(self.num_iter)
        self.scheduler.step(total_loss)

        print '\n Evaluation finished: ', total_loss

        if self.min_loss > total_loss:
            self.min_loss = total_loss
            print 'Saving ...', ' the current loss is ', self.min_loss
            trainer._save('../trained/', str(self.min_loss))

            theta = torch.zeros((self.batch_size, 72)).cuda()
            verts, joints3d, Rs = self.smpl.forward(beta, theta, True)
            predicted_verts, predicted_joints3d, Rs = self.smpl.forward(predicted_beta, theta, True)
            debug_display_cloud(verts[0], joints3d[0], predicted_verts[0], predicted_joints3d[0], self.min_loss, True)


    def _train(self):
        total_loss = 0
        for i in range(self.num_iter):
            inputs, beta = self._get_batch()

            self.net.train()
            # clean the gradients
            self.net.zero_grad()
            self.camera.zero_grad()
            self.smpl.zero_grad()

            # Prediction ===============================================================================================
            # predicted_theta, predicted_beta, predicted_camera_parameters = net.forward(joints2d)
            predicted_beta = self.net.forward(inputs)
            beta_loss = F.mse_loss(predicted_beta, beta)
            beta_loss.backward()
            self.optimizer.step()
            total_loss += float(beta_loss.cpu().detach().numpy())

          #  debug_display_cloud


            if i % 100 == True:
                print i,

        total_loss = total_loss / float(self.num_iter)
        return TrainingResult(timesteps_this_iter=1, mean_loss=total_loss)


if __name__ == "__main__":
    ray.init(num_workers=8, num_cpus=4, num_gpus=2, driver_mode=ray.SILENT_MODE)
    tune.register_trainable("train_model", Trainer)
    # which gpu to use
    with torch.cuda.device(0):
        trainer = Trainer(config= {
            "lr":  0.001,
            "weight_decay":  0.0,
            "batch_size":  32,
            "num_layers":  30,
            "num_blocks": 3,
            "k":  1,
            'activation':  "leaky_relu",
            })

        # The main training loop
        while True:
            trainer._train()
            trainer._eval()



    # Hyper parameter optimization.
    # Currently we do not use it.
    # https://github.com/ray-project/ray/tree/master/python/ray/tune/examples
    '''
    exp = tune.Experiment(
        name="measurements",
        local_dir="/home/sparky/ray_results/",
        run="train_model",
        repeat=32,
        checkpoint_freq=5,
        stop={"timesteps_total": 1e7}, # "timesteps_total": 1000,
        config={
            "lr": lambda spec: np.random.uniform(1e-6, 1e-2),
            "weight_decay": lambda spec: np.random.uniform(0, 0.2),
            "num_blocks": lambda spec: 1,
            "batch_size": lambda spec: np.random.choice([8, 16, 32]),
            "num_layers": lambda spec: np.random.choice([500, 750, 1000]),
            "k": lambda spec: int(np.random.uniform(1, 8)),
            'activation': lambda spec: np.random.choice(["relu", "tanh", "leaky_relu"]),
        },
        trial_resources={"cpu": 1, "gpu": 1})

    pbt = PopulationBasedTraining(
        time_attr="timesteps_total",
        reward_attr="neg_mean_loss",
        perturbation_interval=60, hyperparam_mutations=
        {
            "lr": lambda: np.random.uniform(1e-6, 1e-2),
            "weight_decay": lambda: np.random.uniform(0, 0.2),
            "batch_size": lambda: int(np.random.uniform(1, 32)),
            "num_layers": lambda: int(np.random.uniform(25, 1000)),
            "k": lambda: int(np.random.uniform(1, 16)),
        })

    tune.run_experiments(exp, pbt)
    '''