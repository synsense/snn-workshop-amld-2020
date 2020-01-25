from aermanager import LiveDv
from sinabs.from_torch import from_model
import torch
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from torch import nn


class MNISTClassifier(nn.Module):
    def __init__(self):
        super().__init__()

        self.seq = nn.Sequential(*[
            nn.Conv2d(in_channels=1, out_channels=8,
                      kernel_size=(3, 3), bias=False),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=(2, 2), stride=(2, 2)),
            nn.Conv2d(in_channels=8, out_channels=32,
                      kernel_size=(3, 3), bias=False),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=(2, 2), stride=(2, 2)),
            nn.Conv2d(in_channels=32, out_channels=16,
                      kernel_size=(3, 3), bias=False),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=(2, 2), stride=(2, 2)),
            nn.Dropout2d(0.5),
            nn.Flatten(),
            nn.Linear(576, 32, bias=False),
            nn.ReLU(),
            nn.Linear(32, 10, bias=False),
            nn.ReLU(),
        ])

    def forward(self, x):
        return self.seq(x)


# instantiating the model and transferring to GPU
model = MNISTClassifier()
model.cuda()
# reload the model from saved, if necessary
model.load_state_dict(torch.load('mnist_net_saved.pth'))

net = from_model(
    model.seq,
    input_shape=(1, 64, 64),
    threshold=1.0,
    membrane_subtract=1.0,
    threshold_low=-5.0,
).cuda()

SYNOP_POWER = 10e-8  # mJ
TIMESTEP_LENGTH = 10  # ms
N_TIMESTEPS_IN_BATCH = 10
mw_conversion_factor = SYNOP_POWER * 1000 / TIMESTEP_LENGTH / N_TIMESTEPS_IN_BATCH


live = LiveDv(host='localhost', port=7777, qlen=10)

# we resize and crop our input so that it matches the training data
adaptivepool = torch.nn.AdaptiveAvgPool2d((64, 64))
resize_factor = 16

def transform(x):
    x = x[:, :, 2:-2, 45:-45]  # crop
#     display(x.shape)
    x = torch.tensor(x).float().cuda()
    x = adaptivepool(x) * resize_factor
    return x


def process_batch():
    batch = live.get_batch()
    batch = transform(batch)

    out = net(batch)
    maxval, pred_label = torch.max(out.sum(0), dim=0)
    power = net.get_synops(0)['SynOps'].sum() * mw_conversion_factor
    
    # return out.sum(0).cpu().numpy()
    THR = 30
    if maxval > THR:
        print(pred_label.item(), power)
    else:
        print('.', power)


while True:
    process_batch()


# fig, ax = plt.subplots()
# im = plt.bar(np.arange(10), np.zeros(10))


# def init():
#     bars = [b for b in im]
#     ax.set_xlim(-1, 10)
#     ax.set_ylim(0, 1)
#     return bars

# def update(frame):
#     out = process_batch()
#     total = out.sum()
#     for i, b in enumerate(im):
#         im[i].set_height(out[i] / total + 0.01)
#     bars = [b for b in im]
#     return bars

# ani = FuncAnimation(fig, update, frames=300, init_func=init, blit=True, interval=100)
# plt.show()