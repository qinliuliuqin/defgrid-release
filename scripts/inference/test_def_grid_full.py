from collections import OrderedDict
import torch

from Models.deformable_grid import DeformableGrid
from Utils.parser import get_args


if __name__ == '__main__':

    args = get_args()
    print(args)

    model_path = '/work/defgrid/weights/epoch_100_iter_0.pth'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load model parameters
    state_dict = torch.load(model_path)
    model_params = state_dict['model']
    model_params = OrderedDict(
        {k.replace('module.', ''): model_params[k] for k in model_params.keys()}
    )

    model = DeformableGrid(args, device)
    model.load_state_dict(model_params)

    # df
