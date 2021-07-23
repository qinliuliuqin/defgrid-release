from collections import OrderedDict
import cv2
import numpy as np
import os
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from tqdm import tqdm


from dataloaders import cityscapes_full
from dataloaders import custom_transforms as tr
from Models.deformable_grid import DeformableGrid
from Utils.parser import get_args
from Utils.plot_sample import plot_deformed_lattice_on_image, plot_points
from Utils.matrix_utils import MatrixUtils
from scripts.contour import find_contour

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_test_loader(resolution):
    composed_transforms_ts = transforms.Compose([
        tr.FixedResize({'image': resolution, 'gt': resolution},
                       flagvals={'image': cv2.INTER_LINEAR, 'gt': cv2.INTER_NEAREST}),
        tr.CropRandom(crop_size=resolution, keys=['image', 'gt'], drop_origin=True),
        tr.BilateralFiltering(['crop_image']),
        tr.ToTensor(),
        tr.Normalize('crop_image', mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])
    set = cityscapes_full.cityscapesFullLoader(root = '/work/data/DefInput', split='test', transform=composed_transforms_ts)
    dimensions = set.dimension
    #print(dimensions)
    loader = DataLoader(set, batch_size=1, shuffle=False,
                                 num_workers=1, drop_last=True, )
    #print("set")
   # print(type(set))
    return loader,dimensions

def get_image_set():

    set = cityscapes_full.cityscapesFullLoader(root = '/work/data/DefInput', split='test', transform=None)
    image_set = set.images
    return image_set


def test_batch(model, data, data_idx, resolution, grid_size, grid_type, save_folder, image_dim, counter):
    # crop_gt = data['crop_gt'].to(device)
    # crop_gt = crop_gt
    crop_image = data['crop_image'].to(device)


    bifilter_crop_image = data['bifilter_crop_image']
    net_input = crop_image.clone()

    n_batch = net_input.shape[0]

    input_dict = {'net_input': net_input}
    matrix = MatrixUtils(1, grid_size, grid_type, device)

    base_point = matrix.init_point
    base_normalized_point_adjacent = matrix.init_normalized_point_adjacent
    base_point_mask = matrix.init_point_mask
    base_triangle2point = matrix.init_triangle2point
    base_area_mask = matrix.init_area_mask
    base_triangle_mask = matrix.init_triangle_mask


    input_dict['base_point'] = base_point.expand(n_batch, -1, -1)
    input_dict['base_normalized_point_adjacent'] = base_normalized_point_adjacent.expand(n_batch, -1, -1)
    input_dict['base_point_mask'] = base_point_mask.expand(n_batch, -1, -1)
    input_dict['base_triangle2point'] = base_triangle2point.expand(n_batch, -1, -1)
    input_dict['base_area_mask'] = base_area_mask.expand(n_batch, -1)
    input_dict['base_triangle_mask'] = base_triangle_mask.expand(n_batch, -1)
    input_dict['grid_size'] = np.max(grid_size)

    condition, laplacian_loss, variance, area_variance, reconstruct_loss, pred_points = model(
        **input_dict)

    i_image = 0
    base_point_adjacent = matrix.init_point_adjacent

    plot_pred_points = pred_points[i_image].detach().cpu().numpy()
    plot_pred_points[:, 0] = plot_pred_points[:, 0] * image_dim[0]
    plot_pred_points[:, 1] = plot_pred_points[:, 1] * image_dim[1]
    #print(plot_pred_points)
    final_img = bifilter_crop_image[i_image].permute(1, 2, 0).detach().cpu().numpy()
    final_img = cv2.resize(final_img, image_dim)

    #To print Deformed Grid
    plot_deformed_lattice_on_image(
        plot_pred_points,
        final_img,
        base_point_adjacent[0].detach().cpu().numpy(),
        mask=None,
        return_fig=True,
        save_path=os.path.join(save_folder, '%d.pdf'%(data_idx))
    )

    finder = find_contour.contourFinder(image=get_image_set()[counter])
    new_pred = swap(plot_pred_points)
    contour = finder.borderMask(new_pred)
    contour = swap(contour)
    contour = np.array(contour)
    #contour = finder.splineArray(contour)

    #
    # To print the border
    # #plot_points(contour, final_img, return_fig = True, save_path=os.path.join(save_folder, '%d.pdf'%(data_idx)))

    return plot_pred_points



def swap (list):
    li = np.zeros_like(list)
    for i in range(len(li)):
        li[i][0] = list[i][1]
        li[i][1] = list[i][0]
    return li

def main():
    args = get_args()
    print(args)

    model_path = '/work/defgrid/weights/epoch_100_iter_0.pth'
    save_folder = '/work/defgrid/results'
    if not os.path.isdir(save_folder):
        os.makedirs(save_folder)

    # load model parameters
    state_dict = torch.load(model_path)
    model_params = state_dict['model']
    model_params = OrderedDict(
        {k.replace('module.', ''): model_params[k] for k in model_params.keys()}
    )

    model = DeformableGrid(args, device)
    model.load_state_dict(model_params)
    model.to(device)
    model.eval()

    # load dataset
    test_loader, dimensions = get_test_loader(args.resolution)
    counter = 0
    for idx, data in tqdm(enumerate(test_loader)):
        grid_size = [int(dimensions[counter][0] / 10), int(dimensions[counter][1] / 10)]
        image_dim = [dimensions[counter][1], dimensions[counter][0]]
        with torch.no_grad():
            pred_points = test_batch(model, data, idx, args.resolution, grid_size, args.grid_type, save_folder, image_dim, counter)

        counter += 1


if __name__ == '__main__':
    main()