import argparse
from csrnet_model import CSRNet
from csrnet_dataset import CsrNetDataset
import json
from lora import LocationReconstructionOperator, OperationType, OperationClass
from matplotlib import pyplot as plt
import time
import torch
from torchvision import transforms
import torch.nn.functional as functional

parser = argparse.ArgumentParser(description='Run Location Reconstruction Algorithm')

parser.add_argument('train_json', metavar='TRAIN',
                    help='path to train json')

parser.add_argument('--csrnet_weights', dest='csrnet_weights', type=str, help='CSRNet pretrained weights')

parser.add_argument('--output_dir', dest='output_dir', default='', type=str,
                    help='Location to output files')

parser.add_argument('--output_images', dest='output_images', default=0, type=int,
                    help='If 1, output images of the outputs and targets')

parser.add_argument('--silent', dest='silent', default=1, type=int,
                    help='If 1, only output minimal messages')


def main():
    args = parser.parse_args()
    args.batch_size = 1
    args.workers = 1

    csrnet = CSRNet(weights_path=args.csrnet_weights)
    csrnet = csrnet.cuda()

    with open(args.train_json, 'r') as outfile:
        img_paths = json.load(outfile)

    train_loader = torch.utils.data.DataLoader(
        CsrNetDataset(img_paths,
                      shuffle=False,
                      transform=transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                     std=[0.229, 0.224, 0.225]),
                           ]),
                      train=False),
        batch_size=args.batch_size)
    running_mae_iro_to_truth = 0.
    running_mae_iro_to_csrnet = 0.
    running_mae_csrnet_to_truth = 0.
    with open(args.output_dir + "/log.txt", "w+") as logfile:
        for i, (img, unaltered_image, _, locations) in enumerate(train_loader):
            img = img.cuda()
            density = csrnet(img).detach()
            density = functional.interpolate(density,
                                             size=[img.size()[2], img.size()[3]], mode='bilinear', align_corners=False)
            density = density / 64.

            data = torch.zeros((density.size()[2], density.size()[3])).cuda()

            start_time = time.perf_counter()

            if args.silent == 0:
                print("Current Step: Initial Scan")

            data = initial_scan(data, density, args)

            if args.silent == 0:
                print("Current Step: Point Shifting")

            data = fast_shift_points(data, density, args)

            output_composite_image(unaltered_image.squeeze(), data, args.output_dir + "/" + str(i + 1) + "_overlay.png")

            end_time = time.perf_counter()

            truth_count = locations.sum().item()
            csrnet_count = density.sum().item()
            iro_count = data.sum().item()

            running_mae_iro_to_truth += abs(truth_count - iro_count)
            running_mae_iro_to_csrnet += abs(csrnet_count - iro_count)
            running_mae_csrnet_to_truth += abs(truth_count - csrnet_count)

            print("Image", i + 1)
            print("Truth Count:", truth_count)
            print("CSRNet Predicted Count:", csrnet_count)
            print("IRO Predicted Count:", iro_count)
            print("Elapsed time for image", i + 1, "in seconds:", end_time - start_time)
            print("")

            logfile.write("Image " + str(i + 1) + "...\n")
            logfile.write("Truth Count: " + str(truth_count) + "\n")
            logfile.write("CSRNet Predicted Count: " + str(csrnet_count) + "\n")
            logfile.write("IRO Predicted Count: " + str(iro_count) + "\n")
            logfile.write("Elapsed time for image " + str(i + 1) + " in seconds: " + str(end_time - start_time) + "\n")
            logfile.write("\n")

        iro_to_truth_mae = running_mae_iro_to_truth / len(train_loader)
        iro_to_csrnet_mae = running_mae_iro_to_csrnet / len(train_loader)
        csrnet_to_truth_mae = running_mae_csrnet_to_truth / len(train_loader)
        print("Final Results:")
        print("IRO MAE Relative to Truth:", iro_to_truth_mae)
        print("IRO MAE Relative to CSRNet Output:", iro_to_csrnet_mae)
        print("CSRNet MAE Relative to Truth:", csrnet_to_truth_mae)

        logfile.write("Final Results:\n")
        logfile.write("IRO MAE Relative to Truth: " + str(iro_to_truth_mae) + "\n")
        logfile.write("IRO MAE Relative to CSRNet Output: " + str(iro_to_csrnet_mae) + "\n")
        logfile.write("CSRNet MAE Relative to Truth: " + str(csrnet_to_truth_mae) + "\n")


def output_composite_image(img: torch.Tensor, locations: torch.Tensor, path: str):
    r"""Overlays 'locations' on top of 'img' in green and outputs the result to the location provided in 'path'.
    The green overlay will have a transparency proportional to the strength of the location. e.g. A '1' in the location
    will cause a green pixel as output. A '0.5' will cause a green shadow over the top of the background at that pixel.
    Args:
        img: The background of the composite image. Shape [3, H, W]
        locations: The foreground of the composite image. Shape [1, H, W]
        path: The path to the outputted image.
    """
    assert img.dim() == 3, 'Image tensor must have three dimensions'
    # noinspection PyArgumentList
    assert img.size()[0] == 3, 'Image data must contains three channels'
    assert locations.dim() == 2, 'Location tensor must have two dimensions'

    def _draw_tick(canvas: torch.Tensor, x_pos: int, y_pos: int):
        r"""Draws a mark at the location specified on the 'canvas' tensor.
        """
        green = 1

        def _set_pixel(pixel_layer: int, pixel_x: int, pixel_y: int, value: float):
            # noinspection PyArgumentList
            if 0 <= pixel_x < canvas.size()[1] and 0 <= pixel_y < canvas.size()[2]:
                canvas[pixel_layer][pixel_x][pixel_y] = value

        # Black background...
        for color in range(3):
            _set_pixel(color, x_pos, y_pos + 4, 0.)
            _set_pixel(color, x_pos + 1, y_pos + 4, 0.)
            _set_pixel(color, x_pos - 1, y_pos + 4, 0.)
            _set_pixel(color, x_pos, y_pos - 4, 0.)
            _set_pixel(color, x_pos + 1, y_pos - 4, 0.)
            _set_pixel(color, x_pos - 1, y_pos - 4, 0.)
            _set_pixel(color, x_pos + 4, y_pos, 0.)
            _set_pixel(color, x_pos + 4, y_pos + 1, 0.)
            _set_pixel(color, x_pos + 4, y_pos - 1, 0.)
            _set_pixel(color, x_pos - 4, y_pos, 0.)
            _set_pixel(color, x_pos - 4, y_pos + 1, 0.)
            _set_pixel(color, x_pos - 4, y_pos - 1, 0.)

            for offset_x in range(-3, 4):
                for offset_y in range(-3, 4):
                    _set_pixel(color, x_pos + offset_x, y_pos + offset_y, 0.)

        # Green cross...
        _set_pixel(green, x_pos, y_pos, 1.)
        _set_pixel(green, x_pos - 1, y_pos - 1, 1.)
        _set_pixel(green, x_pos - 1, y_pos + 1, 1.)
        _set_pixel(green, x_pos + 1, y_pos - 1, 1.)
        _set_pixel(green, x_pos + 1, y_pos + 1, 1.)
        _set_pixel(green, x_pos - 2, y_pos - 2, 1.)
        _set_pixel(green, x_pos - 2, y_pos + 2, 1.)
        _set_pixel(green, x_pos + 2, y_pos - 2, 1.)
        _set_pixel(green, x_pos + 2, y_pos + 2, 1.)

    # noinspection PyTypeChecker
    img = torch.mean(img, dim=0)
    background = torch.stack((img, img, img), dim=0)

    pts = torch.nonzero(locations)
    # noinspection PyArgumentList
    for i in range(pts.size()[0]):
        x = int(pts[i][0])
        y = int(pts[i][1])
        _draw_tick(background, x, y)

    to_image = transforms.ToPILImage()
    out_img = to_image(background.detach().cpu())

    plt.imsave(path, out_img)


def fast_shift_points(data: torch.Tensor, density: torch.Tensor, args) -> torch.Tensor:
    """
    Goes through all points and moves them around to see if that improves the outcome.
    """
    start_time = time.perf_counter()

    lro = LocationReconstructionOperator(output_dir=args.output_dir)
    lro.set_data(discrete_data=data, target=density)

    if args.output_images == 1:
        plt.gray()
        plt.imsave(args.output_dir + "/points_round_0.png",
                   lro.get_discrete_data().detach().cpu().numpy().squeeze())

    max_rounds = 100
    movement_history = []
    movement_history_max = 6
    for round_index in range(max_rounds):
        lro.accept_changes()
        pts = torch.nonzero(lro.get_discrete_data())

        # noinspection PyArgumentList
        point_count = pts.size()[0]

        movement_count = 0
        movement_count_by_type = {}
        for i in range(point_count):
            x = int(pts[i][0])
            y = int(pts[i][1])

            op = lro.run_shift(x, y)
            if op != OperationType.NO_CHANGE:
                movement_count += 1
                if op.operation_class() in movement_count_by_type:
                    movement_count_by_type[op.operation_class()] += 1
                else:
                    movement_count_by_type[op.operation_class()] = 1

        if args.output_images == 1:
            plt.gray()
            plt.imsave(args.output_dir + "/points_round_" + str(round_index + 1) + ".png",
                       lro.get_discrete_data().detach().cpu().numpy().squeeze())

        # Record the count for the last few rounds...
        movement_history.append(movement_count)
        if len(movement_history) > movement_history_max:
            movement_history.pop(0)

        if not args.silent == 1:
            print("Round", round_index + 1, "Complete")
            print("Modified", movement_count, "out of", point_count, "points")

            for name, operation_class in OperationClass.__members__.items():
                if operation_class in movement_count_by_type:
                    print(name, movement_count_by_type[operation_class], "out of", point_count, "points")
                else:
                    print(name, "0 out of", point_count, "points")

            print("")

        if movement_count == 0:
            # No movements this round. Let's stop...
            if args.silent == 0:
                print("Completed", round_index + 1, "Rounds")
                print("Stopping due to no more movements")
            break

        change_has_occurred = True
        if len(movement_history) == movement_history_max:
            change_has_occurred = False
            for count_episode in movement_history:
                if count_episode != movement_count:
                    change_has_occurred = True
                    break

        if not change_has_occurred:
            # No change in the number of movements in the last few rounds...
            if args.silent == 0:
                print("Completed", round_index + 1, "Rounds")
                print("Stopping due to number of movements staying constant for " +
                      str(movement_history_max) + " rounds.")
            break

    end_time = time.perf_counter()

    if args.silent == 0:
        print("Elapsed time in seconds:", end_time - start_time)
        print("")

    return lro.get_discrete_data()


def initial_scan(data: torch.Tensor, density: torch.Tensor, args) -> torch.Tensor:
    start_time = time.perf_counter()

    lro = LocationReconstructionOperator(output_dir=args.output_dir)
    lro.set_data(discrete_data=data, target=density)

    replacement_count = 0

    # Subtract any existing locations from the density map...
    partial_density = density - lro.get_continuous_data()

    threshold = 0.001
    density_high_only = partial_density.masked_fill(partial_density < threshold, 0.)
    pts = torch.nonzero(density_high_only)

    # noinspection PyArgumentList
    point_count = pts.size()[0]

    # If they are all zero, there are no more to be found...
    while point_count > 0:
        previous_replacement_count = replacement_count
        for row_index in range(point_count):
            x = int(pts[row_index][2])
            y = int(pts[row_index][3])

            added_point = lro.run_add(x, y)
            if added_point:
                replacement_count += 1

        if previous_replacement_count == replacement_count:
            # We scanned all the non-zero points and haven't found any more to add. Let's call it a day...
            break

    data = lro.get_discrete_data()

    end_time = time.perf_counter()

    if args.silent == 0:
        print("Added", replacement_count, "points.")
        print("Elapsed time in seconds:", end_time - start_time)
    return data


if __name__ == '__main__':
    main()
