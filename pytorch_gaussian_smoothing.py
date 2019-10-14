import math
import numbers
import torch
from torch import nn
from torch.nn import functional


# From https://discuss.pytorch.org/t/is-there-anyway-to-do-gaussian-filtering-for-an-image-2d-3d-in-pytorch/12351/10
class GaussianSmoothing(nn.Module):
    """
    Apply gaussian smoothing on a
    1d, 2d or 3d tensor. Filtering is performed separately for each channel
    in the input using a depthwise convolution.
    Arguments:
        channels (int, sequence): Number of channels of the input tensors. Output will
            have this number of channels as well.
        kernel_size (int, sequence): Size of the gaussian kernel.
        sigma (float, sequence): Standard deviation of the gaussian kernel.
        padding (int, optional): Padding to apply to inputs
        dim (int, optional): The number of dimensions of the data.
            Default value is 2 (spatial).
        use_cuda (bool, optional): If true, we expect CUDA data. Othwerwise CPU data.
    """
    def __init__(self, channels: int, kernel_size: int, sigma: float, padding: int = 2, dim: int = 2, use_cuda: bool = False):
        super(GaussianSmoothing, self).__init__()
        if isinstance(kernel_size, numbers.Number):
            kernel_size = [kernel_size] * dim
        if isinstance(sigma, numbers.Number):
            sigma = [sigma] * dim

        # The gaussian kernel is the product of the
        # gaussian function of each dimension.
        kernel = 1
        meshgrids = torch.meshgrid(
            [
                torch.arange(size, dtype=torch.float32)
                for size in kernel_size
            ]
        )
        for size, std, mgrid in zip(kernel_size, sigma, meshgrids):
            mean = (size - 1) / 2
            kernel *= 1 / (std * math.sqrt(2 * math.pi)) * torch.exp(-((mgrid - mean) / std) ** 2 / 2)

        # Make sure sum of values in gaussian kernel equals 1.
        kernel = kernel / torch.sum(kernel)

        # Reshape to depthwise convolutional weight
        kernel = kernel.view(1, 1, *kernel.size())
        kernel = kernel.repeat(channels, *[1] * (kernel.dim() - 1))

        if use_cuda:
            kernel = kernel.cuda()

        self.register_buffer('weight', kernel)
        self.groups = channels
        self.padding = padding

        if dim == 1:
            self.conv = functional.conv1d
        elif dim == 2:
            self.conv = functional.conv2d
        elif dim == 3:
            self.conv = functional.conv3d
        else:
            raise RuntimeError(
                'Only 1, 2 and 3 dimensions are supported. Received {}.'.format(dim)
            )

    def forward(self, data: torch.Tensor):
        """
        Apply gaussian filter to input.
        Arguments:
            data (torch.Tensor): Input to apply gaussian filter on.
        Returns:
            filtered (torch.Tensor): Filtered output.
        """
        return self.conv(data, weight=self.weight, groups=self.groups, padding=self.padding)
