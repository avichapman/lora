from enum import Enum, auto, unique
from matplotlib import pyplot as plt
from pytorch_gaussian_smoothing import GaussianSmoothing
import torch


class OperationClass(Enum):
    """
    Broad classes of operations.
    """
    NO_CHANGE = auto()
    MOVE = auto()
    REMOVAL = auto()
    SPLIT = auto()


@unique
class OperationType(Enum):
    """
    Describes the possible supported operations.

    The following operations are supported:
        - No Change
        - Moving a pixel North
        - Moving a pixel North East
        - Moving a pixel East
        - Moving a pixel South East
        - Moving a pixel South
        - Moving a pixel South West
        - Moving a pixel West
        - Moving a pixel North West
        - Deletion of a pixel
        - Splitting a pixel into two horizontally (one pixel becomes two pixels separated by an empty pixel)
        - Splitting a pixel into two vertically (one pixel becomes two pixels separated by an empty pixel)
        - Splitting a pixel into two diagonally ascending (one pixel becomes two pixels separated by an empty pixel)
        - Splitting a pixel into two diagonally descending (one pixel becomes two pixels separated by an empty pixel)
    """
    NO_CHANGE = 0
    MOVE_N = 1
    MOVE_NE = 2
    MOVE_E = 3
    MOVE_SE = 4
    MOVE_S = 5
    MOVE_SW = 6
    MOVE_W = 7
    MOVE_NW = 8
    REMOVAL = 9
    SPLIT_H = 10
    SPLIT_V = 11
    SPLIT_DA = 12
    SPLIT_DD = 13

    def operation_class(self) -> OperationClass:
        """
        Returns the broad class of operation this is.
        :return: OperationClass
        """
        if OperationType.MOVE_N.value <= self.value <= OperationType.MOVE_NW.value:
            return OperationClass.MOVE

        if OperationType.SPLIT_H.value <= self.value <= OperationType.SPLIT_DD.value:
            return OperationClass.SPLIT

        if self is OperationType.REMOVAL:
            return OperationClass.REMOVAL

        return OperationClass.NO_CHANGE


class LocationReconstructionOperator(object):
    """
    Runs a discrete tensor through a series of potential operations and selects the one that provides the greatest
    improvement in loss, when compared with a target tensor.

    It is assumed that the target tensor is the result of some unknown discrete map (2d tensor containing only ones and
    zeros) convolved with a Gaussian kernel, referred to herein as a continuous map. Loss is therefore calculated by
    convolving the candidate discrete map and then performing an MSE comparison with the target Tensor.

    There are two phases of operation. The 'add' phase of operations concentrates on adding new points to the discrete
    map. It can be triggered with a call to 'run_add'. The 'shift' phase of operations concentrates on moving, splitting
    or removing existing points. It can be triggered with a call to 'run_shift'.

    See the OperationType enum for a list of what operations are supported.
    """
    def __init__(self,
                 output_dir: str):
        super().__init__()
        self.output_dir = output_dir

        self.sigma = 4.881709115
        self.kernel_size = 37
        self.kernel_padding = 18
        self.max_fling_distance = 9
        self.padding = self.kernel_padding + self.max_fling_distance
        self.slice_width = self.kernel_size + self.max_fling_distance * 2
        self.diagnostic_mode = False
        self.output_effect_tensor_graphs = False
        self.shift_operation_count = len(OperationType.__members__.items())
        self.add_operation_count = 2
        self.max_operation_count = max(self.shift_operation_count, self.add_operation_count)

        # This is the data to be operated on...
        self.padded_discrete_data = None
        self.continuous_data_cube = None
        self.padded_target = None
        self.target_cube = None

        self.add_convolution = GaussianSmoothing(channels=self.add_operation_count,
                                                 kernel_size=self.kernel_size,
                                                 sigma=self.sigma,
                                                 padding=self.kernel_padding,
                                                 use_cuda=torch.cuda.is_available())
        self.shift_convolution = GaussianSmoothing(channels=self.shift_operation_count,
                                                   kernel_size=self.kernel_size,
                                                   sigma=self.sigma,
                                                   padding=self.kernel_padding,
                                                   use_cuda=torch.cuda.is_available())

        # The x and y pos of the centre of the convolution...
        self.zero_x = self.slice_width // 2
        self.zero_y = self.slice_width // 2

        # Create set of operations for the shifting phase of operations...
        self.discrete_shift_operations = self._create_discrete_operations()
        self.continuous_shift_operations = self._create_continuous_operations(self.discrete_shift_operations)
        self.continuous_shift_operations_cube = self._create_operations_cube(self.continuous_shift_operations)

        # Create set of operations for the adding phase of operations...
        self.discrete_add_operations = self._create_discrete_add_operation()
        self.continuous_add_operations = self._create_continuous_operations(self.discrete_add_operations)
        self.continuous_add_operations_cube = self._create_operations_cube(self.continuous_add_operations)

    def run_convolution(self, discrete_data: torch.Tensor) -> torch.Tensor:
        """
        Accepts a discrete data tensor and produces the equivalent continuous density tensor.

        Arguments:
            discrete_data: A discrete data tensor consisting of '1's and '0's only.

        :returns A continuous density tensor.
        """
        # noinspection PyArgumentList
        discrete_data_size = discrete_data.size()
        layer_count = discrete_data_size[1]

        if layer_count == self.add_operation_count:
            continuous_data = self.add_convolution(discrete_data)
        elif layer_count == self.shift_operation_count:
            continuous_data = self.shift_convolution(discrete_data)
        else:
            raise ValueError("Invalid number of channels in discrete data. We support "
                             + str(self.shift_operation_count) + " or " + str(self.add_operation_count))

        return continuous_data

    def accept_changes(self):
        """
        Accepts the changes that have been made through calls to 'run_shift' and 'run_add'. After this call, future
        calls to 'run_shift' and 'run_add' will see the affect of past calls to 'run_shift' and 'run_add'.
        """
        self.continuous_data_cube = self._stack_data_layers(self.padded_discrete_data,
                                                            self.max_operation_count).unsqueeze(0)
        self.continuous_data_cube = self.run_convolution(self.continuous_data_cube)  # Convert to continuous data

    def init_data(self):
        """
        Initialises the discrete data using non-maximum suppression. Updates the continuous data to match.
        """

        # Start by stacking the target density map nine high. The bottom level has the original density map
        # and each subsequent level has the density map offset by 1 pixels in each direction...
        x_offsets = [0, 0, 1, 1, 1, 0, -1, -1, -1]
        y_offsets = [0, -1, -1, 0, 1, 1, 1, 0, -1]
        density_cube = []
        data_size = self.padded_discrete_data.size()

        for x_offset, y_offset in zip(x_offsets, y_offsets):
            density_cube.append(self.padded_target\
                                    .narrow(0, self.padding + x_offset, data_size[0] - 2 * self.padding)\
                                    .narrow(1, self.padding + y_offset, data_size[1] - 2 * self.padding))
        # noinspection PyTypeChecker
        density_cube = torch.stack(density_cube)

        # Now get the argmax for each column of the cube. Where the argmax is zero, the pixel is surrounded by pixels
        # with smaller values...
        argmax_pixels = torch.argmax(density_cube, dim=0)
        argmax_pixels = argmax_pixels + 1
        argmax_pixels = argmax_pixels.masked_fill(argmax_pixels > 1, 0.)

        # The discrete data should be all zeros, except for a '1' at each pixel that is a local maximum...
        self.padded_discrete_data[self.padding:(data_size[0] - self.padding),
                                  self.padding:(data_size[1] - self.padding)] = argmax_pixels

        # Apply mask to only include points in the areas where the target has values...
        threshold = 0.001
        density_mask = self.padded_target.masked_fill(self.padded_target < threshold, 0.)
        density_mask = density_mask.masked_fill(density_mask > 0., 1.)
        self.padded_discrete_data[:][:] = self.padded_discrete_data * density_mask

        # Transform new discrete data into new continuous data...
        self.continuous_data_cube = self._stack_data_layers(self.padded_discrete_data,
                                                            self.max_operation_count).unsqueeze(0)
        self.continuous_data_cube = self.run_convolution(self.continuous_data_cube)  # Convert to continuous data

    def set_data(self, discrete_data: torch.Tensor, target: torch.Tensor):
        """
        Sets the data to be operated on.

        Arguments:
            discrete_data: A discrete data map (2d tensor with only ones and zeros.)
            target: 'data' will be convolved and compared to this target. The goal is a perfect match.
        """

        # noinspection PyArgumentList
        discrete_data_size = discrete_data.size()
        # noinspection PyArgumentList
        target_data_size = target.size()

        # Zero pad the data...
        padded_data = torch.zeros((discrete_data_size[0] + self.padding * 2,
                                   discrete_data_size[1] + self.padding * 2))
        padded_data[self.padding:(self.padding + discrete_data_size[0]),
                    self.padding:(self.padding + discrete_data_size[1])] = discrete_data
        self.padded_discrete_data = padded_data
        if torch.cuda.is_available():
            self.padded_discrete_data = self.padded_discrete_data.cuda()
        self.continuous_data_cube = self._stack_data_layers(self.padded_discrete_data,
                                                            self.max_operation_count).unsqueeze(0)
        self.continuous_data_cube = self.run_convolution(self.continuous_data_cube)  # Convert to continuous data

        # Do the same with the target...
        target_x_dim = target.dim() - 2
        target_y_dim = target_x_dim + 1
        padded_target = torch.zeros((target_data_size[target_x_dim] + self.padding * 2,
                                     target_data_size[target_y_dim] + self.padding * 2))
        padded_target[self.padding:(self.padding + target_data_size[target_x_dim]),
                      self.padding:(self.padding + target_data_size[target_y_dim])] = target
        self.padded_target = padded_target
        if torch.cuda.is_available():
            self.padded_target = self.padded_target.cuda()
        self.target_cube = self._stack_data_layers(self.padded_target,
                                                   self.max_operation_count).unsqueeze(0)

    def get_discrete_data(self) -> torch.Tensor:
        """
        Gets the discrete data to be operated on. This is a tensor containing only '1's and '0's.
        """
        data_size = self.padded_discrete_data.size()
        data = self.padded_discrete_data\
            .narrow(0, self.padding, data_size[0] - 2 * self.padding)\
            .narrow(1, self.padding, data_size[1] - 2 * self.padding)
        return data

    def get_continuous_data(self) -> torch.Tensor:
        """
        Gets the continuous data created by transforming the discrete data being operated on.
        """
        data_size = self.continuous_data_cube.size()

        # All layers are the same. Just return the first one...
        data = self.continuous_data_cube\
            .narrow(1, 0, 1)\
            .narrow(2, self.padding, data_size[2] - 2 * self.padding)\
            .narrow(3, self.padding, data_size[3] - 2 * self.padding)
        return data.squeeze()

    def _create_operations_cube(self, operations: torch.Tensor) -> list:
        """
        Creates a list of 3D tensors containing a cube version of each operation in 'operations'.
        Each entry in the list corresponds to an operation defined in 'OperationType'.
        Each entry will be a 3D tensor containing that supported operation repeated in the 3rd
        dimension to the same depth as the 'data_cube', e.g. the number of supported operations.
        This is to allow the operation to be performed simultaneously on all layers of the data cube.

        Arguments:
            operations: 3D tensor containing all supported operations, either discrete or continuous. The tensor can be
            viewed as stack of 2d maps containing the changes to individual pixels that the operation entails. For
            example, shifting a pixel to the right will mean the centre pixel has 1 subtracted and the pixel to the
            right has one added.

        Returns a list of 3d Tensors.
        """
        assert operations is not None, "The operations must be defined first."

        operations_cube = []

        # noinspection PyArgumentList
        operation_count = operations.size()[1]

        for operation_id in range(operation_count):
            operation = operations[0][operation_id]
            cube = self._stack_data_layers(operation, operation_count).unsqueeze(0)
            operations_cube.append(cube)

        return operations_cube

    def _create_discrete_operations(self) -> torch.Tensor:
        """
        Creates a 3D tensor containing the discrete version of all supported operations.
        The tensor can be viewed as stack of 2d maps containing the changes to individual pixels that the operation
        entails. For example, shifting a pixel to the right will mean the centre pixel has 1 subtracted and the pixel
        to the right has one added.
        """
        operation = torch.zeros((self.shift_operation_count, self.slice_width, self.slice_width))

        # Move pixel up...
        operation[OperationType.MOVE_N.value][self.zero_y][self.zero_x] = -1.
        operation[OperationType.MOVE_N.value][self.zero_y - 1][self.zero_x] = 1.

        # Move pixel up and right...
        operation[OperationType.MOVE_NE.value][self.zero_y][self.zero_x] = -1.
        operation[OperationType.MOVE_NE.value][self.zero_y - 1][self.zero_x + 1] = 1.

        # Move pixel right...
        operation[OperationType.MOVE_E.value][self.zero_y][self.zero_x] = -1.
        operation[OperationType.MOVE_E.value][self.zero_y][self.zero_x + 1] = 1.

        # Move pixel down and right...
        operation[OperationType.MOVE_SE.value][self.zero_y][self.zero_x] = -1.
        operation[OperationType.MOVE_SE.value][self.zero_y + 1][self.zero_x + 1] = 1.

        # Move pixel down...
        operation[OperationType.MOVE_S.value][self.zero_y][self.zero_x] = -1.
        operation[OperationType.MOVE_S.value][self.zero_y + 1][self.zero_x] = 1.

        # Move pixel down and left...
        operation[OperationType.MOVE_SW.value][self.zero_y][self.zero_x] = -1.
        operation[OperationType.MOVE_SW.value][self.zero_y + 1][self.zero_x - 1] = 1.

        # Move pixel left...
        operation[OperationType.MOVE_W.value][self.zero_y][self.zero_x] = -1.
        operation[OperationType.MOVE_W.value][self.zero_y][self.zero_x - 1] = 1.

        # Move pixel up and left...
        operation[OperationType.MOVE_NW.value][self.zero_y][self.zero_x] = -1.
        operation[OperationType.MOVE_NW.value][self.zero_y - 1][self.zero_x - 1] = 1.

        # Remove the pixel...
        operation[OperationType.REMOVAL.value][self.zero_y][self.zero_x] = -1.

        split_distance = 5

        # Split pixel horizontally...
        operation[OperationType.SPLIT_H.value][self.zero_y][self.zero_x] = -1.
        operation[OperationType.SPLIT_H.value][self.zero_y][self.zero_x - split_distance] = 1.
        operation[OperationType.SPLIT_H.value][self.zero_y][self.zero_x + split_distance] = 1.

        # Split pixel vertically...
        operation[OperationType.SPLIT_V.value][self.zero_y][self.zero_x] = -1.
        operation[OperationType.SPLIT_V.value][self.zero_y - split_distance][self.zero_x] = 1.
        operation[OperationType.SPLIT_V.value][self.zero_y + split_distance][self.zero_x] = 1.

        # Split pixel diagonally ascending...
        operation[OperationType.SPLIT_DA.value][self.zero_y][self.zero_x] = -1.
        operation[OperationType.SPLIT_DA.value][self.zero_y + split_distance][self.zero_x - split_distance] = 1.
        operation[OperationType.SPLIT_DA.value][self.zero_y - split_distance][self.zero_x + split_distance] = 1.

        # Split pixel diagonally descending...
        operation[OperationType.SPLIT_DD.value][self.zero_y][self.zero_x] = -1.
        operation[OperationType.SPLIT_DD.value][self.zero_y - split_distance][self.zero_x - split_distance] = 1.
        operation[OperationType.SPLIT_DD.value][self.zero_y + split_distance][self.zero_x + split_distance] = 1.

        if torch.cuda.is_available():
            operation = operation.cuda()

        return operation

    def _create_discrete_add_operation(self) -> torch.Tensor:
        """
        Creates a 3D tensor containing the discrete version of the add operation along with a no-add alternative.
        The tensor can be viewed as stack of 2d maps containing the changes to individual pixels that the operation
        entails. The 'add' operation would increase the value of the origin pixel by '1'.
        """
        operation = torch.zeros((2, self.slice_width, self.slice_width))

        # The first row is untouched, as it is the 'do nothing' option. The second row is the add operation...
        operation[1][self.zero_y][self.zero_x] = 1.

        if torch.cuda.is_available():
            operation = operation.cuda()

        return operation

    def _create_continuous_operations(self, discrete_operations: torch.Tensor) -> torch.Tensor:
        """
        Creates a 3D tensor containing the continuous version of a set of discrete operations.
        The tensor can be viewed as stack of 2d maps containing the changes to individual pixels that the operation
        entails. For example, shifting a pixel to the right would look like subtracting one from the centre pixel and
        adding one to the pixel to the right, followed by running the data through a Gaussian convolution.
        """
        operation = discrete_operations.clone()

        # noinspection PyArgumentList
        operation_count = operation.size()[0]

        # Create a batch dimension of size one...
        operation = operation.unsqueeze(0)

        # Run the discrete data through the convolution...
        operation = self.run_convolution(operation)

        if self.diagnostic_mode:
            plt.gray()
            for i in range(operation_count):
                layer = discrete_operations[i]
                plt.imsave(self.output_dir + "/operations/discrete_operation_" + str(i) + ".png",
                           layer.detach().cpu().numpy().squeeze())

                layer = operation[0][i]
                plt.imsave(self.output_dir + "/operations/continuous_operation_" + str(i) + ".png",
                           layer.detach().cpu().numpy().squeeze())

        if self.output_effect_tensor_graphs:
            # for i in range(self.operation_count):
            #     layer = operation[0][i]
            #
            #     from mpl_toolkits.mplot3d import Axes3D
            #     from matplotlib import cm
            #     import numpy as np
            #     fig = plt.figure()
            #     ax = fig.gca(projection='3d')
            #     x = np.arange(0, layer.size()[0], 1)
            #     y = np.arange(0, layer.size()[1], 1)
            #     x, y = np.meshgrid(x, y)
            #     z = layer.detach().cpu().numpy().squeeze()
            #     surf = ax.plot_surface(x, y, z, cmap=cm.coolwarm,
            #                            linewidth=0, antialiased=False)
            #     plt.show()

            for i in range(operation_count):
                layer = discrete_operations[i]

                import numpy as np
                x = np.arange(0, layer.size()[0], 1)
                y = [layer[x_iter].max() for x_iter in x]
                plt.plot(x, y)
                plt.show()

        if torch.cuda.is_available():
            operation = operation.cuda()

        return operation

    def _stack_data_layers(self, layer: torch.Tensor, layer_count: int) -> torch.Tensor:
        """
        Takes a 2D tensor and stacks copies of it with 'layer_count' layers.

        Arguments:
            layer: A 2d tensor to be stacked.
        """
        layer = layer.unsqueeze(0)
        data_cube = []
        for _ in range(layer_count):
            data_cube.append(layer)
        # noinspection PyTypeChecker
        data_cube = torch.cat(data_cube, dim=0)
        return data_cube

    def run_shift(self, x: int, y: int) -> OperationType:
        """
        Takes the prospective 2d discrete map (a tensor with values 0 and 1 only) and finds the shift operation that
        will result in the best improvement to the loss after the map is convolved and compared with the target tensor.
        That operation will be automatically applied and the map returned.

        The discrete map and the target tensor must have already been set by a call to 'set_data'. The changed data
        after this call can be accessed with 'get_discrete_data'.

        Shift operations are operations that operate on already existing points. They include moving, splitting and
        removing the points.

        Arguments:
            x: The x position at which the operation is being performed.
            y: The y position at which the operation is being performed.

        :returns the operation performed.
        """
        assert self.padded_discrete_data is not None, "Data must have been set with 'set_data'"
        assert self.continuous_data_cube is not None, "Target must have been set with 'set_data'"
        assert self.target_cube is not None, "Target must have been set with 'set_data'"

        # Start by preparing a slice of the data at the correct location. 'data_slice' contains the discrete data being
        # operated on and must be cut down to size. 'continuous_data_cube_slice' contains the a copy of the continuous
        # data on each layer. There should be one layer for each supported operation. Since there could be fewer shift
        # operations than the height of the cube, the cube must be reduced in height. Leaving the extra layers
        # untouched is not an issue.
        #
        # The data is already padded by half the kernel size, so using the coordinates passed in will refer to the top
        # left edge of a square centred on the coordinates indicated...
        slice_x_start = x + self.padding - self.slice_width // 2
        slice_y_start = y + self.padding - self.slice_width // 2
        data_slice = self.padded_discrete_data\
            .narrow(0, slice_x_start, self.slice_width)\
            .narrow(1, slice_y_start, self.slice_width)
        continuous_data_cube_slice = self.continuous_data_cube\
            .narrow(1, 0, self.shift_operation_count)\
            .narrow(2, slice_x_start, self.slice_width)\
            .narrow(3, slice_y_start, self.slice_width)
        target_cube_slice = self.target_cube\
            .narrow(1, 0, self.shift_operation_count)\
            .narrow(2, slice_x_start, self.slice_width)\
            .narrow(3, slice_y_start, self.slice_width)

        # Apply prospective operations to the data...
        continuous_data_cube_slice = continuous_data_cube_slice + self.continuous_shift_operations

        # Apply loss against target...
        loss_cube = torch.abs(continuous_data_cube_slice - target_cube_slice)
        losses_per_operation = loss_cube.sum(3).sum(2)[0]
        # noinspection PyTypeChecker
        best_operation_index = int(torch.min(losses_per_operation, 0)[1])

        if best_operation_index != 0:
            # Now perform the best operation on the in-place data...
            data_slice[:][:] = data_slice + self.discrete_shift_operations[best_operation_index]

            # Do the same with the working 3d discrete data...
            continuous_data_cube_slice = self.continuous_data_cube\
                .narrow(1, 0, self.shift_operation_count)\
                .narrow(2, slice_x_start, self.slice_width)\
                .narrow(3, slice_y_start, self.slice_width)
            continuous_data_cube_slice[:][:][:] = continuous_data_cube_slice + \
                self.continuous_shift_operations_cube[best_operation_index]

        return OperationType(best_operation_index)

    def try_shift(self, x: int, y: int) -> OperationType:
        """
        Takes the prospective 2d discrete map (a tensor with values 0 and 1 only) and finds the shift operation that
        will result in the best improvement to the loss after the map is convolved and compared with the target tensor.

        The discrete map and the target tensor must have already been set by a call to 'set_data'. The changed data
        after this call can be accessed with 'get_discrete_data'.

        Shift operations are operations that operate on already existing points. They include moving, splitting and
        removing the points.

        Arguments:
            x: The x position at which the operation is being performed.
            y: The y position at which the operation is being performed.

        :returns the operation performed.
        """
        assert self.padded_discrete_data is not None, "Data must have been set with 'set_data'"
        assert self.continuous_data_cube is not None, "Target must have been set with 'set_data'"
        assert self.target_cube is not None, "Target must have been set with 'set_data'"

        # Start by preparing a slice of the data at the correct location. 'data_slice' contains the discrete data being
        # operated on and must be cut down to size. 'continuous_data_cube_slice' contains the a copy of the continuous
        # data on each layer. There should be one layer for each supported operation. Since there could be fewer shift
        # operations than the height of the cube, the cube must be reduced in height. Leaving the extra layers
        # untouched is not an issue.
        #
        # The data is already padded by half the kernel size, so using the coordinates passed in will refer to the top
        # left edge of a square centred on the coordinates indicated...
        slice_x_start = x + self.padding - self.slice_width // 2
        slice_y_start = y + self.padding - self.slice_width // 2
        data_slice = self.padded_discrete_data\
            .narrow(0, slice_x_start, self.slice_width)\
            .narrow(1, slice_y_start, self.slice_width)
        continuous_data_cube_slice = self.continuous_data_cube\
            .narrow(1, 0, self.shift_operation_count)\
            .narrow(2, slice_x_start, self.slice_width)\
            .narrow(3, slice_y_start, self.slice_width)
        target_cube_slice = self.target_cube\
            .narrow(1, 0, self.shift_operation_count)\
            .narrow(2, slice_x_start, self.slice_width)\
            .narrow(3, slice_y_start, self.slice_width)

        # Apply prospective operations to the data...
        continuous_data_cube_slice = continuous_data_cube_slice + self.continuous_shift_operations

        # Apply loss against target...
        loss_cube = torch.abs(continuous_data_cube_slice - target_cube_slice)
        losses_per_operation = loss_cube.sum(3).sum(2)[0]
        # noinspection PyTypeChecker
        best_operation_index = int(torch.min(losses_per_operation, 0)[1])

        return OperationType(best_operation_index)

    def run_add(self, x: int, y: int) -> bool:
        """
        Takes the prospective 2d discrete map (a tensor with values 0 and 1 only) and evaluates whether an add operation
        at the given location will result in an improvement to the loss after the map is convolved and compared with the
        target tensor. If so, the add operation will be automatically applied and the map returned.

        The discrete map and the target tensor must have already been set by a call to 'set_data'. The changed data
        after this call can be accessed with 'get_discrete_data'.

        Arguments:
            x: The x position at which the operation is being performed.
            y: The y position at which the operation is being performed.

        :returns True if the add was performed.
        """
        assert self.padded_discrete_data is not None, "Data must have been set with 'set_data'"
        assert self.continuous_data_cube is not None, "Target must have been set with 'set_data'"
        assert self.target_cube is not None, "Target must have been set with 'set_data'"

        # Start by preparing a slice of the data at the correct location. 'data_slice' contains the discrete data being
        # operated on and must be cut down to size. 'continuous_data_cube_slice' contains the a copy of the discrete
        # data on each layer. There should be one layer for each supported operation. Since there could be fewer add
        # operations than the height of the cube, the cube must be reduced in height. Leaving the extra layers
        # untouched is not an issue.
        #
        # The data is already padded by half the kernel size, so using the coordinates passed in will refer to the top
        # left edge of a square centred on the coordinates indicated...
        slice_x_start = x + self.padding - self.slice_width // 2
        slice_y_start = y + self.padding - self.slice_width // 2
        data_slice = self.padded_discrete_data\
            .narrow(0, slice_x_start, self.slice_width)\
            .narrow(1, slice_y_start, self.slice_width)
        continuous_data_cube_slice = self.continuous_data_cube\
            .narrow(1, 0, self.add_operation_count)\
            .narrow(2, slice_x_start, self.slice_width)\
            .narrow(3, slice_y_start, self.slice_width)
        target_cube_slice = self.target_cube\
            .narrow(1, 0, self.add_operation_count)\
            .narrow(2, slice_x_start, self.slice_width)\
            .narrow(3, slice_y_start, self.slice_width)

        # Apply prospective operations to the data...
        continuous_data_cube_slice = continuous_data_cube_slice + self.continuous_add_operations

        # Apply loss against target...
        loss_cube = torch.abs(continuous_data_cube_slice - target_cube_slice)
        losses_per_operation = loss_cube.sum(3).sum(2)[0]
        # noinspection PyTypeChecker
        best_operation_index = int(torch.min(losses_per_operation, 0)[1])

        if best_operation_index != 0:
            # Now perform the best operation on the in-place data...
            data_slice[:][:] = data_slice + self.discrete_add_operations[best_operation_index]

            # Do the same with the working 3d data...
            continuous_data_cube_slice = self.continuous_data_cube\
                .narrow(1, 0, self.add_operation_count)\
                .narrow(2, slice_x_start, self.slice_width)\
                .narrow(3, slice_y_start, self.slice_width)
            continuous_data_cube_slice[:][:][:] = continuous_data_cube_slice + \
                self.continuous_add_operations_cube[best_operation_index]

        return best_operation_index == 1
