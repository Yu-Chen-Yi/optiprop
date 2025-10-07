import torch
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import rcParams
from skimage.measure import label, regionprops
from scipy.ndimage import maximum_position
import numpy as np

def cart_grid(
    grid_size: list, 
    grid_dx: list, 
    dtype: torch.dtype, 
    device: torch.device
    )->tuple[torch.Tensor, torch.Tensor]:
    """
    Args:
        grid_size (list): Grid size [Nx, Ny] (unit: #)
        grid_dx (list): Grid pixel size [dx, dy] (unit: m)
        dtype (torch.dtype): 'torch.float32' or 'torch.float64'
        device (torch.device): 'cpu' or 'cuda'
    Outputs:
        x (torch.Tensor): x-coordinate grid (unit: m)
        y (torch.Tensor): y-coordinate grid (unit: m)
    """
    x, y = torch.meshgrid(
        torch.arange(0, grid_size[-1], dtype=dtype),
        torch.arange(0, grid_size[-2], dtype=dtype),
        indexing="xy",
    )
    x = x - (x.shape[-1] - 1) / 2
    y = y - (y.shape[-2] - 1) / 2

    x = x * grid_dx[-1]
    y = y * grid_dx[-2]
    return x.to(device=device), y.to(device=device)

def get_grid_size(
    length: float, 
    width: float, 
    pixel_size: float
    )->tuple[int, int]:
    """ 
    Args:
        length (float): Length of the complex amplitude (unit: m)
        width (float): Width of the complex amplitude (unit: m)
        pixel_size (float): Size of each pixel (unit: m)
    Outputs:
        Nx (int): Adjusted number of pixels along the length (odd number)
        Ny (int): Adjusted number of pixels along the width (odd number)
    """
    Nx = int(length / pixel_size)
    Ny = int(width / pixel_size)

    if Nx % 2 == 0:
        Nx += 1
    if Ny % 2 == 0:
        Ny += 1

    return Nx, Ny

def pad_to_center(
    input_Uin: torch.Tensor, 
    N: int
    )->torch.Tensor:
    """
    Pad the input tensor to be centered in an N x N matrix.

    Parameters:
        input_Uin (torch.Tensor): Input tensor with dimensions (H, W).
        N (int): The size of the square zeros matrix (N x N).

    Returns:
        torch.Tensor: The resulting matrix with input_Uin centered.
    """
    H, W = input_Uin.shape

    # Ensure N is larger than both dimensions of input_Uin
    if N < max(H, W):
        raise ValueError("N must be larger than both dimensions of input_Uin.")

    # Compute the padding dimensions
    pad_top = (N - H) // 2
    pad_bottom = N - H - pad_top
    pad_left = (N - W) // 2
    pad_right = N - W - pad_left

    # Pad the input tensor with zeros to center it
    padded_tensor = torch.nn.functional.pad(input_Uin, (pad_left, pad_right, pad_top, pad_bottom), mode='constant', value=0)

    return padded_tensor

def check_available_cuda(device):
    """ Check if CUDA is available. """
    if device == 'cuda' and not torch.cuda.is_available():
        print("CUDA is not available. Using CPU instead.")
        device = 'cpu'
    return device

def extract_peak_points(
    I: torch.Tensor, 
    threshold: float = 0.2
    ) -> tuple[list, list, list]:
    """
    Extract peak points from intensity distribution I.
    
    Args:
        I (torch.Tensor): Intensity distribution (2D tensor)
        threshold (float): Binary threshold for separating bright regions
        
    Returns:
        tuple: (cx, cy, intensity) - coordinates and values of peak points
    """
    if I.ndim != 2:
        raise ValueError("Input tensor I must be 2D.")

    binary_img = (I > threshold).numpy()
    labeled = label(binary_img)

    cx = []
    cy = []
    intensity = []
    for region in regionprops(labeled):
        minr, minc, maxr, maxc = region.bbox
        sub_img = I[minr:maxr, minc:maxc]
        if sub_img.numel() == 0:
            continue
        max_val = sub_img.max().item()
        max_pos = maximum_position(sub_img.numpy())
        row_idx = minr + max_pos[0]
        col_idx = minc + max_pos[1]
        cx.append(row_idx)
        cy.append(col_idx)
        intensity.append(max_val)

    return cx, cy, intensity

def make_colorbar_with_padding(ax):
    """
    Create colorbar axis that fits the size of a plot - detailed here: http://chris35wills.github.io/matplotlib_axis/
    """
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    return(cax) 

def plot_field_amplitude_phase(
    U0: torch.Tensor, 
    X: torch.Tensor, 
    Y: torch.Tensor, 
    cmap: str = 'turbo', 
    fontsize: int = 16, 
    xlim: list = None, 
    ylim: list = None, 
    selected_field: str = 'all', 
    dark_style: bool = False,
    show: bool = True
    ):
    """
    Plot the field, amplitude, and phase.
    Args:
        U0 (torch.Tensor): The field to plot.
        X (torch.Tensor): The x-coordinate of the field.
        Y (torch.Tensor): The y-coordinate of the field.
        cmap (str): The colormap to use. 'turbo', 'viridis', 'plasma', 'inferno', 'magma'
        fontsize (int): The font size.
        xlim (list): The x-axis limits. [xmin, xmax] Unit: µm
        ylim (list): The y-axis limits. [ymin, ymax] Unit: µm
        selected_field (str): The field to plot. 'all', 'field', 'amplitude', 'phase'
        dark_style (bool): The style of the plot. 'True' for dark style, 'False' for light style.
        show (bool): Whether to show the plot. 'True' for show, 'False' for not show.
    """
    x = X.cpu().numpy() * 1e6
    y = Y.cpu().numpy() * 1e6
    field = torch.real(U0).cpu().numpy()
    amplitude = torch.abs(U0).cpu().numpy()
    phase_unwrapped = torch.angle(U0)
    phase = (phase_unwrapped % (2 * np.pi)).cpu().numpy()

    # Set dark style
    if dark_style:
        plt.style.use('dark_background')
    else:
        plt.style.use('default')
    FONT_FAMILY = 'serif'
    rcParams['font.weight'] = 'bold'
    rcParams['font.size'] = fontsize
    rcParams['font.family'] = FONT_FAMILY
    if selected_field == 'all':
        fig = plt.figure("Complex field, Amplitude, and Phase", figsize=(24, 8))
        # Subplot for Field
        ax1 = fig.add_subplot(131)
        plt.imshow(field, extent=[x.min(), x.max(), y.min(), y.max()], cmap=cmap, origin='lower', aspect='equal')
        plt.xlabel('x (µm)', fontsize=fontsize, fontweight='bold', fontname=FONT_FAMILY)
        plt.ylabel('y (µm)', fontsize=fontsize, fontweight='bold', fontname=FONT_FAMILY)
        plt.title("Field", fontsize=fontsize, fontweight='bold', fontname=FONT_FAMILY)
        plt.xlim(xlim)
        plt.ylim(ylim)
        cax1 = make_colorbar_with_padding(ax1)
        cb1 = plt.colorbar(cax=cax1)
        
        fig.subplots_adjust(right=0.9)
        # Subplot for Amplitude
        ax2 = fig.add_subplot(132)
        plt.imshow(amplitude, extent=[x.min(), x.max(), y.min(), y.max()], cmap=cmap, origin='lower', aspect='equal')
        plt.xlabel('x (µm)', fontsize=fontsize, fontweight='bold', fontname=FONT_FAMILY)
        plt.ylabel('y (µm)', fontsize=fontsize, fontweight='bold', fontname=FONT_FAMILY)
        plt.title("Amplitude", fontsize=fontsize, fontweight='bold', fontname=FONT_FAMILY)
        plt.xlim(xlim)
        plt.ylim(ylim)
        cax2 = make_colorbar_with_padding(ax2)
        cb2 = plt.colorbar(cax=cax2)
        

        # Subplot for Phase
        ax3 = fig.add_subplot(133)
        plt.imshow(phase, extent=[x.min(), x.max(), y.min(), y.max()], cmap=cmap, origin='lower', aspect='equal')
        plt.xlabel('x (µm)', fontsize=fontsize, fontweight='bold', fontname=FONT_FAMILY)
        plt.ylabel('y (µm)', fontsize=fontsize, fontweight='bold', fontname=FONT_FAMILY)
        plt.title("Phase", fontsize=fontsize, fontweight='bold', fontname=FONT_FAMILY)
        plt.xlim(xlim)
        plt.ylim(ylim)
        plt.clim([0, 2 * np.pi]) #correct
        cax3 = make_colorbar_with_padding(ax3)
        cb3 = plt.colorbar(cax=cax3)

    elif selected_field.lower() == 'field':
        fig = plt.figure("Complex field", figsize=(8, 8))
        # Subplot for Field
        ax1 = fig.add_subplot(111)
        plt.imshow(field, extent=[x.min(), x.max(), y.min(), y.max()], cmap=cmap, origin='lower', aspect='equal')
        plt.xlabel('x (µm)', fontsize=fontsize, fontweight='bold', fontname=FONT_FAMILY)
        plt.ylabel('y (µm)', fontsize=fontsize, fontweight='bold', fontname=FONT_FAMILY)
        plt.title("Field", fontsize=fontsize, fontweight='bold', fontname=FONT_FAMILY)
        plt.xlim(xlim)
        plt.ylim(ylim)
        cax1 = make_colorbar_with_padding(ax1)
        cb1 = plt.colorbar(cax=cax1)

    elif selected_field.lower() == 'amplitude':
        fig = plt.figure("Amplitude", figsize=(8, 8))
        # Subplot for Amplitude
        ax2 = fig.add_subplot(111)
        plt.imshow(amplitude, extent=[x.min(), x.max(), y.min(), y.max()], cmap=cmap, origin='lower', aspect='equal')
        plt.xlabel('x (µm)', fontsize=fontsize, fontweight='bold', fontname=FONT_FAMILY)
        plt.ylabel('y (µm)', fontsize=fontsize, fontweight='bold', fontname=FONT_FAMILY)
        plt.title("Amplitude", fontsize=fontsize, fontweight='bold', fontname=FONT_FAMILY)
        plt.xlim(xlim)
        plt.ylim(ylim)
        cax2 = make_colorbar_with_padding(ax2)
        cb2 = plt.colorbar(cax=cax2)

    elif selected_field.lower() == 'phase':
        fig = plt.figure("Phase", figsize=(8, 8))
        # Subplot for Phase
        ax3 = fig.add_subplot(111)
        plt.imshow(
            phase, 
            extent=[x.min(), x.max(), y.min(), y.max()], 
            cmap=cmap, 
            origin='lower', 
            aspect='equal',
            )
        plt.xlabel('x (µm)', fontsize=fontsize, fontweight='bold', fontname=FONT_FAMILY)
        plt.ylabel('y (µm)', fontsize=fontsize, fontweight='bold', fontname=FONT_FAMILY)
        plt.title("Phase", fontsize=fontsize, fontweight='bold', fontname=FONT_FAMILY)
        plt.xlim(xlim)
        plt.ylim(ylim)
        plt.clim([0, 2 * np.pi]) #correct
        cax3 = make_colorbar_with_padding(ax3)
        cb3 = plt.colorbar(cax=cax3)
    plt.tight_layout()
    if show:
        plt.show()

def plot_field_intensity(
    U0: torch.Tensor,
    X: torch.Tensor, 
    Y: torch.Tensor, 
    cmap: str = 'turbo', 
    fontsize: int = 16, 
    xlim: list = None, 
    ylim: list = None, 
    clim: list = None, 
    unit: str = 'µm', 
    norm: str = 'linear', 
    dark_style: bool = False,
    title: str = None,
    show: bool = True
    ):
    """
    Plot the intensity.
    Args:
        U0 (torch.Tensor): The field to plot.
        X (torch.Tensor): The x-coordinate of the field.
        Y (torch.Tensor): The y-coordinate of the field.
        cmap (str): The colormap to use.
        fontsize (int): The font size.
        xlim (list): The x-axis limits. [xmin, xmax] Unit: µm
        ylim (list): The y-axis limits. [ymin, ymax] Unit: µm
        clim (list): The color limits. [cmin, cmax] Unit: arbitrary
        unit (str): The unit of the x and y axes. 'µm', 'mm', 'm'
        norm (str): The normalization of the intensity. 'linear', 'log'
        dark_style (bool): The style of the plot. 'True' for dark style, 'False' for light style.
        title (str): The title of the plot.
        show (bool): Whether to show the plot. 'True' for show, 'False' for not show.
    """
    x = X.cpu().numpy() * 1e6
    y = Y.cpu().numpy() * 1e6
    intensity = (torch.abs(U0)**2).cpu().numpy()
    if unit == 'µm' or unit == 'um':
        xlabel_name = 'x (µm)'
        ylabel_name = 'y (µm)'
    elif unit == 'mm' or unit == 'millimeter':
        x = x * 1e-3
        y = y * 1e-3
        xlabel_name = 'x (mm)'
        ylabel_name = 'y (mm)'
    elif unit == 'm' or unit == 'meter':
        x = x * 1e-6
        y = y * 1e-6
        xlabel_name = 'x (m)'
        ylabel_name = 'y (m)'

    # Set dark style
    if dark_style:
        plt.style.use('dark_background')
    else:
        plt.style.use('default')
    FONT_FAMILY = 'Times New Roman'
    rcParams['font.weight'] = 'bold'
    rcParams['font.size'] = fontsize
    rcParams['font.family'] = FONT_FAMILY
    
    fig = plt.figure("Complex field, Amplitude, and Phase", figsize=(8, 8))

    # Subplot for Field
    ax1 = fig.add_subplot(111)
    plt.imshow(intensity, extent=[x.min(), x.max(), y.min(), y.max()], cmap=cmap, origin='lower', aspect='equal', norm=norm)
    plt.xlabel(xlabel_name, fontsize=fontsize, fontweight='bold', fontname=FONT_FAMILY)
    plt.ylabel(ylabel_name, fontsize=fontsize, fontweight='bold', fontname=FONT_FAMILY)
    if title is not None:
        plt.title(title, fontsize=fontsize, fontweight='bold', fontname=FONT_FAMILY)
    else:
        plt.title("Intensity", fontsize=fontsize, fontweight='bold', fontname=FONT_FAMILY)
    plt.xlim(xlim)
    plt.ylim(ylim)
    cax1 = make_colorbar_with_padding(ax1)
    # set colorbar is logscaled
    cb1 = plt.colorbar(cax=cax1)
    plt.clim(clim)
    plt.tight_layout()
    if show:
        plt.show()

def plot_xz_field_intensity(
    U0: torch.Tensor, 
    X: torch.Tensor, 
    Z: torch.Tensor, 
    cmap: str = 'turbo', 
    fontsize: int = 16, 
    xlim: list = None, 
    ylim: list = None, 
    clim: list = None, 
    unit: str = 'µm', 
    norm: str = 'linear', 
    dark_style: bool = False, 
    title: str = None, 
    show: bool = True
    ):
    """
    Plot the intensity.
    Args:
        U0 (torch.Tensor): The field to plot.
        X (torch.Tensor): The x-coordinate of the field.
        Z (torch.Tensor): The z-coordinate of the field.
        cmap (str): The colormap to use.
        fontsize (int): The font size.
        xlim (list): The x-axis limits. [xmin, xmax] Unit: µm
        zlim (list): The z-axis limits. [zmin, zmax] Unit: µm
        clim (list): The color limits. [cmin, cmax] Unit: arbitrary
        unit (str): The unit of the x and z axes. 'µm', 'mm', 'm'
        norm (str): The normalization of the intensity. 'linear', 'log'
        dark_style (bool): The style of the plot. 'True' for dark style, 'False' for light style.
        title (str): The title of the plot.
        show (bool): Whether to show the plot. 'True' for show, 'False' for not show.
    """
    x = X.cpu().numpy() * 1e6
    z = Z.cpu().numpy() * 1e6
    intensity = torch.transpose(torch.abs(U0)**2, 0, 1).cpu().numpy()

    if unit == 'µm' or unit == 'um':
        xlabel_name = 'x (µm)'
        zlabel_name = 'z (µm)'
    elif unit == 'mm' or unit == 'millimeter':
        x = x * 1e-3
        z = z * 1e-3
        xlabel_name = 'x (mm)'
        zlabel_name = 'z (mm)'
    elif unit == 'm' or unit == 'meter':
        x = x * 1e-6
        z = z * 1e-6
        xlabel_name = 'x (m)'
        zlabel_name = 'z (m)'

    # Set dark style
    if dark_style:
        plt.style.use('dark_background')
    else:
        plt.style.use('default')
    FONT_FAMILY = 'Times New Roman'
    rcParams['font.weight'] = 'bold'
    rcParams['font.size'] = fontsize
    rcParams['font.family'] = FONT_FAMILY
    
    fig = plt.figure("Complex field, Amplitude, and Phase", figsize=(24, 8))

    # Subplot for Field
    ax1 = fig.add_subplot(111)
    plt.imshow(
            intensity, 
            extent=[z.min(), z.max(), x.min(), x.max()], 
            cmap=cmap, 
            origin='lower', 
            norm=norm
            )
    plt.xlabel(zlabel_name, fontsize=fontsize, fontweight='bold', fontname=FONT_FAMILY)
    plt.ylabel(xlabel_name, fontsize=fontsize, fontweight='bold', fontname=FONT_FAMILY)
    if title is not None:
        plt.title(title, fontsize=fontsize, fontweight='bold', fontname=FONT_FAMILY)
    else:
        plt.title("Intensity", fontsize=fontsize, fontweight='bold', fontname=FONT_FAMILY)
    plt.xlim(xlim)
    plt.ylim(ylim)
    cax1 = make_colorbar_with_padding(ax1)
    # set colorbar is logscaled
    cb1 = plt.colorbar(cax=cax1)
    plt.clim(clim) #correct
    if show:
        plt.show()
