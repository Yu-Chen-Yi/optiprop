import torch
from .utils import *
import matplotlib.pyplot as plt
from .elements import NearField
from rich.table import Table
from rich.console import Console
import numpy as np


class FresnelPropagation:
    def __init__(
        self, 
        propagation_wavelength: float = 0.94e-6, 
        propagation_distance: float = 1000e-6,
        n: float= 1.0,
        device: str = 'cpu',
        dtype: torch.dtype = torch.float32,
        ):
        """
        Initialize Fresnel propagation.
        
        Args:
            propagation_wavelength (float): Propagation wavelength (unit: m)
            propagation_distance (float): Propagation distance (unit: m)
            n (float): Refractive index
            device (str): 'cpu' or 'cuda' for GPU
            dtype (torch.dtype): Data type for computation
        """
        self.propagation_wavelength = propagation_wavelength/n
        self.propagation_distance = propagation_distance
        self.device = check_available_cuda(device=device)        
        self.dtype = dtype
        self.n = n
        # Calculate wave number k
        self.k = 2 * torch.pi / self.propagation_wavelength

    def set_input_field(
        self, 
        u_in: torch.Tensor, 
        pixel_size: float
        ):
        """
        Set input field for propagation.
        
        Args:
            u_in (torch.Tensor): Input complex field (shape: [Nx, Ny])
            pixel_size (float): Pixel size (unit: m)
        """
        if not torch.is_tensor(u_in):
            raise ValueError("Input u_in must be a PyTorch tensor.")
        
        # Check if it's a complex tensor
        if not torch.is_complex(u_in):
            if u_in.ndim == 2:
                # If it's a real tensor, convert to complex tensor with zero imaginary part
                u_in = u_in + 1j * torch.zeros_like(u_in)
            else:
                raise ValueError("Tensor shape is not compatible for complex conversion.")
            
        if self.dtype == torch.float32:
            u_in = u_in.to(dtype=torch.complex64)
        elif self.dtype == torch.float64:
            u_in = u_in.to(dtype=torch.complex128)
        # Transfer tensor to corresponding device
        self.input_U = u_in.to(self.device)
        self.input_Nx, self.input_Ny = u_in.shape
        self.input_pixel_size = pixel_size
        self.input_X, self.input_Y = cart_grid(
            grid_size=[self.input_Nx, self.input_Ny], 
            grid_dx=[self.input_pixel_size, self.input_pixel_size], 
            dtype=self.dtype, 
            device=self.device
            )
        self.output_U = torch.zeros((self.input_Nx, self.input_Ny), dtype=self.dtype, device=self.device)
        self.output_X = self.propagation_wavelength * self.propagation_distance * torch.fft.fftfreq(self.input_Nx, device=self.device) / (self.input_pixel_size)
        self.output_Y = self.propagation_wavelength * self.propagation_distance * torch.fft.fftfreq(self.input_Ny, device=self.device) / (self.input_pixel_size)
        self.output_pixel_size = self.propagation_wavelength  * self.propagation_distance/ (self.input_pixel_size)

    def set_output_field(
        self, 
        output_pixel_size: float
        ):
        """
        Set output field parameters.
        
        Args:
            output_pixel_size (float): Output pixel size (unit: m)
        """
        self.output_pixel_size = output_pixel_size
        # Calculate the u_in padding size
        padding_size = int(self.propagation_wavelength * self.propagation_distance / self.input_pixel_size / self.output_pixel_size)
        if padding_size > self.input_Nx:
            self.input_U = pad_to_center(self.input_U, padding_size)
            # Transfer tensor to corresponding device
            self.input_Nx, self.input_Ny = padding_size, padding_size
            self.input_X, self.input_Y = self.X, self.Y = cart_grid(
                grid_size=[self.input_Nx, self.input_Ny], 
                grid_dx=[self.input_pixel_size, self.input_pixel_size], 
                dtype=self.dtype, 
                device=self.device
                )
        self.output_X = self.propagation_wavelength * self.propagation_distance * torch.fft.fftfreq(self.input_Nx, device=self.device) / (self.input_pixel_size)
        self.output_Y = self.propagation_wavelength * self.propagation_distance * torch.fft.fftfreq(self.input_Ny, device=self.device) / (self.input_pixel_size)
    
    def propagate(self):
        """Execute Fresnel propagation."""
        self.output_U = self._fresnel()       

    def _fresnel(self):
        """
        Fresnel propagation calculation.
        
        Returns:
            torch.Tensor: Output complex field
        """
        z = torch.tensor(self.propagation_distance, dtype=self.dtype, device=self.device)
        wvl = torch.tensor(self.propagation_wavelength, dtype=self.dtype, device=self.device)
        k = self.k
        self.output_X = wvl * z * torch.fft.fftfreq(self.input_Nx, device=self.device) / (self.input_pixel_size)
        self.output_Y = wvl * z * torch.fft.fftfreq(self.input_Ny, device=self.device) / (self.input_pixel_size)
        # Input phase (pre-phase)
        in_phase = torch.exp(1j * k / (2*z) * (self.input_X**2 + self.input_Y**2))
        U_in_mod = self.input_U * in_phase

        # Spatial domain -> Frequency domain
        U_in_freq = torch.fft.fft2(U_in_mod)

        # Output phase and amplitude constant (post-phase)
        out_phase = torch.exp(1j*k*z) / (1j*wvl*z) * torch.exp(1j*k/(2*z)*(self.output_X**2 + self.output_Y**2))
        output_U = torch.fft.fftshift(U_in_freq)*out_phase*self.input_pixel_size**2
        
        return output_U

    def propagate_xz(
        self, 
        z_range: list = np.arange(1000e-6, 2000e-6, 50e-6), 
        output_pixel_size: float = None
        ):
        """
        Scan Fresnel propagation at different distances.
        
        Args:
            z_range (list or array): Propagation distance range (unit: m)
            output_pixel_size (float): Output pixel size, use input pixel size if None
        Returns:
            torch.Tensor: Output field with shape [len(z_range), Nx]
        """
        z_num = len(z_range)
        self.z_range = z_range
        # Initialize output field array
        if self.dtype == torch.float32:
            self.output_UZ = torch.zeros((z_num, self.input_Nx), 
                               dtype=torch.complex64, device=self.device)
        else:
            self.output_UZ = torch.zeros((z_num, self.input_Nx), 
                               dtype=torch.complex128, device=self.device)
        self.output_X = torch.zeros((z_num, self.input_Nx), 
                               dtype=self.dtype, device=self.device)
        self.output_Z = torch.zeros((z_num, self.input_Nx), 
                               dtype=self.dtype, device=self.device)
        for i, z in enumerate(z_range):
            # Set current propagation distance
            self.propagation_distance = z
            
            # Calculate Fresnel propagation
            z_tensor = torch.tensor(z)
            wvl = torch.tensor(self.propagation_wavelength)
            k = self.k
            
            # Calculate output coordinates (x direction only)
            output_X = wvl * z_tensor * torch.fft.fftfreq(self.input_Nx, device=self.device) / (self.input_pixel_size)
            
            # Input phase (pre-phase)
            in_phase = torch.exp(1j * k / (2*z_tensor) * (self.input_X**2 + self.input_Y**2))
            U_in_mod = self.input_U * in_phase

            # Spatial domain -> Frequency domain
            U_in_freq = torch.fft.fft2(U_in_mod)

            # Output phase and amplitude constant (post-phase)
            out_phase = torch.exp(1j*k*z_tensor) / (1j*wvl*z_tensor) * torch.exp(1j*k/(2*z_tensor)*(output_X**2))
            output_U = torch.fft.fftshift(U_in_freq) * out_phase * self.input_pixel_size**2
            
            # Take center line (y = 0)
            center_idx = self.input_Ny // 2
            self.output_UZ[i, :] = output_U[center_idx, :]
            self.output_X[i, :] = output_X
            # Repeat z_tensor to shape (1, Nx)
            self.output_Z[i, :] = z_tensor.repeat(self.input_Nx)
    
    def show_input_U(
        self, 
        cmap='turbo', 
        fontsize=16, 
        xlim=None, 
        ylim=None, 
        selected_field='all',
        dark_style=False,
        show=True
        ):
        """
        Show the input field.
        
        Args:
            cmap (str): The colormap to use. 'turbo', 'viridis', 'plasma', 'inferno', 'magma'
            fontsize (int): The font size.
            xlim (list): The x-axis limits. [xmin, xmax] Unit: µm
            ylim (list): The y-axis limits. [ymin, ymax] Unit: µm
            selected_field (str): The field to plot. 'all', 'field', 'amplitude', 'phase'
            dark_style (bool): The style of the plot. 'True' for dark style, 'False' for light style.
            show (bool): Whether to show the plot. 'True' for show, 'False' for not show.
        """
        # Rich print the input field setting
        table = Table(title="INCIDENT FIELD", show_header=True, header_style="bold magenta")
        table.add_column("Attribute", style="cyan", no_wrap=True)
        table.add_column("Value", style="green")
        table.add_column("Unit", style="yellow")
        table.add_column("type", style="yellow")
        table.add_column("device", style="yellow")
        table.add_row("Propagation wavelength", str(self.propagation_wavelength), "m", str(type(self.propagation_wavelength)), str(self.device))
        table.add_row("Propagation distance", str(self.propagation_distance), "m", str(type(self.propagation_distance)), str(self.device))
        table.add_row("Refractive index", str(self.n), "", str(type(self.n)), str(self.device))
        table.add_row("Input Size", str(self.input_Nx) + " x " + str(self.input_Ny), "pixels", str(type(self.input_Nx)), str(self.device))
        table.add_row("Input Pixel size", str(self.input_pixel_size), "m", str(type(self.input_pixel_size)), str(self.device))
        table.add_row("Input X", str(self.input_X.min().item()) + " ~ " + str(self.input_X.max().item()), "m", str(type(self.input_X.min())), str(self.device))
        table.add_row("Input Y", str(self.input_Y.min().item()) + " ~ " + str(self.input_Y.max().item()), "m", str(type(self.input_Y.min())), str(self.device))
        console = Console()
        console.print(table)

        plot_field_amplitude_phase(
            self.input_U, 
            self.input_X, 
            self.input_Y, 
            cmap=cmap, 
            fontsize=fontsize, 
            xlim=xlim, 
            ylim=ylim, 
            selected_field=selected_field,
            dark_style=dark_style,
            show=show
            )

    def show_output_U(
        self, 
        cmap='turbo', 
        fontsize=16, 
        xlim=None, 
        ylim=None, 
        selected_field='all',
        dark_style=False,
        show=True
        ):
        """
        Show the output field.
        
        Args:
            cmap (str): The colormap to use. 'turbo', 'viridis', 'plasma', 'inferno', 'magma'
            fontsize (int): The font size.
            xlim (list): The x-axis limits. [xmin, xmax] Unit: µm
            ylim (list): The y-axis limits. [ymin, ymax] Unit: µm
            selected_field (str): The field to plot. 'all', 'field', 'amplitude', 'phase'
            dark_style (bool): The style of the plot. 'True' for dark style, 'False' for light style.
            show (bool): Whether to show the plot. 'True' for show, 'False' for not show.
        """
        # Rich print the output field setting
        table = Table(title="OUTPUT FIELD", show_header=True, header_style="bold magenta")
        table.add_column("Attribute", style="cyan", no_wrap=True)
        table.add_column("Value", style="green")
        table.add_column("Unit", style="yellow")
        table.add_column("type", style="yellow")
        table.add_column("device", style="yellow")
        table.add_row("Output Size", str(self.input_Nx) + " x " + str(self.input_Ny), "pixels", str(type(self.input_Nx)), str(self.device))
        table.add_row("Output Pixel size", str(self.output_pixel_size), "m", str(type(self.output_pixel_size)), str(self.device))
        table.add_row("Output X", str(self.output_X.min().item()) + " ~ " + str(self.output_X.max().item()), "m", str(type(self.output_X.min())), str(self.device))
        table.add_row("Output Y", str(self.output_Y.min().item()) + " ~ " + str(self.output_Y.max().item()), "m", str(type(self.output_Y.min())), str(self.device))
        console = Console()
        console.print(table)
        plot_field_amplitude_phase(
            self.output_U, 
            self.output_X, 
            self.output_Y, 
            cmap=cmap, 
            fontsize=fontsize, 
            xlim=xlim, 
            ylim=ylim, 
            selected_field=selected_field,
            dark_style=dark_style,
            show=show
            )

    def show_intensity(
        self, 
        cmap='turbo', 
        unit='µm',
        norm='linear',
        fontsize=16, 
        xlim=None, 
        ylim=None, 
        dark_style=False,
        clim=None,
        title=None,
        show=True
        ):
        """Show intensity distribution."""
        plot_field_intensity(
            self.output_U, 
            self.output_X, 
            self.output_Y, 
            cmap=cmap, 
            unit=unit,
            norm=norm,
            fontsize=fontsize, 
            xlim=xlim, 
            ylim=ylim, 
            clim=clim,
            dark_style=dark_style,
            title=title,
            show=show
            )
    
    def show_xz_intensity(
        self, 
        cmap='turbo', 
        unit='µm',
        norm='linear',
        fontsize=16, 
        xlim=None, 
        zlim=None, 
        dark_style=False,
        clim=None,
        title=None,
        show=True
        ):
        """Show XZ intensity distribution."""
        plot_xz_field_intensity(
            U0=self.output_UZ, 
            X=self.output_X, 
            Z=self.z_range, 
            cmap=cmap, 
            unit=unit,
            norm=norm,
            fontsize=fontsize, 
            xlim=zlim, 
            ylim=xlim, 
            clim=clim,
            dark_style=dark_style,
            title=title,
            show=show
            )

    @property
    def get_intensity(self):
        """Get intensity distribution."""
        return torch.abs(self.output_U)**2
    
    @property
    def get_output_X(self):
        """Get output X coordinates."""
        return self.output_X
    
    @property
    def get_output_Y(self):
        """Get output Y coordinates."""
        return self.output_Y
    
    @property
    def get_output_U(self):
        """Get output field."""
        return self.output_U
    
    @property
    def get_output_UZ(self):
        """Get output field for XZ propagation."""
        return self.output_UZ


class ASMPropagation:
    def __init__(
        self, 
        propagation_wavelength: float, 
        propagation_distance: float,
        n: float = 1.0,
        dtype: torch.dtype = torch.float32,
        device: str = 'cpu'
        ):
        """
        Initialize Angular Spectrum Method propagation.
        
        Args:
            propagation_wavelength (float): Propagation wavelength (unit: m)
            propagation_distance (float): Propagation distance (unit: m)
            n (float): Refractive index
            dtype (torch.dtype): Data type for computation
            device (str): 'cpu' or 'cuda' for GPU
        """
        self.propagation_wavelength = propagation_wavelength / n
        self.propagation_distance = propagation_distance
        self.device = check_available_cuda(device=device)        
        self.dtype = dtype
        self.n = n
        # Calculate wave number k
        self.k = 2 * torch.pi / self.propagation_wavelength

    def set_input_field(
        self, 
        u_in: torch.Tensor, 
        pixel_size: float
        ):
        """
        Set input field for propagation.
        
        Args:
            u_in (torch.Tensor): Input complex field (shape: [Nx, Ny])
            pixel_size (float): Pixel size (unit: m)
        """
        if not torch.is_tensor(u_in):
            raise ValueError("Input u_in must be a PyTorch tensor.")
        
        # Check if it's a complex tensor
        if not torch.is_complex(u_in):
            if u_in.ndim == 2:
                # If it's a real tensor, convert to complex tensor with zero imaginary part
                u_in = u_in + 1j * torch.zeros_like(u_in)
            else:
                raise ValueError("Tensor shape is not compatible for complex conversion.")
        if self.dtype == torch.float32:
            u_in = u_in.to(dtype=torch.complex64)
        elif self.dtype == torch.float64:
            u_in = u_in.to(dtype=torch.complex128)
        # Transfer tensor to corresponding device
        self.input_U = u_in.to(self.device)
        self.input_Nx, self.input_Ny = u_in.shape
        self.input_pixel_size = pixel_size
        self.input_X, self.input_Y = cart_grid(
            grid_size=[self.input_Nx, self.input_Ny], 
            grid_dx=[self.input_pixel_size, self.input_pixel_size], 
            dtype=self.dtype, 
            device=self.device
        )
        self.output_U = torch.zeros((self.input_Nx, self.input_Ny), dtype=self.dtype, device=self.device)
        self.output_X = self.input_X
        self.output_Y = self.input_Y
        self.output_pixel_size = self.input_pixel_size
    
    def propagate(self):
        """Execute Angular Spectrum propagation."""
        self.output_U = self._angular_spectrum()

    def propagate_xz(
        self, 
        z_range: list = np.arange(1000e-6, 2000e-6, 50e-6), 
        ):
        """
        Scan Angular Spectrum propagation at different distances.
        
        Args:
            z_range (list or array): Propagation distance range (unit: m)
        Returns:
            torch.Tensor: Output field with shape [len(z_range), Nx]
        """
        self.z_range = z_range
            
        # Initialize output field array
        if self.dtype == torch.float32:
            self.output_UZ = torch.zeros((len(z_range), self.input_Nx), 
                               dtype=torch.complex64, device=self.device)
        elif self.dtype == torch.float64:
            self.output_UZ = torch.zeros((len(z_range), self.input_Nx), 
                               dtype=torch.complex128, device=self.device)
        
        for i, z in enumerate(z_range):
            # Set current propagation distance
            self.propagation_distance = z
            
            # Calculate Angular Spectrum propagation
            wvl = self.propagation_wavelength
            k = self.k
            
            # Frequency domain coordinates
            fx = torch.fft.fftfreq(self.input_Nx, d=self.input_pixel_size, dtype=self.dtype, device=self.device)
            fy = torch.fft.fftfreq(self.input_Ny, d=self.input_pixel_size, dtype=self.dtype, device=self.device)
            fx, fy = torch.meshgrid(fx, fy, indexing='ij')
            
            # Transfer function
            sqrt_term = torch.sqrt(1 - (wvl**2)*(fx**2 + fy**2) + 0j)
            H = torch.exp(1j * k * z * sqrt_term)
            
            # FFT calculation
            U_in_freq = torch.fft.fft2(self.input_U)
            U_out_freq = U_in_freq * H
            output_U = torch.fft.ifft2(U_out_freq)
            
            # Take center line (y = 0)
            center_idx = self.input_Ny // 2
            self.output_UZ[i, :] = output_U[center_idx, :]
        
    def _angular_spectrum(self):
        """
        Angular Spectrum method.
        This method assumes the output grid has the same size and spacing as the input grid.
        For different output grids, interpolation or modified AS method with frequency domain resampling is needed.
        Current example does not handle this difference.
        
        Returns:
            torch.Tensor: Output complex field
        """
        # Frequency domain coordinates (fx, fy) for AS and Fresnel calculations
        fx = torch.fft.fftfreq(self.input_Nx, d=self.input_pixel_size, dtype=self.dtype, device=self.device)
        fy = torch.fft.fftfreq(self.input_Ny, d=self.input_pixel_size, dtype=self.dtype, device=self.device)
        fx, fy = torch.meshgrid(fx, fy, indexing='ij')
        wvl = self.propagation_wavelength
        z = self.propagation_distance
        
        sqrt_term = torch.sqrt(1 - (wvl**2)*(fx**2 + fy**2) + 0j)
        H = torch.exp(1j * self.k * z * sqrt_term)
        
        U_in_freq = torch.fft.fft2(self.input_U)
        U_out_freq = U_in_freq * H
        output_U = torch.fft.ifft2(U_out_freq)
        return output_U

    def set_propagation_distance(self, propagation_distance: float):
        """Set propagation distance."""
        self.propagation_distance = propagation_distance

    def show_input_U(
        self, 
        cmap='turbo', 
        fontsize=16, 
        xlim=None, 
        ylim=None, 
        selected_field='all',
        dark_style=False,
        show=True
        ):
        """
        Show the input field.
        
        Args:
            cmap (str): The colormap to use. 'turbo', 'viridis', 'plasma', 'inferno', 'magma'
            fontsize (int): The font size.
            xlim (list): The x-axis limits. [xmin, xmax] Unit: µm
            ylim (list): The y-axis limits. [ymin, ymax] Unit: µm
            selected_field (str): The field to plot. 'all', 'field', 'amplitude', 'phase'
            dark_style (bool): The style of the plot. 'True' for dark style, 'False' for light style.
            show (bool): Whether to show the plot. 'True' for show, 'False' for not show.
        """
        # Rich print the input field setting
        table = Table(title="INCIDENT FIELD", show_header=True, header_style="bold magenta")
        table.add_column("Attribute", style="cyan", no_wrap=True)
        table.add_column("Value", style="green")
        table.add_column("Unit", style="yellow")
        table.add_column("type", style="yellow")
        table.add_column("device", style="yellow")
        table.add_row("Propagation wavelength", str(self.propagation_wavelength), "m", str(type(self.propagation_wavelength)), str(self.device))
        table.add_row("Propagation distance", str(self.propagation_distance), "m", str(type(self.propagation_distance)), str(self.device))
        table.add_row("Refractive index", str(self.n), "", str(type(self.n)), str(self.device))
        table.add_row("Input Size", str(self.input_Nx) + " x " + str(self.input_Ny), "pixels", str(type(self.input_Nx)), str(self.device))
        table.add_row("Input Pixel size", str(self.input_pixel_size), "m", str(type(self.input_pixel_size)), str(self.device))
        table.add_row("Input X", str(self.input_X.min().item()) + " ~ " + str(self.input_X.max().item()), "m", str(type(self.input_X.min())), str(self.device))
        table.add_row("Input Y", str(self.input_Y.min().item()) + " ~ " + str(self.input_Y.max().item()), "m", str(type(self.input_Y.min())), str(self.device))
        console = Console()
        console.print(table)

        plot_field_amplitude_phase(
            self.input_U, 
            self.input_X, 
            self.input_Y, 
            cmap=cmap, 
            fontsize=fontsize, 
            xlim=xlim, 
            ylim=ylim, 
            selected_field=selected_field,
            dark_style=dark_style,
            show=show
            )

    def show_output_U(
        self, 
        cmap='turbo', 
        fontsize=16, 
        xlim=None, 
        ylim=None, 
        selected_field='all',
        dark_style=False,
        show=True
        ):
        """
        Show the output field.
        
        Args:
            cmap (str): The colormap to use. 'turbo', 'viridis', 'plasma', 'inferno', 'magma'
            fontsize (int): The font size.
            xlim (list): The x-axis limits. [xmin, xmax] Unit: µm
            ylim (list): The y-axis limits. [ymin, ymax] Unit: µm
            selected_field (str): The field to plot. 'all', 'field', 'amplitude', 'phase'
            dark_style (bool): The style of the plot. 'True' for dark style, 'False' for light style.
            show (bool): Whether to show the plot. 'True' for show, 'False' for not show.
        """
        # Rich print the output field setting
        table = Table(title="OUTPUT FIELD", show_header=True, header_style="bold magenta")
        table.add_column("Attribute", style="cyan", no_wrap=True)
        table.add_column("Value", style="green")
        table.add_column("Unit", style="yellow")
        table.add_column("type", style="yellow")
        table.add_column("device", style="yellow")
        table.add_row("Output Size", str(self.input_Nx) + " x " + str(self.input_Ny), "pixels", str(type(self.input_Nx)), str(self.device))
        table.add_row("Output Pixel size", str(self.output_pixel_size), "m", str(type(self.output_pixel_size)), str(self.device))
        table.add_row("Output X", str(self.output_X.min().item()) + " ~ " + str(self.output_X.max().item()), "m", str(type(self.output_X.min())), str(self.device))
        table.add_row("Output Y", str(self.output_Y.min().item()) + " ~ " + str(self.output_Y.max().item()), "m", str(type(self.output_Y.min())), str(self.device))
        console = Console()
        console.print(table)
        plot_field_amplitude_phase(
            self.output_U, 
            self.output_X, 
            self.output_Y, 
            cmap=cmap, 
            fontsize=fontsize, 
            xlim=xlim, 
            ylim=ylim, 
            selected_field=selected_field,
            dark_style=dark_style,
            show=show
            )

    def show_intensity(
        self, 
        cmap='turbo', 
        unit='µm',
        norm='linear',
        fontsize=16, 
        xlim=None, 
        ylim=None, 
        dark_style=False,
        clim=None,
        title=None,
        show=True
        ):
        """Show intensity distribution."""
        plot_field_intensity(
            self.output_U, 
            self.output_X, 
            self.output_Y, 
            cmap=cmap, 
            unit=unit,
            norm=norm,
            fontsize=fontsize, 
            xlim=xlim, 
            ylim=ylim, 
            clim=clim,
            dark_style=dark_style,
            title=title,
            show=show
            )
    
    def show_xz_intensity(
        self, 
        cmap='turbo', 
        unit='µm',
        norm='linear',
        fontsize=16, 
        xlim=None, 
        zlim=None, 
        dark_style=False,
        clim=None,
        title=None,
        show=True
        ):
        """Show XZ intensity distribution."""
        plot_xz_field_intensity(
            U0=self.output_UZ, 
            X=self.output_X, 
            Z=self.z_range, 
            cmap=cmap, 
            unit=unit,
            norm=norm,
            fontsize=fontsize, 
            xlim=zlim, 
            ylim=xlim, 
            clim=clim,
            dark_style=dark_style,
            title=title,
            show=show
            )

    @property
    def get_intensity(self):
        """Get intensity distribution."""
        return torch.abs(self.output_U)**2
    
    @property
    def get_output_X(self):
        """Get output X coordinates."""
        return self.output_X
    
    @property
    def get_output_Y(self):
        """Get output Y coordinates."""
        return self.output_Y
    
    @property
    def get_output_U(self):
        """Get output field."""
        return self.output_U
    
    @property
    def get_output_UZ(self):
        """Get output field for XZ propagation."""
        return self.output_UZ
    

class RayleighSommerfeldPropagation:
    def __init__(
        self, 
        propagation_wavelength: float, 
        propagation_distance: float,
        n: float = 1.0,
        device: str = 'cpu',
        dtype: torch.dtype = torch.float32,
        ):
        """
        Initialize Rayleigh-Sommerfeld propagation.
        
        Args:
            propagation_wavelength (float): Propagation wavelength (unit: m)
            propagation_distance (float): Propagation distance (unit: m)
            n (float): Refractive index
            device (str): 'cpu' or 'cuda' for GPU
            dtype (torch.dtype): Data type for computation
        """
        self.propagation_wavelength = propagation_wavelength / n
        self.propagation_distance = propagation_distance
        self.device = check_available_cuda(device=device)        
        self.dtype = dtype
        self.n = n
        # Calculate wave number k
        self.k = 2 * torch.pi / self.propagation_wavelength

    def set_input_field(
        self, 
        u_in: torch.Tensor, 
        pixel_size: float, 
        field_center: list = None,
        input_X = None,
        input_Y = None
        ):
        """
        Set input field for propagation.
        
        Args:
            u_in (torch.Tensor): Input complex field (shape: [Nx, Ny])
            pixel_size (float): Pixel size (unit: m)
            field_center (list): Input plane center coordinates [x_center, y_center], use origin if None
            input_X (torch.Tensor): Input X coordinates, calculate using pixel_size if None
            input_Y (torch.Tensor): Input Y coordinates, calculate using pixel_size if None
        """
        if not torch.is_tensor(u_in):
            raise ValueError("Input u_in must be a PyTorch tensor.")
        
        # Check if it's a complex tensor
        if not torch.is_complex(u_in):
            if u_in.ndim == 2:
                # If it's a real tensor, convert to complex tensor with zero imaginary part
                u_in = u_in + 1j * torch.zeros_like(u_in)
            else:
                raise ValueError("Tensor shape is not compatible for complex conversion.")
            if self.dtype == torch.float32:
                u_in = u_in.to(dtype=torch.complex64)
            elif self.dtype == torch.float64:
                u_in = u_in.to(dtype=torch.complex128)
        
        # Transfer tensor to corresponding device
        self.input_U = u_in.to(self.device)
        if input_X is not None and input_Y is not None:
            self.input_X = input_X
            self.input_Y = input_Y
            self.input_pixel_size = input_X[1] - input_X[0]
            print(self.input_pixel_size)
        else:
            self.input_Nx, self.input_Ny = u_in.shape
            self.input_pixel_size = pixel_size
            self.input_X, self.input_Y = cart_grid([self.input_Nx, self.input_Ny], 
                                                  [self.input_pixel_size, self.input_pixel_size], 
                                                  dtype=torch.float64, device=self.device)
        if field_center is not None:
            self.input_X = self.input_X + field_center[0]
            self.input_Y = self.input_Y + field_center[1]

    def set_output_field(
        self, 
        output_pixel_size: float, 
        output_size: list = None, 
        center: list = None
        ):
        """
        Set output field parameters.
        
        Args:
            output_pixel_size (float): Output pixel size (unit: m)
            output_size (list): Output grid size [Nx_out, Ny_out], use input size if None
            center (list): Output plane center coordinates [x_center, y_center], use origin if None
        """
        self.output_pixel_size = output_pixel_size
        
        if output_size is None:
            self.output_Nx, self.output_Ny = self.input_Nx, self.input_Ny
        else:
            self.output_Nx, self.output_Ny = output_size
            
        self.output_X, self.output_Y = cart_grid([self.output_Nx, self.output_Ny], 
                                                [self.output_pixel_size, self.output_pixel_size], 
                                                dtype=self.dtype, device=self.device)
        if self.dtype == torch.float32:
            self.output_U = torch.zeros((self.output_Nx, self.output_Ny), 
                                    dtype=torch.complex64, device=self.device)
        elif self.dtype == torch.float64:
            self.output_U = torch.zeros((self.output_Nx, self.output_Ny), 
                                dtype=torch.complex128, device=self.device)
        if center is not None:
            self.output_X = self.output_X + center[0]
            self.output_Y = self.output_Y + center[1]

    def propagate(self):
        """Execute Rayleigh-Sommerfeld propagation calculation."""
        self.output_U = self._rayleigh_sommerfeld()

    def _rayleigh_sommerfeld(self):
        """
        Rayleigh-Sommerfeld diffraction integral implementation.

        Returns:
            output_U (torch.Tensor): Output complex field (shape: [Nx_out, Ny_out])
        """
        z = self.propagation_distance
        wvl = self.propagation_wavelength
        k = self.k
        
        # Initialize output field
        if self.dtype == torch.float32:
            output_U = torch.zeros((self.output_Nx, self.output_Ny), 
                                dtype=torch.complex64, device=self.device)
        elif self.dtype == torch.float64:
            output_U = torch.zeros((self.output_Nx, self.output_Ny), 
                                dtype=torch.complex128, device=self.device)
        
        # Calculate propagation kernel function
        for a in range(self.output_Nx):
            for b in range(self.output_Ny):
                # Calculate distance r01, corresponding to MATLAB: r01 = sqrt(GPU_z^2+(GPU_Kersi(a,b)-GPU_X).^2+(GPU_Ita(a,b)-GPU_Y).^2)
                r01 = torch.sqrt(z**2 + (self.output_X[a, b] - self.input_X)**2 + 
                               (self.output_Y[a, b] - self.input_Y)**2)
                
                # Rayleigh-Sommerfeld kernel function, corresponding to MATLAB: GPU_z/(1j*GPU_lambda)*exp(1j*GPU_k.*r01)./r01./r01
                H = z / (1j * wvl) * torch.exp(1j * k * r01) / (r01 ** 2)
                
                # Integration calculation, corresponding to MATLAB: sum(sum(GPU_U0.*H))
                integrand = self.input_U * H * self.input_pixel_size**2
                output_U[a, b] = torch.sum(integrand)
        
        return output_U

    def propagate_xz(
        self, 
        z_range: list = np.arange(1000e-6, 2000e-6, 50e-6), 
        yaxis: str = 'x'
        ):
        """
        Scan Rayleigh-Sommerfeld propagation at different distances.
        
        Args:
            z_range (list or array): Propagation distance range (unit: m)
            yaxis (str): Axis for cross-section, 'x' or 'y'
        Returns:
            torch.Tensor: Output field with shape [len(z_range), Nx]
        """
            
        # Initialize output field array
        if self.dtype == torch.float32:
            self.output_UZ = torch.zeros((len(z_range), len(self.output_X)), 
                                dtype=torch.complex64, device=self.device)
        elif self.dtype == torch.float64:
            self.output_UZ = torch.zeros((len(z_range), len(self.output_X)), 
                                dtype=torch.complex128, device=self.device)
        self.z_range = z_range
        for c, z in enumerate(z_range):
            # Set current propagation distance
            self.propagation_distance = z
            
            # Initialize temporary output field
            if self.dtype == torch.float32:
                output_U_temp = torch.zeros(len(self.output_X), dtype=torch.complex64, device=self.device)
            elif self.dtype == torch.float64:
                output_U_temp = torch.zeros(len(self.output_X), dtype=torch.complex128, device=self.device)
            
            # Calculate propagation for each x position
            for a in range(len(self.output_X)):
                # Here b = round(size(Ita,2)/2) means taking the center line
                if yaxis == 'x':
                    b = self.output_Ny // 2
                    r01 = torch.sqrt(z**2 + (self.output_X[b, a] - self.input_X)**2 + 
                               (self.output_Y[b, a] - self.input_Y)**2)
                elif yaxis == 'y':
                    b = self.output_Nx // 2
                    r01 = torch.sqrt(z**2 + (self.output_X[a, b] - self.input_X)**2 + 
                               (self.output_Y[a, b] - self.input_Y)**2)
                else:
                    raise ValueError("yaxis must be 'x' or 'y'")

                # Rayleigh-Sommerfeld kernel function
                H = z / (1j * self.propagation_wavelength) * torch.exp(1j * self.k * r01) / (r01 ** 2)
                
                # Integration calculation
                integrand = self.input_U * H * self.input_pixel_size**2
                output_U_temp[a] = torch.sum(integrand)
            
            # Save results for current distance
            self.output_UZ[c, :] = output_U_temp
        
    def set_propagation_distance(self, propagation_distance: float):
        """Set propagation distance."""
        self.propagation_distance = propagation_distance

    def show_input_U(
        self, 
        cmap='turbo', 
        fontsize=16, 
        xlim=None, 
        ylim=None, 
        selected_field='all',
        dark_style=False,
        show=True
        ):
        """
        Show the input field.
        
        Args:
            cmap (str): The colormap to use. 'turbo', 'viridis', 'plasma', 'inferno', 'magma'
            fontsize (int): The font size.
            xlim (list): The x-axis limits. [xmin, xmax] Unit: µm
            ylim (list): The y-axis limits. [ymin, ymax] Unit: µm
            selected_field (str): The field to plot. 'all', 'field', 'amplitude', 'phase'
            dark_style (bool): The style of the plot. 'True' for dark style, 'False' for light style.
            show (bool): Whether to show the plot. 'True' for show, 'False' for not show.
        """
        # Rich print the input field setting
        table = Table(title="INCIDENT FIELD", show_header=True, header_style="bold magenta")
        table.add_column("Attribute", style="cyan", no_wrap=True)
        table.add_column("Value", style="green")
        table.add_column("Unit", style="yellow")
        table.add_column("type", style="yellow")
        table.add_column("device", style="yellow")
        table.add_row("Propagation wavelength", str(self.propagation_wavelength), "m", str(type(self.propagation_wavelength)), str(self.device))
        table.add_row("Propagation distance", str(self.propagation_distance), "m", str(type(self.propagation_distance)), str(self.device))
        table.add_row("Refractive index", str(self.n), "", str(type(self.n)), str(self.device))
        table.add_row("Input Size", str(self.input_Nx) + " x " + str(self.input_Ny), "pixels", str(type(self.input_Nx)), str(self.device))
        table.add_row("Input Pixel size", str(self.input_pixel_size), "m", str(type(self.input_pixel_size)), str(self.device))
        table.add_row("Input X", str(self.input_X.min().item()) + " ~ " + str(self.input_X.max().item()), "m", str(type(self.input_X.min())), str(self.device))
        table.add_row("Input Y", str(self.input_Y.min().item()) + " ~ " + str(self.input_Y.max().item()), "m", str(type(self.input_Y.min())), str(self.device))
        console = Console()
        console.print(table)

        plot_field_amplitude_phase(
            self.input_U, 
            self.input_X, 
            self.input_Y, 
            cmap=cmap, 
            fontsize=fontsize, 
            xlim=xlim, 
            ylim=ylim, 
            selected_field=selected_field,
            dark_style=dark_style,
            show=show
            )

    def show_output_U(
        self, 
        cmap='turbo', 
        fontsize=16, 
        xlim=None, 
        ylim=None, 
        selected_field='all',
        dark_style=False,
        show=True
        ):
        """
        Show the output field.
        
        Args:
            cmap (str): The colormap to use. 'turbo', 'viridis', 'plasma', 'inferno', 'magma'
            fontsize (int): The font size.
            xlim (list): The x-axis limits. [xmin, xmax] Unit: µm
            ylim (list): The y-axis limits. [ymin, ymax] Unit: µm
            selected_field (str): The field to plot. 'all', 'field', 'amplitude', 'phase'
            dark_style (bool): The style of the plot. 'True' for dark style, 'False' for light style.
            show (bool): Whether to show the plot. 'True' for show, 'False' for not show.
        """
        # Rich print the output field setting
        table = Table(title="OUTPUT FIELD", show_header=True, header_style="bold magenta")
        table.add_column("Attribute", style="cyan", no_wrap=True)
        table.add_column("Value", style="green")
        table.add_column("Unit", style="yellow")
        table.add_column("type", style="yellow")
        table.add_column("device", style="yellow")
        table.add_row("Output Size", str(self.output_Nx) + " x " + str(self.output_Ny), "pixels", str(type(self.output_Nx)), str(self.device))
        table.add_row("Output Pixel size", str(self.output_pixel_size), "m", str(type(self.output_pixel_size)), str(self.device))
        table.add_row("Output X", str(self.output_X.min().item()) + " ~ " + str(self.output_X.max().item()), "m", str(type(self.output_X.min())), str(self.device))
        table.add_row("Output Y", str(self.output_Y.min().item()) + " ~ " + str(self.output_Y.max().item()), "m", str(type(self.output_Y.min())), str(self.device))
        console = Console()
        console.print(table)
        plot_field_amplitude_phase(
            self.output_U, 
            self.output_X, 
            self.output_Y, 
            cmap=cmap, 
            fontsize=fontsize, 
            xlim=xlim, 
            ylim=ylim, 
            selected_field=selected_field,
            dark_style=dark_style,
            show=show
            )

    def show_intensity(
        self, 
        cmap='turbo', 
        unit='µm',
        norm='linear',
        fontsize=16, 
        xlim=None, 
        ylim=None, 
        dark_style=False,
        clim=None,
        title=None,
        show=True
        ):
        """Show intensity distribution."""
        plot_field_intensity(
            self.output_U, 
            self.output_X, 
            self.output_Y, 
            cmap=cmap, 
            unit=unit,
            norm=norm,
            fontsize=fontsize, 
            xlim=xlim, 
            ylim=ylim, 
            clim=clim,
            dark_style=dark_style,
            title=title,
            show=show
            )
    
    def show_xz_intensity(
        self, 
        cmap='turbo', 
        unit='µm',
        norm='linear',
        fontsize=16, 
        xlim=None, 
        zlim=None, 
        dark_style=False,
        clim=None,
        title=None,
        show=True
        ):
        """Show XZ intensity distribution."""
        plot_xz_field_intensity(
            U0=self.output_UZ, 
            X=self.output_X, 
            Z=self.z_range, 
            cmap=cmap, 
            unit=unit,
            norm=norm,
            fontsize=fontsize, 
            xlim=zlim, 
            ylim=xlim, 
            clim=clim,
            dark_style=dark_style,
            title=title,
            show=show
            )

    @property
    def get_intensity(self):
        """Get intensity distribution."""
        return torch.abs(self.output_U)**2
    
    @property
    def get_output_X(self):
        """Get output X coordinates."""
        return self.output_X
    
    @property
    def get_output_Y(self):
        """Get output Y coordinates."""
        return self.output_Y
    
    @property
    def get_output_U(self):
        """Get output field."""
        return self.output_U
    
    @property
    def get_output_UZ(self):
        """Get output field for XZ propagation."""
        return self.output_UZ
    
    @property
    def get_z_range(self):
        """Get propagation distance range."""
        return self.z_range