import torch
import torch.nn.functional as F
from .utils import *
from rich.table import Table
from rich.console import Console
from abc import ABC, abstractmethod

class NearField:
    """
    """
    def __init__(
        self,
        pixel_size : float = 0.5e-6,
        field_Lx :float = 1000e-6, 
        field_Ly :float = 1000e-6, 
        dtype: torch.dtype = torch.float32,
        field_center: list = [0, 0],
        device: str = 'cpu'
        ):
        """
        Initialize the FlatLens class.
        Args:
            pixel_size (float): The pixel size of the field (unit: m).
            field_Lx (float): The length of the field (unit: m).
            field_Ly (float): The width of the field (unit: m).
            dtype (torch.dtype): The data type of the field.
            field_center (list): The center of the field [cx, cy] (unit: m).
            device (str): 'cpu' or 'cuda'.
        """
        # Raises check all the parameters must be positive
        if pixel_size <= 0:
            raise ValueError("Pixel size must be positive.")
        # Raises check dtype must be torch.float32 or torch.float64
        if dtype not in [torch.float32, torch.float64]:
            raise ValueError("Data type must be torch.float32 or torch.float64.")
        # Raises check device must be 'cpu' or 'cuda'
        if device not in ['cpu', 'cuda']:
            raise ValueError("Device must be 'cpu' or 'cuda'.")
        
        # Assign the parameters to the attributes
        self.device = check_available_cuda(device)
        self.dtype = dtype
        self.pixel_size = pixel_size
        self.field_Lx = field_Lx
        self.field_Ly = field_Ly
        
        # Calculate the phase profile
        self.Nx, self.Ny = get_grid_size(length=self.field_Lx, width=self.field_Ly , pixel_size=pixel_size)
        self.X, self.Y = cart_grid(grid_size=[self.Nx, self.Ny], grid_dx=[pixel_size, pixel_size], dtype=dtype, device=self.device)
        self.X = self.X - field_center[0]
        self.Y = self.Y - field_center[1]

    def __repr__(self):
        lines = []
        lines.append(f"Pixel size: {self.pixel_size}")
        lines.append(f"Field size: {self.field_Lx} x {self.field_Ly}")
        lines.append(f"Number of pixels: {self.Nx} x {self.Ny}")
        lines.append(f"Device: {self.device}")
        lines.append(f"Data type: {self.dtype}")
        return "\n".join(lines)

    def rich_print(self):
        table = Table(title="FIELD DATA", show_header=True, header_style="bold magenta")
        table.add_column("Attribute", style="cyan", no_wrap=True)
        table.add_column("Value", style="green")
        table.add_column("Unit", style="yellow")
        table.add_column("type", style="yellow")
        table.add_column("device", style="yellow")
        table.add_row("Pixel size", str(self.pixel_size), "m", str(type(self.pixel_size)), str(self.device))
        table.add_row("Field size", str(self.field_Lx) + " x " + str(self.field_Ly), "m", str(type(self.field_Lx)), str(self.device))
        table.add_row("Number of pixels", str(self.Nx) + " x " + str(self.Ny), "pixels", str(type(self.Nx)), str(self.device))
        table.add_row("X", str(self.X.min().item()) + " ~ " + str(self.X.max().item()), "m", str(type(self.X.min())), str(self.device))
        table.add_row("Y", str(self.Y.min().item()) + " ~ " + str(self.Y.max().item()), "m", str(type(self.Y.min())), str(self.device))
        table.add_row("Device", str(self.device), "", str(type(self.device)), str(self.device))
        table.add_row("Data type", str(self.dtype), "", str(type(self.dtype)), str(self.device))
        console = Console()
        console.print(table)

    def input_field(
        self, 
        U0: torch.Tensor
        ):
        self.U0 = U0
        return self.U0

    def draw(
        self, 
        U0 : torch.Tensor, 
        cmap : str = 'turbo', 
        fontsize : int = 16, 
        xlim : list = None, 
        ylim : list = None, 
        selected_field : str = 'all',
        dark_style : bool = False,
        title : str = None,
        show : bool = True
        ):
        """
        Draw the field
        Args:
            U0 (torch.Tensor): The field to plot.
            cmap (str): The colormap to use. 'turbo', 'viridis', 'plasma', 'inferno', 'magma'
            fontsize (int): The font size.
            xlim (list): The x-axis limits. [xmin, xmax] Unit: µm
            ylim (list): The y-axis limits. [ymin, ymax] Unit: µm
            selected_field (str): The field to plot. 'all', 'field', 'amplitude', 'phase'
            dark_style (bool): The style of the plot. 'True' for dark style, 'False' for light style.
            show (bool): Whether to show the plot. 'True' for show, 'False' for not show. 
        """
        plot_field_amplitude_phase(
            U0, 
            self.X, 
            self.Y, 
            cmap=cmap, 
            fontsize=fontsize, 
            xlim=xlim, 
            ylim=ylim, 
            selected_field=selected_field,
            dark_style=dark_style,
            title=title,
            show=show
        )


class PhaseElement(ABC):
    """
    Abstract base class for defining common interfaces for phase elements
    """
    def __init__(self, near_field: NearField):
        """
        Initialize phase element
        
        Args:
            near_field (NearField): NearField object used for calculation
        """
        self.near_field = near_field
        # Inherit basic attributes from NearField
        self.X = near_field.X
        self.Y = near_field.Y
        self.device = near_field.device
        self.dtype = near_field.dtype
        self.pixel_size = near_field.pixel_size
        self.Nx = near_field.Nx
        self.Ny = near_field.Ny
        self.U0 = None
    
    @abstractmethod
    def calculate_phase(self, **kwargs):
        """
        Abstract method: Calculate phase distribution
        Each subclass must implement this method
        """
        pass
    
    def _create_aperture(self, aperture_type: str, aperture_size: list, 
                        center: list = [0, 0], reference_X=None, reference_Y=None):
        """
        Create aperture function
        
        Args:
            aperture_type (str): Aperture type 'circle' or 'rectangle'
            aperture_size (list): Aperture size [width, height] or [diameter]
            center (list): Aperture center position [cx, cy]
            reference_X, reference_Y: Reference coordinates, use self.X, self.Y if None
        """
        if reference_X is None:
            reference_X = self.X - center[0]
        if reference_Y is None:
            reference_Y = self.Y - center[1]
            
        if aperture_type == 'circle':
            diameter = aperture_size[0] if len(aperture_size) == 1 else aperture_size[0]
            aperture = (reference_X**2 + reference_Y**2) < (diameter / 2)**2
        elif aperture_type == 'rectangle':
            mask1 = reference_X > -aperture_size[0]/2
            mask2 = reference_X < aperture_size[0]/2
            mask3 = reference_Y > -aperture_size[1]/2
            mask4 = reference_Y < aperture_size[1]/2
            aperture = mask1 * mask2 * mask3 * mask4
        else:
            raise ValueError("Aperture type must be 'circle' or 'rectangle'.")
        
        return aperture
    
    @abstractmethod
    def rich_print(self):
        pass

    def draw(
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
        Inherited drawing method from NearField
        """
        if self.U0 is None:
            raise ValueError("Please calculate phase first by calling calculate_phase()")
        self.near_field.draw(
            self.U0, 
            cmap=cmap, 
            fontsize=fontsize, 
            xlim=xlim, 
            ylim=ylim, 
            selected_field=selected_field,
            dark_style=dark_style,
            show=show
        )


class Binary2Phase(PhaseElement):
    """
    BINARY2 PHASE
    Method:
        calculate_phase(binary2: list, lens_diameter: float, lens_center: list, aperture_type: str, aperture_size: list) -> torch.Tensor:
            Calculate binary2 phase distribution
        rich_print() -> None:
            Print the binary2 phase coefficients
        draw(cmap: str, fontsize: int, xlim: list, ylim: list, selected_field: str, dark_style: bool, show: bool) -> None:
            Draw the binary2 phase distribution
    """
    def calculate_phase(
        self, 
        binary2: list = [-10000, 0, 0, 0], 
        lens_diameter: float = 1600e-6, 
        lens_center: list = [0, 0],
        aperture_type: str = 'circle',
        aperture_size: list = [1600e-6, 1600e-6],
        amplitude : float = 1,
        phase_offset : float = 0,
    ):
        """
        Calculate binary2 phase distribution
        
        Args:
            binary2 (list): Binary2 phase coefficients
            lens_diameter (float): Lens diameter (m)
            lens_center (list): Lens center position [cx, cy] (m)
            aperture_type (str): Aperture type 'circle' or 'rectangle'
            aperture_size (list): Aperture size [width, height] (m)
        
        Returns:
            torch.Tensor: Complex field U0
        """
        self.binary2 = binary2
        self.lens_diameter = lens_diameter
        self.lens_center = lens_center
        self.aperture_type = aperture_type
        self.aperture_size = aperture_size
        reference_X = self.X - lens_center[0]
        reference_Y = self.Y - lens_center[1]
        
        # Create aperture
        aperture = self._create_aperture(self.aperture_type, self.aperture_size, self.lens_center, reference_X, reference_Y)
        
        # Calculate phase
        ai = torch.tensor(self.binary2, dtype=self.dtype, device=self.device)
        ai_power = torch.arange(1, len(self.binary2)+1, dtype=self.dtype, device=self.device)
        r2 = (reference_X*1e3)**2 + (reference_Y*1e3)**2
        phase = torch.sum(ai * r2.unsqueeze(-1) ** ai_power, dim=-1)
        phase = phase + phase_offset
        self.U0 = amplitude * torch.exp(1j * phase) * aperture
        return self.U0
        
    def rich_print(self):
        self.near_field.rich_print()
        table = Table(title="BINARY2 PHASE", show_header=True, header_style="bold magenta")
        table.add_column("Attribute", style="cyan", no_wrap=True)
        table.add_column("Value", style="green")
        table.add_column("Unit", style="yellow")
        table.add_column("type", style="yellow")
        table.add_column("device", style="yellow")
        table.add_row("Binary2", str(self.binary2), "", str(type(self.binary2)), str(self.device))
        table.add_row("Lens diameter", str(self.lens_diameter), "m", str(type(self.lens_diameter)), str(self.device))
        table.add_row("Lens center", str(self.lens_center), "m", str(type(self.lens_center)), str(self.device))
        table.add_row("Aperture type", str(self.aperture_type), "", str(type(self.aperture_type)), str(self.device))
        table.add_row("Aperture size", str(self.aperture_size), "m", str(type(self.aperture_size)), str(self.device))
        console = Console()
        console.print(table)


class EqualPathPhase(PhaseElement):
    """
    EQUAL PATH PHASE
    """
    def calculate_phase(
        self, 
        focal_length: float = 1000e-6, 
        design_lambda: float = 0.94e-6, 
        lens_diameter: float = 1600e-6, 
        lens_center: list = [0, 0],
        aperture_type: str = 'circle',
        aperture_size: list = [1600e-6, 1600e-6],
        amplitude : float = 1,
        phase_offset : float = 0,
    ):
        """
        Calculate equal path phase distribution
        
        Args:
            focal_length (float): Focal length (m)
            design_lambda (float): Design wavelength (m)
            lens_diameter (float): Lens diameter (m)
            lens_center (list): Lens center position [cx, cy] (m)
            aperture_type (str): Aperture type 'circle' or 'rectangle'
            aperture_size (list): Aperture size [width, height] (m)
        
        Returns:
            torch.Tensor: Complex field U0
        """
        self.focal_length = focal_length
        self.design_lambda = design_lambda
        self.lens_diameter = lens_diameter
        self.lens_center = lens_center
        self.aperture_type = aperture_type
        self.aperture_size = aperture_size
        
        reference_X = self.X - lens_center[0]
        reference_Y = self.Y - lens_center[1]
        
        # Create aperture
        aperture = self._create_aperture(aperture_type, aperture_size, lens_center, reference_X, reference_Y)
        
        # Calculate phase
        phase = -2 * torch.pi / design_lambda * \
            (torch.sqrt(reference_X**2 + reference_Y**2 + focal_length**2) - focal_length)
        phase = phase + phase_offset
        self.U0 = amplitude * torch.exp(1j * phase) * aperture
        return self.U0
    
    def rich_print(self):
        self.near_field.rich_print()
        table = Table(title="EQUAL PATH PHASE", show_header=True, header_style="bold magenta")
        table.add_column("Attribute", style="cyan", no_wrap=True)
        table.add_column("Value", style="green")
        table.add_column("Unit", style="yellow")
        table.add_column("type", style="yellow")
        table.add_column("device", style="yellow")
        table.add_row("Focal length", str(self.focal_length), "m", str(type(self.focal_length)), str(self.device))
        table.add_row("Design lambda", str(self.design_lambda), "m", str(type(self.design_lambda)), str(self.device))
        table.add_row("Lens diameter", str(self.lens_diameter), "m", str(type(self.lens_diameter)), str(self.device))
        table.add_row("Lens center", str(self.lens_center), "m", str(type(self.lens_center)), str(self.device))
        table.add_row("Aperture type", str(self.aperture_type), "", str(type(self.aperture_type)), str(self.device))
        table.add_row("Aperture size", str(self.aperture_size), "m", str(type(self.aperture_size)), str(self.device))
        console = Console()
        console.print(table)

class AxiCubicPhase(PhaseElement):
    """
    CUBIC PHASE
    """
    def calculate_phase(
        self, 
        focal_length: float = 1000e-6, 
        axi: float = 0,
        alpha: float = 0, 
        design_lambda: float = 0.94e-6, 
        lens_diameter: float = 1600e-6, 
        lens_center: list = [0, 0],
        aperture_type: str = 'circle',
        aperture_size: list = [1600e-6, 1600e-6],
        amplitude : float = 1,
        phase_offset : float = 0,
    ):
        """
        Calculate cubic phase distribution
        
        Args:
            focal_length (float): Focal length (m)
            alpha (float): Cubic phase parameter (rad)
            design_lambda (float): Design wavelength (m)
            lens_diameter (float): Lens diameter (m)
            lens_center (list): Lens center position [cx, cy] (m)
            aperture_type (str): Aperture type 'circle' or 'rectangle'
            aperture_size (list): Aperture size [width, height] (m)
        
        Returns:
            torch.Tensor: Complex field U0
        """
        self.focal_length = focal_length
        self.axi = axi
        self.alpha = alpha
        self.design_lambda = design_lambda
        self.lens_diameter = lens_diameter
        self.lens_center = lens_center
        self.aperture_type = aperture_type
        self.aperture_size = aperture_size
        
        reference_X = self.X - lens_center[0]
        reference_Y = self.Y - lens_center[1]
        
        # Create aperture
        aperture = self._create_aperture(aperture_type, aperture_size, [0, 0], reference_X, reference_Y)
        
        r = torch.sqrt((reference_X/(lens_diameter / 2))**2 + (reference_Y/(lens_diameter / 2))**2)
        # Calculate phase
        phase = axi * r + alpha * (torch.abs((reference_X/(lens_diameter / 2))**3) + torch.abs((reference_Y/(lens_diameter / 2))**3))
        phase = phase + phase_offset
        self.U0 = amplitude * torch.exp(1j * phase) * aperture
        return self.U0

    def rich_print(self):
        self.near_field.rich_print()
        table = Table(title="AXICON + CUBIC PHASE", show_header=True, header_style="bold magenta")
        table.add_column("Attribute", style="cyan", no_wrap=True)
        table.add_column("Value", style="green")
        table.add_column("Unit", style="yellow")
        table.add_column("type", style="yellow")
        table.add_column("device", style="yellow")
        table.add_row("Focal length", str(self.focal_length), "m", str(type(self.focal_length)), str(self.device))
        table.add_row("Axi", str(self.axi), "rad", str(type(self.axi)), str(self.device))
        table.add_row("Alpha", str(self.alpha), "rad", str(type(self.alpha)), str(self.device))
        table.add_row("Design lambda", str(self.design_lambda), "m", str(type(self.design_lambda)), str(self.device))
        table.add_row("Lens diameter", str(self.lens_diameter), "m", str(type(self.lens_diameter)), str(self.device))
        table.add_row("Lens center", str(self.lens_center), "m", str(type(self.lens_center)), str(self.device))
        table.add_row("Aperture type", str(self.aperture_type), "", str(type(self.aperture_type)), str(self.device))
        table.add_row("Aperture size", str(self.aperture_size), "m", str(type(self.aperture_size)), str(self.device))
        console = Console()
        console.print(table)

class CubicPhase(PhaseElement):
    """
    CUBIC PHASE
    """
    def calculate_phase(
        self, 
        focal_length: float = 1000e-6, 
        alpha: float = 0, 
        design_lambda: float = 0.94e-6, 
        lens_diameter: float = 1600e-6, 
        lens_center: list = [0, 0],
        aperture_type: str = 'circle',
        aperture_size: list = [1600e-6, 1600e-6],
        amplitude : float = 1,
        phase_offset : float = 0,
    ):
        """
        Calculate cubic phase distribution
        
        Args:
            focal_length (float): Focal length (m)
            alpha (float): Cubic phase parameter (rad)
            design_lambda (float): Design wavelength (m)
            lens_diameter (float): Lens diameter (m)
            lens_center (list): Lens center position [cx, cy] (m)
            aperture_type (str): Aperture type 'circle' or 'rectangle'
            aperture_size (list): Aperture size [width, height] (m)
        
        Returns:
            torch.Tensor: Complex field U0
        """
        self.focal_length = focal_length
        self.alpha = alpha
        self.design_lambda = design_lambda
        self.lens_diameter = lens_diameter
        self.lens_center = lens_center
        self.aperture_type = aperture_type
        self.aperture_size = aperture_size
        
        reference_X = self.X - lens_center[0]
        reference_Y = self.Y - lens_center[1]
        
        # Create aperture
        aperture = self._create_aperture(aperture_type, aperture_size, [0, 0], reference_X, reference_Y)
        
        # Calculate phase
        phase = -2 * torch.pi / design_lambda * \
            ((torch.sqrt(reference_X**2 + reference_Y**2 + focal_length**2) - focal_length) + 
             alpha * (torch.abs((reference_X/(lens_diameter / 2))**3) + torch.abs((reference_Y/(lens_diameter / 2))**3)))
        phase = phase + phase_offset
        self.U0 = amplitude * torch.exp(1j * phase) * aperture
        return self.U0
    
    def rich_print(self):
        self.near_field.rich_print()
        table = Table(title="CUBIC PHASE", show_header=True, header_style="bold magenta")
        table.add_column("Attribute", style="cyan", no_wrap=True)
        table.add_column("Value", style="green")
        table.add_column("Unit", style="yellow")
        table.add_column("type", style="yellow")
        table.add_column("device", style="yellow")
        table.add_row("Focal length", str(self.focal_length), "m", str(type(self.focal_length)), str(self.device))
        table.add_row("Alpha", str(self.alpha), "rad", str(type(self.alpha)), str(self.device))
        table.add_row("Design lambda", str(self.design_lambda), "m", str(type(self.design_lambda)), str(self.device))
        table.add_row("Lens diameter", str(self.lens_diameter), "m", str(type(self.lens_diameter)), str(self.device))
        table.add_row("Lens center", str(self.lens_center), "m", str(type(self.lens_center)), str(self.device))
        table.add_row("Aperture type", str(self.aperture_type), "", str(type(self.aperture_type)), str(self.device))
        table.add_row("Aperture size", str(self.aperture_size), "m", str(type(self.aperture_size)), str(self.device))
        console = Console()
        console.print(table)


class Binary1Phase(PhaseElement):
    """
    BINARY1 PHASE
    """
    def calculate_phase(
        self, 
        A1: float = 1, 
        lens_diameter: float = 1600e-6, 
        lens_center: list = [0, 0],
        aperture_type: str = 'circle',
        aperture_size: list = [1600e-6, 1600e-6],
        amplitude : float = 1,
        phase_offset : float = 0,
    ):
        """
        Calculate binary1 phase distribution
        
        Args:
            A1 (float): Binary1 phase coefficient
            lens_diameter (float): Lens diameter (m)
            lens_center (list): Lens center position [cx, cy] (m)
            aperture_type (str): Aperture type 'circle' or 'rectangle'
            aperture_size (list): Aperture size [width, height] (m)
        
        Returns:
            torch.Tensor: Complex field U0
        """
        self.A1 = A1
        self.lens_diameter = lens_diameter
        self.lens_center = lens_center
        self.aperture_type = aperture_type
        self.aperture_size = aperture_size
        
        reference_X = self.X - lens_center[0]
        reference_Y = self.Y - lens_center[1]
        
        # Create aperture
        aperture = self._create_aperture(aperture_type, aperture_size, [0, 0], reference_X, reference_Y)
        
        # Calculate phase
        phase = A1 * reference_X * 1e3
        phase = phase + phase_offset
        self.U0 = amplitude * torch.exp(1j * phase) * aperture  
        return self.U0
    
    def rich_print(self):
        self.near_field.rich_print()
        table = Table(title="BINARY1 PHASE", show_header=True, header_style="bold magenta")
        table.add_column("Attribute", style="cyan", no_wrap=True)
        table.add_column("Value", style="green")
        table.add_column("Unit", style="yellow")
        table.add_column("type", style="yellow")
        table.add_column("device", style="yellow")
        table.add_row("Lens diameter", str(self.lens_diameter), "m", str(type(self.lens_diameter)), str(self.device))
        table.add_row("Lens center", str(self.lens_center), "m", str(type(self.lens_center)), str(self.device))
        table.add_row("Aperture type", str(self.aperture_type), "", str(type(self.aperture_type)), str(self.device))
        table.add_row("Aperture size", str(self.aperture_size), "m", str(type(self.aperture_size)), str(self.device))
        console = Console()
        console.print(table)


class VSCELPhase(PhaseElement):
    """
    VSCEL PHASE
    """
    def calculate_phase(
        self, 
        design_lambda: float = 0.94e-6, 
        divergence_angle: float = 24, 
        vscel_center: list = [0, 0]
    ):
        """
        Calculate VSCEL phase distribution
        
        Args:
            design_lambda (float): Design wavelength (m)
            divergence_angle (float): Divergence angle (degree)
            vscel_center (list): VSCEL center position [cx, cy] (m)
        
        Returns:
            torch.Tensor: Complex field U0
        """
        self.design_lambda = design_lambda
        self.divergence_angle = divergence_angle
        self.vscel_center = vscel_center
        
        reference_X = self.X - vscel_center[0]
        reference_Y = self.Y - vscel_center[1]
        
        # Calculate theta, w0, and U0
        theta = torch.sin(torch.tensor(divergence_angle/180*torch.pi))
        w0 = design_lambda / (torch.pi * theta)
        self.U0 = torch.exp(-(reference_X**2 + reference_Y**2) / (w0**2))
        self.U0 = torch.sqrt(torch.abs(self.U0)**2 / torch.sum(torch.abs(self.U0)**2))
        return self.U0
    
    def rich_print(self):
        self.near_field.rich_print()
        table = Table(title="VSCEL PHASE", show_header=True, header_style="bold magenta")
        table.add_column("Attribute", style="cyan", no_wrap=True)
        table.add_column("Value", style="green")
        table.add_column("Unit", style="yellow")
        table.add_column("type", style="yellow")
        table.add_column("device", style="yellow")
        table.add_row("Design lambda", str(self.design_lambda), "m", str(type(self.design_lambda)), str(self.device))
        table.add_row("Divergence angle", str(self.divergence_angle), "degree", str(type(self.divergence_angle)), str(self.device))
        table.add_row("VSCEL center", str(self.vscel_center), "m", str(type(self.vscel_center)), str(self.device))
        console = Console()
        console.print(table)


class DiffractiveOpticsElement(PhaseElement):
    """
    DIFFRACTIVE OPTICS ELEMENT
    """
    def calculate_phase(
        self, 
        unit_cell: torch.Tensor, 
        xperiod: float = 5.04e-6, 
        yperiod: float = 5.04e-6,
        aperture_type: str = 'circle',
        aperture_size: list = [1600e-6, 1600e-6]
    ):
        """
        Calculate diffractive optics element phase distribution
        
        Args:
            unit_cell (torch.Tensor): Unit cell pattern
            xperiod (float): X direction period (m)
            yperiod (float): Y direction period (m)
        
        Returns:
            torch.Tensor: Complex field U0
        """
        unit_cell = unit_cell.to(self.device)
        self.unit_cell = unit_cell
        self.xperiod = xperiod
        self.yperiod = yperiod
        # Convert period (meters) to corresponding pixel counts
        x_period_pixels = int(torch.round(torch.tensor(xperiod / self.pixel_size)))
        y_period_pixels = int(torch.round(torch.tensor(yperiod / self.pixel_size)))

        # If unit_cell size != (x_period_pixels, y_period_pixels), resize first
        if unit_cell.shape != (x_period_pixels, y_period_pixels):
            # Expand to (N=1, C=1, H, W) for interpolate convenience
            unit_cell_4d = unit_cell.unsqueeze(0).unsqueeze(0)  # shape: [1, 1, H, W]
            
            # Resize using bilinear interpolation (can be 'nearest', 'bicubic', ...)
            unit_cell_4d_resized = F.interpolate(
                unit_cell_4d, 
                size=(x_period_pixels, y_period_pixels),
                mode='bilinear', 
                align_corners=False
            )
            
            # Squeeze back to 2D (H, W)
            unit_cell = unit_cell_4d_resized.squeeze(0).squeeze(0)

        # Compute number of repetitions along x and y
        repeat_x = self.Nx // x_period_pixels + 1
        repeat_y = self.Ny // y_period_pixels + 1

        # Tile the unit pattern
        tiled = unit_cell.repeat(repeat_x, repeat_y)

        # Crop to (Nx, Ny)
        pattern = tiled[:self.Nx, :self.Ny]
        # aperture
        aperture = self._create_aperture(aperture_type, aperture_size, [0, 0], self.X, self.Y)
        self.U0 = torch.exp(1j *1.0* pattern*torch.pi) * aperture
        return self.U0
    
    def rich_print(self):
        self.near_field.rich_print()
        table = Table(title="DIFFRACTIVE OPTICS ELEMENT", show_header=True, header_style="bold magenta")
        table.add_column("Attribute", style="cyan", no_wrap=True)
        table.add_column("Value", style="green")
        table.add_column("Unit", style="yellow")
        table.add_column("type", style="yellow")
        table.add_column("device", style="yellow")
        table.add_row("Unit cell shape", str(self.unit_cell.shape), "pixels", str(type(self.unit_cell)), str(self.device))
        table.add_row("X period", str(self.xperiod), "m", str(type(self.xperiod)), str(self.device))
        table.add_row("Y period", str(self.yperiod), "m", str(type(self.yperiod)), str(self.device))
        console = Console()
        console.print(table)


class IncidentField(PhaseElement):
    """
    INCIDENT FIELD
    """
    def calculate_phase(
        self, 
        incident_angle: list = [0, 0],
        n: float = 1, 
        design_lambda: float = 0.94e-6, 
        lens_center: list = [0, 0],
        aperture_type: str = 'circle',
        aperture_size: list = [1600e-6, 1600e-6],
        amplitude : float = 1,
        phase_offset : float = 0,
    ):
        """
        Calculate incident field
        
        Args:
            incident_angle (list): Incident angle [theta_x, theta_y] (degree)
            n (float): Refractive index of medium
            design_lambda (float): Design wavelength (m)
            lens_diameter (float): Lens diameter (m)
            lens_center (list): Lens center position [cx, cy] (m)
            aperture_type (str): Aperture type 'circle' or 'rectangle'
            aperture_size (list): Aperture size [width, height] (m)
        
        Returns:
            torch.Tensor: Complex field U0
        """
        self.incident_angle = incident_angle
        self.n = n
        self.design_lambda = design_lambda
        self.lens_center = lens_center
        self.aperture_type = aperture_type
        self.aperture_size = aperture_size
        
        reference_X = self.X - lens_center[0]
        reference_Y = self.Y - lens_center[1]
        
        # Create aperture
        aperture = self._create_aperture(aperture_type, aperture_size, lens_center, reference_X, reference_Y)
        
        # Calculate phase
        phase = -2 * torch.pi / design_lambda * n * -torch.sin(torch.tensor(incident_angle[0])/180*torch.pi) * reference_X \
                -2 * torch.pi / design_lambda * n * -torch.sin(torch.tensor(incident_angle[1])/180*torch.pi) * reference_Y
        phase = phase + phase_offset
        self.U0 = amplitude * torch.exp(1j * phase) * aperture
        return self.U0
    
    def rich_print(self):
        self.near_field.rich_print()
        table = Table(title="INCIDENT FIELD", show_header=True, header_style="bold magenta")
        table.add_column("Attribute", style="cyan", no_wrap=True)
        table.add_column("Value", style="green")
        table.add_column("Unit", style="yellow")
        table.add_column("type", style="yellow")
        table.add_column("device", style="yellow")
        table.add_row("Incident angle", str(self.incident_angle), "degree", str(type(self.incident_angle)), str(self.device))
        table.add_row("Refractive index", str(self.n), "", str(type(self.n)), str(self.device))
        table.add_row("Design lambda", str(self.design_lambda), "m", str(type(self.design_lambda)), str(self.device))
        table.add_row("Lens center", str(self.lens_center), "m", str(type(self.lens_center)), str(self.device))
        table.add_row("Aperture type", str(self.aperture_type), "", str(type(self.aperture_type)), str(self.device))
        table.add_row("Aperture size", str(self.aperture_size), "m", str(type(self.aperture_size)), str(self.device))
        console = Console()
        console.print(table)


class GaussianBeamSource(PhaseElement):
    """
    GAUSSIAN BEAM SOURCE

    Generates a TEM00 Gaussian beam field at the beam waist plane.
    The complex field is:
        U(x,y) = amplitude * exp(-(x^2 + y^2) / w0^2) * aperture * exp(j * phase_offset)

    Methods:
        calculate_phase(design_lambda, beam_center, gaussian_beam_waist_size,
                        aperture_type, aperture_size, amplitude, phase_offset) -> torch.Tensor:
            Calculate the Gaussian beam field distribution
        rich_print() -> None:
            Print the Gaussian beam source parameters
        draw(...) -> None:
            Draw the Gaussian beam field (inherited from PhaseElement)
    """
    def calculate_phase(
        self,
        design_lambda: float = 0.94e-6,
        beam_center: list = [0, 0],
        gaussian_beam_waist_size: float = 100e-6,
        aperture_type: str = 'circle',
        aperture_size: list = [1600e-6, 1600e-6],
        amplitude: float = 1,
        phase_offset: float = 0,
    ):
        """
        Calculate Gaussian beam field at the waist plane

        Args:
            design_lambda (float): Wavelength (m)
            beam_center (list): Beam center position [cx, cy] (m)
            gaussian_beam_waist_size (float): Gaussian beam waist radius w0 (m).
                The beam intensity drops to 1/e^2 at distance w0 from center.
            aperture_type (str): Aperture type 'circle' or 'rectangle'
            aperture_size (list): Aperture size [width, height] or [diameter] (m)
            amplitude (float): Peak amplitude of the beam
            phase_offset (float): Additional constant phase offset (rad)

        Returns:
            torch.Tensor: Complex field U0
        """
        if gaussian_beam_waist_size <= 0:
            raise ValueError("gaussian_beam_waist_size must be positive.")

        self.design_lambda = design_lambda
        self.beam_center = beam_center
        self.gaussian_beam_waist_size = gaussian_beam_waist_size
        self.aperture_type = aperture_type
        self.aperture_size = aperture_size

        reference_X = self.X - beam_center[0]
        reference_Y = self.Y - beam_center[1]

        # Create aperture
        aperture = self._create_aperture(
            aperture_type, aperture_size, beam_center, reference_X, reference_Y
        )

        # Gaussian beam amplitude profile at waist: exp(-(x^2+y^2)/w0^2)
        w0 = gaussian_beam_waist_size
        gaussian_amplitude = torch.exp(
            -(reference_X**2 + reference_Y**2) / (w0**2)
        )

        # Construct complex field
        self.U0 = (
            amplitude
            * gaussian_amplitude
            * torch.exp(1j * torch.tensor(phase_offset, dtype=self.dtype, device=self.device))
            * aperture
        )
        return self.U0

    def rich_print(self):
        self.near_field.rich_print()
        table = Table(title="GAUSSIAN BEAM SOURCE", show_header=True, header_style="bold magenta")
        table.add_column("Attribute", style="cyan", no_wrap=True)
        table.add_column("Value", style="green")
        table.add_column("Unit", style="yellow")
        table.add_column("type", style="yellow")
        table.add_column("device", style="yellow")
        table.add_row("Design lambda", str(self.design_lambda), "m", str(type(self.design_lambda)), str(self.device))
        table.add_row("Beam center", str(self.beam_center), "m", str(type(self.beam_center)), str(self.device))
        table.add_row("Waist size (w0)", str(self.gaussian_beam_waist_size), "m", str(type(self.gaussian_beam_waist_size)), str(self.device))
        table.add_row("Aperture type", str(self.aperture_type), "", str(type(self.aperture_type)), str(self.device))
        table.add_row("Aperture size", str(self.aperture_size), "m", str(type(self.aperture_size)), str(self.device))
        console = Console()
        console.print(table)


class ZemaxPOPSource(PhaseElement):
    """
    ZEMAX POP SOURCE

    Loads a source field from Zemax Physical Optics Propagation (POP) text
    listings ("Listing of POP Irradiance Data" and "Listing of POP Phase Data
    in radians") and resamples it onto the NearField grid.

    The complex field is reconstructed as:
        U(x,y) = sqrt(I(x,y)) * exp(j * phase(x,y))
    and bilinearly interpolated (real and imaginary parts separately, to
    avoid phase-wrapping artifacts) onto the simulation grid. Points outside
    the source data window are set to zero.

    Methods:
        calculate_phase(intensity_file, phase_file, source_center,
                        normalize_power, flip_y) -> torch.Tensor:
            Load, reconstruct and resample the source field
        rich_print() -> None:
            Print the source file metadata and grid parameters
        draw(...) -> None:
            Draw the source field (inherited from PhaseElement)
    """

    @staticmethod
    def _parse_pop_file(file_path: str):
        """
        Parse a Zemax POP text listing (UTF-16 encoded).

        Args:
            file_path (str): Path to the POP irradiance or phase listing

        Returns:
            tuple: (data, header) where data is a numpy array of shape
                [Ny, Nx] and header is a dict with keys 'Nx', 'Ny',
                'dx' (m), 'dy' (m), 'wavelength' (m or None),
                'index' (float or None).
        """
        import re
        import numpy as np

        with open(file_path, encoding='utf-16') as f:
            lines = f.readlines()

        header = {'Nx': None, 'Ny': None, 'dx': None, 'dy': None,
                  'wavelength': None, 'index': None}
        data_start = None
        for i, line in enumerate(lines):
            s = line.strip()
            if s.startswith('Grid size'):
                nx, ny = re.findall(r'(\d+)\s*by\s*(\d+)', s)[0]
                header['Nx'], header['Ny'] = int(nx), int(ny)
            elif s.startswith('Point spacing'):
                dx, dy = re.findall(r'([\d.E+-]+)\s*by\s*([\d.E+-]+)', s)[0]
                header['dx'], header['dy'] = float(dx) * 1e-3, float(dy) * 1e-3  # mm -> m
            elif s.startswith('Beam wavelength'):
                m = re.search(r'wavelength is\s*([\d.E+-]+).*index\s*([\d.E+-]+)', s)
                if m:
                    header['wavelength'] = float(m.group(1)) * 1e-6  # um -> m
                    header['index'] = float(m.group(2))
            elif header['Nx'] is not None and s and data_start is None:
                parts = s.split()
                if len(parts) == header['Nx']:
                    try:
                        float(parts[0])
                        data_start = i
                    except ValueError:
                        pass
        if header['Nx'] is None or data_start is None:
            raise ValueError(f"Could not parse Zemax POP listing: {file_path}")

        rows = []
        for line in lines[data_start:]:
            parts = line.split()
            if len(parts) == header['Nx']:
                rows.append(np.array(parts, dtype=np.float64))
        data = np.stack(rows)
        if data.shape != (header['Ny'], header['Nx']):
            raise ValueError(
                f"Data shape {data.shape} does not match grid size "
                f"({header['Ny']}, {header['Nx']}) in {file_path}"
            )
        return data, header

    def calculate_phase(
        self,
        intensity_file: str = None,
        phase_file: str = None,
        source_center: list = [0, 0],
        normalize_power: bool = False,
        flip_y: bool = False,
    ):
        """
        Load a Zemax POP source and resample it onto the NearField grid

        Args:
            intensity_file (str): Path to the POP irradiance listing (.txt)
            phase_file (str): Path to the POP phase listing (.txt) in
                radians. If None, a flat phase is assumed.
            source_center (list): Position [cx, cy] (m) to place the source
                center on the simulation grid
            normalize_power (bool): If True, normalize the field to unit
                total power on the simulation grid
            flip_y (bool): If True, flip the source data along Y (use when
                the listing row order is opposite to the grid convention)

        Returns:
            torch.Tensor: Complex field U0
        """
        import numpy as np

        if intensity_file is None:
            raise ValueError("intensity_file is required.")

        self.intensity_file = intensity_file
        self.phase_file = phase_file
        self.source_center = source_center
        self.normalize_power = normalize_power
        self.flip_y = flip_y

        intensity, header = self._parse_pop_file(intensity_file)
        if phase_file is not None:
            phase, _ = self._parse_pop_file(phase_file)
            if phase.shape != intensity.shape:
                raise ValueError(
                    f"Phase grid {phase.shape} does not match "
                    f"irradiance grid {intensity.shape}"
                )
        else:
            phase = np.zeros_like(intensity)

        if flip_y:
            intensity = intensity[::-1, :].copy()
            phase = phase[::-1, :].copy()

        self.source_Nx = header['Nx']
        self.source_Ny = header['Ny']
        self.source_dx = header['dx']
        self.source_dy = header['dy']
        self.source_wavelength = header['wavelength']
        self.source_index = header['index']
        self.source_total_power = float(intensity.sum() * header['dx'] * header['dy'])

        # Reconstruct the complex field on the source grid
        amplitude = np.sqrt(np.clip(intensity, 0, None))
        U_src = torch.tensor(
            amplitude * np.exp(1j * phase),
            dtype=torch.complex64 if self.dtype == torch.float32 else torch.complex128,
            device=self.device,
        )

        # Bilinear resampling of real/imag parts onto the NearField grid
        # (grid_sample expects normalized coordinates in [-1, 1] that map to
        # the outermost pixel centers with align_corners=True)
        half_wx = (self.source_Nx - 1) / 2 * self.source_dx
        half_wy = (self.source_Ny - 1) / 2 * self.source_dy
        grid_x = (self.X - source_center[0]) / half_wx
        grid_y = (self.Y - source_center[1]) / half_wy
        sample_grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0)
        src_stack = torch.stack([U_src.real, U_src.imag]).unsqueeze(0).to(self.dtype)
        resampled = F.grid_sample(
            src_stack, sample_grid.to(self.dtype),
            mode='bilinear', padding_mode='zeros', align_corners=True,
        ).squeeze(0)
        self.U0 = torch.complex(resampled[0], resampled[1])

        if normalize_power:
            power = (torch.abs(self.U0) ** 2).sum() * self.pixel_size ** 2
            if power > 0:
                self.U0 = self.U0 / torch.sqrt(power)

        return self.U0

    def rich_print(self):
        self.near_field.rich_print()
        table = Table(title="ZEMAX POP SOURCE", show_header=True, header_style="bold magenta")
        table.add_column("Attribute", style="cyan", no_wrap=True)
        table.add_column("Value", style="green")
        table.add_column("Unit", style="yellow")
        table.add_column("type", style="yellow")
        table.add_column("device", style="yellow")
        table.add_row("Intensity file", str(self.intensity_file), "", str(type(self.intensity_file)), str(self.device))
        table.add_row("Phase file", str(self.phase_file), "", str(type(self.phase_file)), str(self.device))
        table.add_row("Source grid", f"{self.source_Nx} x {self.source_Ny}", "points", str(type(self.source_Nx)), str(self.device))
        table.add_row("Source spacing", f"{self.source_dx:.4e} x {self.source_dy:.4e}", "m", str(type(self.source_dx)), str(self.device))
        table.add_row("Source wavelength", str(self.source_wavelength), "m", str(type(self.source_wavelength)), str(self.device))
        table.add_row("Medium index", str(self.source_index), "", str(type(self.source_index)), str(self.device))
        table.add_row("Total power (file)", f"{self.source_total_power:.4e}", "W", str(type(self.source_total_power)), str(self.device))
        table.add_row("Source center", str(self.source_center), "m", str(type(self.source_center)), str(self.device))
        table.add_row("Normalize power", str(self.normalize_power), "", str(type(self.normalize_power)), str(self.device))
        console = Console()
        console.print(table)


