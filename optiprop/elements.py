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
