import torch
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import rcParams
from rich.table import Table
from rich.console import Console
from .utils import check_available_cuda, make_colorbar_with_padding
from .elements import PhaseElement


def _wrap_phase(phase):
    """Wrap phase to the interval [-pi, pi).

    Args:
        phase (torch.Tensor): Phase values (rad).

    Returns:
        torch.Tensor: Wrapped phase values (rad).
    """
    return torch.remainder(phase + torch.pi, 2 * torch.pi) - torch.pi


class MetaAtomLibrary:
    """
    META-ATOM LIBRARY

    Loads a meta-atom database (.npy dictionary with a 'data_sheet' entry) and
    provides nearest wrapped-phase lookup so that an ideal phase profile can be
    replaced by the realizable amplitude/phase of fabricated meta-atoms.

    Expected database layout (data_sheet dictionary):
        shape_type (str): Meta-atom shape, e.g. 'circle'
        Dimension_name (list): Names of the sweep dimensions
        Wavelength, Period, Thickness (1-D arrays): Sweep axes (nm)
        R (1-D array): Structure parameter axis (nm)
        transmission_tensor (5-D array): [wavelength, period, thickness, R, polarization]
        phase_tensor (5-D array): [wavelength, period, thickness, R, polarization] (rad)

    Methods:
        lookup(target_phase, alpha) -> (index_map, phase_map, amplitude_map):
            Nearest wrapped-phase lookup for arbitrary target phase tensors
        rich_print() -> None:
            Print library metadata and phase coverage statistics
        plot(...) -> None:
            Plot structure parameter vs phase and vs amplitude
    """

    def __init__(
        self,
        library_path: str,
        wavelength: float = None,
        period: float = None,
        thickness: float = None,
        polarization_index: int = 0,
        transmission_type: str = 'intensity',
        dtype: torch.dtype = torch.float32,
        device: str = 'cpu'
    ):
        """
        Initialize the meta-atom library.

        Args:
            library_path (str): Path to the .npy database file.
            wavelength (float): Desired wavelength, same unit as the database
                axis (nm). Nearest entry is selected; None selects index 0.
            period (float): Desired lattice period (nm). Nearest entry is
                selected; None selects index 0.
            thickness (float): Desired meta-atom thickness (nm). Nearest entry
                is selected; None selects index 0.
            polarization_index (int): Column of the polarization axis to use
                (0 = Txx co-polarization, valid for isotropic meta-atoms).
            transmission_type (str): 'intensity' if the transmission tensor
                stores |t|^2 (field amplitude = sqrt(T)), 'amplitude' if it
                already stores the field amplitude |t|.
            dtype (torch.dtype): torch.float32 or torch.float64.
            device (str): 'cpu' or 'cuda'.
        """
        if transmission_type not in ['intensity', 'amplitude']:
            raise ValueError("transmission_type must be 'intensity' or 'amplitude'.")
        if dtype not in [torch.float32, torch.float64]:
            raise ValueError("Data type must be torch.float32 or torch.float64.")

        self.device = check_available_cuda(device)
        self.dtype = dtype
        self.library_path = library_path
        self.polarization_index = polarization_index
        self.transmission_type = transmission_type

        data_sheet = np.load(library_path, allow_pickle=True).item()['data_sheet']
        self.shape_type = data_sheet['shape_type']
        self.dimension_name = list(data_sheet['Dimension_name'])

        wavelength_axis = np.atleast_1d(np.asarray(data_sheet['Wavelength'], dtype=np.float64))
        period_axis = np.atleast_1d(np.asarray(data_sheet['Period'], dtype=np.float64))
        thickness_axis = np.atleast_1d(np.asarray(data_sheet['Thickness'], dtype=np.float64))

        iw = self._nearest_index(wavelength_axis, wavelength)
        ip = self._nearest_index(period_axis, period)
        it = self._nearest_index(thickness_axis, thickness)

        self.wavelength = float(wavelength_axis[iw])
        self.period = float(period_axis[ip])
        self.thickness = float(thickness_axis[it])

        transmission = np.asarray(data_sheet['transmission_tensor'])[iw, ip, it, :, polarization_index]
        phase = np.asarray(data_sheet['phase_tensor'])[iw, ip, it, :, polarization_index]
        parameter = np.asarray(data_sheet['R'], dtype=np.float64)

        if transmission_type == 'intensity':
            amplitude = np.sqrt(np.clip(transmission, 0, None))
        else:
            amplitude = transmission

        self.parameter = torch.tensor(parameter, dtype=dtype, device=self.device)
        self.amplitude = torch.tensor(amplitude, dtype=dtype, device=self.device)
        self.phase = _wrap_phase(torch.tensor(phase, dtype=dtype, device=self.device))
        self.n_atoms = self.parameter.numel()

        # Phase coverage statistics on the unit circle
        sorted_phase, _ = torch.sort(self.phase)
        if self.n_atoms > 1:
            gaps = torch.diff(sorted_phase)
            wrap_gap = 2 * torch.pi - (sorted_phase[-1] - sorted_phase[0])
            all_gaps = torch.cat([gaps, wrap_gap.reshape(1)])
            self.largest_gap = all_gaps.max().item()
        else:
            self.largest_gap = 2 * torch.pi
        self.phase_coverage = 2 * torch.pi - self.largest_gap
        self.worst_lookup_error = self.largest_gap / 2

    @staticmethod
    def _nearest_index(axis: np.ndarray, value) -> int:
        """
        Find the nearest index of `value` in a 1-D axis.

        Args:
            axis (np.ndarray): 1-D sweep axis.
            value (float): Requested value; None selects index 0.

        Returns:
            int: Nearest index.
        """
        if value is None or axis.size == 1:
            return 0
        return int(np.argmin(np.abs(axis - value)))

    def lookup(
        self,
        target_phase: torch.Tensor,
        alpha: float = 0.0
    ):
        """
        Vectorized nearest wrapped-phase lookup.

        For each target phase value the meta-atom minimizing
        wrap(phi_db - phi_target)^2 + alpha * (1 - amp_db)^2 is selected,
        where wrap(x) = (x + pi) mod 2*pi - pi.

        Args:
            target_phase (torch.Tensor): Target phase (rad), any shape.
            alpha (float): Weight of the amplitude penalty term. 0 gives a
                pure nearest-phase lookup; larger values avoid low-transmission
                (resonant) meta-atoms at the cost of a larger phase error.

        Returns:
            tuple: (index_map, phase_map, amplitude_map)
                index_map (torch.LongTensor): Selected library index per pixel.
                phase_map (torch.Tensor): Realized phase (rad) per pixel.
                amplitude_map (torch.Tensor): Realized field amplitude per pixel.
        """
        original_shape = target_phase.shape
        target = target_phase.to(device=self.device, dtype=self.dtype).reshape(-1)

        amp_penalty = alpha * (1 - self.amplitude) ** 2  # [NR]
        n_pixels = target.numel()
        n_atoms = self.n_atoms

        # Chunk over pixels to bound the [chunk, NR] cost matrix to ~2e8 elements
        max_elements = int(2e8)
        chunk_size = max(1, max_elements // max(n_atoms, 1))

        index_flat = torch.empty(n_pixels, dtype=torch.long, device=self.device)
        for start in range(0, n_pixels, chunk_size):
            stop = min(start + chunk_size, n_pixels)
            diff = _wrap_phase(self.phase.unsqueeze(0) - target[start:stop].unsqueeze(1))
            cost = diff ** 2 + amp_penalty.unsqueeze(0)
            index_flat[start:stop] = torch.argmin(cost, dim=1)

        index_map = index_flat.reshape(original_shape)
        phase_map = self.phase[index_map]
        amplitude_map = self.amplitude[index_map]
        return index_map, phase_map, amplitude_map

    def rich_print(self):
        table = Table(title="META-ATOM LIBRARY", show_header=True, header_style="bold magenta")
        table.add_column("Attribute", style="cyan", no_wrap=True)
        table.add_column("Value", style="green")
        table.add_column("Unit", style="yellow")
        table.add_column("type", style="yellow")
        table.add_column("device", style="yellow")
        table.add_row("Library path", str(self.library_path), "", str(type(self.library_path)), str(self.device))
        table.add_row("Shape type", str(self.shape_type), "", str(type(self.shape_type)), str(self.device))
        table.add_row("Dimension names", str(self.dimension_name), "", str(type(self.dimension_name)), str(self.device))
        table.add_row("Wavelength", str(self.wavelength), "nm", str(type(self.wavelength)), str(self.device))
        table.add_row("Period", str(self.period), "nm", str(type(self.period)), str(self.device))
        table.add_row("Thickness", str(self.thickness), "nm", str(type(self.thickness)), str(self.device))
        table.add_row("Polarization index", str(self.polarization_index), "", str(type(self.polarization_index)), str(self.device))
        table.add_row("Transmission type", str(self.transmission_type), "", str(type(self.transmission_type)), str(self.device))
        table.add_row("Number of meta-atoms", str(self.n_atoms), "", str(type(self.n_atoms)), str(self.device))
        table.add_row("Parameter range", f"{self.parameter.min().item():.1f} ~ {self.parameter.max().item():.1f}", "nm", str(type(self.parameter)), str(self.device))
        table.add_row("Amplitude range", f"{self.amplitude.min().item():.4f} ~ {self.amplitude.max().item():.4f}", "", str(type(self.amplitude)), str(self.device))
        table.add_row("Phase coverage", f"{self.phase_coverage:.4f} ({self.phase_coverage / (2 * torch.pi) * 100:.1f}% of 2π)", "rad", str(type(self.phase)), str(self.device))
        table.add_row("Largest phase gap", f"{self.largest_gap:.4f}", "rad", str(type(self.largest_gap)), str(self.device))
        table.add_row("Worst-case lookup error", f"{self.worst_lookup_error:.4f}", "rad", str(type(self.worst_lookup_error)), str(self.device))
        console = Console()
        console.print(table)

    def plot(
        self,
        fontsize: int = 16,
        dark_style: bool = False,
        title: str = None,
        show: bool = True,
        save_path: str = None
    ):
        """
        Plot structure parameter vs phase and vs amplitude.

        Args:
            fontsize (int): The font size.
            dark_style (bool): 'True' for dark style, 'False' for light style.
            title (str): The figure title.
            show (bool): Whether to show the plot.
            save_path (str): If given, save the figure to this path.
        """
        parameter = self.parameter.cpu().numpy()
        phase = self.phase.cpu().numpy()
        amplitude = self.amplitude.cpu().numpy()

        if dark_style:
            plt.style.use('dark_background')
        else:
            plt.style.use('default')
        FONT_FAMILY = 'serif'
        rcParams['font.weight'] = 'bold'
        rcParams['font.size'] = fontsize
        rcParams['font.family'] = FONT_FAMILY

        fig = plt.figure("Meta-atom library", figsize=(14, 6))
        if title is not None:
            fig.suptitle(title, fontsize=fontsize, fontweight='bold', fontname=FONT_FAMILY)

        ax1 = fig.add_subplot(121)
        ax1.plot(parameter, phase, 'o-')
        ax1.set_xlabel('R (nm)', fontsize=fontsize, fontweight='bold', fontname=FONT_FAMILY)
        ax1.set_ylabel('Phase (rad)', fontsize=fontsize, fontweight='bold', fontname=FONT_FAMILY)
        ax1.set_title("Phase", fontsize=fontsize, fontweight='bold', fontname=FONT_FAMILY)
        ax1.grid(True, alpha=0.3)

        ax2 = fig.add_subplot(122)
        ax2.plot(parameter, amplitude, 'o-')
        ax2.set_xlabel('R (nm)', fontsize=fontsize, fontweight='bold', fontname=FONT_FAMILY)
        ax2.set_ylabel('Amplitude', fontsize=fontsize, fontweight='bold', fontname=FONT_FAMILY)
        ax2.set_title("Amplitude", fontsize=fontsize, fontweight='bold', fontname=FONT_FAMILY)
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        if save_path is not None:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        if show:
            plt.show()
        else:
            plt.close()


class MetaAtomElement(PhaseElement):
    """
    META-ATOM ELEMENT

    Replaces the ideal phase/amplitude of a phase element with the realizable
    values from a MetaAtomLibrary using nearest wrapped-phase lookup, and
    exposes the structure parameter map (R per pixel) for fabrication.

    Methods:
        calculate_phase(ideal_element, ideal_U0, library, alpha,
                        optimize_global_offset, offset_steps,
                        keep_ideal_amplitude) -> torch.Tensor:
            Compute the realized complex field U0
        rich_print() -> None:
            Print library info and phase-error statistics
        draw(...) -> None:
            Draw the realized field (inherited from PhaseElement)
        draw_parameter_map(...) -> None:
            Draw the structure parameter map (R in nm)
        draw_phase_error(...) -> None:
            Draw the wrapped phase-error map (rad)
        export_parameter_map(path) -> None:
            Export the structure parameter map for fabrication
    """

    def calculate_phase(
        self,
        ideal_element: PhaseElement = None,
        ideal_U0: torch.Tensor = None,
        library: MetaAtomLibrary = None,
        alpha: float = 0.0,
        optimize_global_offset: bool = False,
        offset_steps: int = 360,
        keep_ideal_amplitude: bool = True,
    ):
        """
        Replace an ideal complex field with realizable meta-atom values.

        Args:
            ideal_element (PhaseElement): Element whose U0 provides the ideal
                target field. Ignored if ideal_U0 is given.
            ideal_U0 (torch.Tensor): Ideal complex field [Nx, Ny]. Used
                instead of ideal_element.U0 when provided.
            library (MetaAtomLibrary): Meta-atom library used for the lookup.
            alpha (float): Amplitude penalty weight in the lookup cost.
            optimize_global_offset (bool): If True, scan a constant phase
                offset in [0, 2*pi) and add the offset minimizing the total
                squared wrapped phase error inside the aperture.
            offset_steps (int): Number of offset samples in [0, 2*pi).
            keep_ideal_amplitude (bool): If True the ideal amplitude
                multiplies the realized meta-atom amplitude; if False a flat
                unit amplitude (inside the aperture) is used instead.

        Returns:
            torch.Tensor: Realized complex field U0.
        """
        if library is None:
            raise ValueError("A MetaAtomLibrary must be provided via 'library'.")
        if ideal_U0 is None:
            if ideal_element is None or ideal_element.U0 is None:
                raise ValueError(
                    "Provide 'ideal_U0' or an 'ideal_element' whose "
                    "calculate_phase() has been called."
                )
            ideal_U0 = ideal_element.U0

        ideal_U0 = ideal_U0.to(self.device)
        complex_dtype = ideal_U0.dtype if torch.is_complex(ideal_U0) else (
            torch.complex64 if self.dtype == torch.float32 else torch.complex128
        )
        real_dtype = torch.float64 if complex_dtype == torch.complex128 else torch.float32

        target_phase = torch.angle(ideal_U0).to(real_dtype)
        ideal_amp = torch.abs(ideal_U0).to(real_dtype)
        aperture = ideal_amp > 0

        # Optional global constant phase offset optimization (phase-only cost)
        global_offset = 0.0
        if optimize_global_offset:
            global_offset = self._optimize_global_offset(
                target_phase[aperture], library, offset_steps
            )

        lookup_target = _wrap_phase(target_phase + global_offset)
        index_map, phase_map, amplitude_map = library.lookup(lookup_target, alpha=alpha)

        index_map = index_map.to(self.device)
        realized_phase = phase_map.to(device=self.device, dtype=real_dtype)
        realized_amplitude = amplitude_map.to(device=self.device, dtype=real_dtype)

        # Mask everything outside the aperture
        aperture_f = aperture.to(real_dtype)
        realized_phase = realized_phase * aperture_f
        realized_amplitude = realized_amplitude * aperture_f
        parameter_map = library.parameter.to(device=self.device, dtype=real_dtype)[index_map] * aperture_f

        base_amplitude = ideal_amp if keep_ideal_amplitude else aperture_f
        self.U0 = (base_amplitude * realized_amplitude
                   * torch.exp(1j * realized_phase)).to(complex_dtype)

        # Store results for inspection and export
        self.library = library
        self.alpha = alpha
        self.global_offset = float(global_offset)
        self.keep_ideal_amplitude = keep_ideal_amplitude
        self.aperture = aperture
        self.ideal_phase = target_phase
        self.realized_phase = realized_phase
        self.realized_amplitude = realized_amplitude
        self.parameter_map = parameter_map
        self.index_map = index_map
        self.phase_error = _wrap_phase(realized_phase - lookup_target) * aperture_f
        return self.U0

    def _optimize_global_offset(
        self,
        target_phase_in_aperture: torch.Tensor,
        library: MetaAtomLibrary,
        offset_steps: int
    ) -> float:
        """
        Scan a constant phase offset minimizing the total squared wrapped
        phase error of the nearest-phase lookup inside the aperture.

        Args:
            target_phase_in_aperture (torch.Tensor): 1-D target phases (rad).
            library (MetaAtomLibrary): Meta-atom library.
            offset_steps (int): Number of offsets sampled in [0, 2*pi).

        Returns:
            float: Optimal global phase offset (rad).
        """
        target = target_phase_in_aperture.to(device=library.device, dtype=library.dtype).reshape(-1)
        phase_db = library.phase

        # Histogram the target phases so the offset scan cost is independent
        # of the number of pixels: error(bin, offset) is evaluated on bin
        # centers and weighted by bin counts.
        n_bins = max(4 * offset_steps, 720)
        bin_edges = torch.linspace(-torch.pi, torch.pi, n_bins + 1,
                                   dtype=library.dtype, device=library.device)
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        counts = torch.histc(_wrap_phase(target), bins=n_bins, min=-torch.pi, max=torch.pi)

        offsets = torch.arange(offset_steps, dtype=library.dtype, device=library.device) \
            * (2 * torch.pi / offset_steps)

        # [n_bins, offset_steps] shifted targets, min over NR library phases
        shifted = _wrap_phase(bin_centers.unsqueeze(1) + offsets.unsqueeze(0))
        diff = _wrap_phase(phase_db.reshape(-1, 1, 1) - shifted.unsqueeze(0))
        min_sq_error, _ = (diff ** 2).min(dim=0)  # [n_bins, offset_steps]
        total_error = (counts.unsqueeze(1) * min_sq_error).sum(dim=0)
        best = torch.argmin(total_error)
        return float(offsets[best].item())

    def rich_print(self):
        if self.U0 is None:
            raise ValueError("Please calculate phase first by calling calculate_phase()")
        aperture = self.aperture
        n_pixels = int(aperture.sum().item())
        err = self.phase_error[aperture]
        amp = self.realized_amplitude[aperture]
        mean_err = err.abs().mean().item()
        rms_err = torch.sqrt((err ** 2).mean()).item()
        max_err = err.abs().max().item()
        rad2deg = 180 / torch.pi

        self.near_field.rich_print()
        table = Table(title="META-ATOM ELEMENT", show_header=True, header_style="bold magenta")
        table.add_column("Attribute", style="cyan", no_wrap=True)
        table.add_column("Value", style="green")
        table.add_column("Unit", style="yellow")
        table.add_column("type", style="yellow")
        table.add_column("device", style="yellow")
        table.add_row("Library path", str(self.library.library_path), "", str(type(self.library.library_path)), str(self.device))
        table.add_row("Shape type", str(self.library.shape_type), "", str(type(self.library.shape_type)), str(self.device))
        table.add_row("Wavelength", str(self.library.wavelength), "nm", str(type(self.library.wavelength)), str(self.device))
        table.add_row("Period", str(self.library.period), "nm", str(type(self.library.period)), str(self.device))
        table.add_row("Alpha", str(self.alpha), "", str(type(self.alpha)), str(self.device))
        table.add_row("Global offset", f"{self.global_offset:.4f}", "rad", str(type(self.global_offset)), str(self.device))
        table.add_row("Keep ideal amplitude", str(self.keep_ideal_amplitude), "", str(type(self.keep_ideal_amplitude)), str(self.device))
        table.add_row("Aperture pixels", str(n_pixels), "pixels", str(type(n_pixels)), str(self.device))
        table.add_row("Phase error (mean)", f"{mean_err:.4f} / {mean_err * rad2deg:.2f}", "rad / deg", str(type(self.phase_error)), str(self.device))
        table.add_row("Phase error (RMS)", f"{rms_err:.4f} / {rms_err * rad2deg:.2f}", "rad / deg", str(type(self.phase_error)), str(self.device))
        table.add_row("Phase error (max)", f"{max_err:.4f} / {max_err * rad2deg:.2f}", "rad / deg", str(type(self.phase_error)), str(self.device))
        table.add_row("Amplitude (min)", f"{amp.min().item():.4f}", "", str(type(amp)), str(self.device))
        table.add_row("Amplitude (mean)", f"{amp.mean().item():.4f}", "", str(type(amp)), str(self.device))
        console = Console()
        console.print(table)

    def _draw_map(
        self,
        data: torch.Tensor,
        colorbar_label: str,
        default_title: str,
        cmap: str,
        fontsize: int,
        xlim: list,
        ylim: list,
        dark_style: bool,
        title: str,
        show: bool,
        save_path: str
    ):
        """Shared imshow helper following the utils plotting style."""
        x = self.X.cpu().numpy() * 1e6
        y = self.Y.cpu().numpy() * 1e6
        data = data.cpu().numpy()

        if dark_style:
            plt.style.use('dark_background')
        else:
            plt.style.use('default')
        FONT_FAMILY = 'serif'
        rcParams['font.weight'] = 'bold'
        rcParams['font.size'] = fontsize
        rcParams['font.family'] = FONT_FAMILY

        fig = plt.figure(default_title, figsize=(7, 6))
        ax = fig.add_subplot(111)
        plt.imshow(data, extent=[x.min(), x.max(), y.min(), y.max()],
                   cmap=cmap, origin='lower', aspect='equal')
        plt.xlabel('x (µm)', fontsize=fontsize, fontweight='bold', fontname=FONT_FAMILY)
        plt.ylabel('y (µm)', fontsize=fontsize, fontweight='bold', fontname=FONT_FAMILY)
        plt.title(title if title is not None else default_title,
                  fontsize=fontsize, fontweight='bold', fontname=FONT_FAMILY)
        plt.xlim(xlim)
        plt.ylim(ylim)
        cax = make_colorbar_with_padding(ax)
        cb = plt.colorbar(cax=cax)
        cb.set_label(colorbar_label, fontsize=fontsize, fontweight='bold', fontname=FONT_FAMILY)
        plt.tight_layout()
        if save_path is not None:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        if show:
            plt.show()
        else:
            plt.close()

    def draw_parameter_map(
        self,
        cmap: str = 'viridis',
        fontsize: int = 16,
        xlim: list = None,
        ylim: list = None,
        dark_style: bool = False,
        title: str = None,
        show: bool = True,
        save_path: str = None
    ):
        """
        Draw the structure parameter map (R in nm).

        Args:
            cmap (str): The colormap to use.
            fontsize (int): The font size.
            xlim (list): The x-axis limits [xmin, xmax] (µm).
            ylim (list): The y-axis limits [ymin, ymax] (µm).
            dark_style (bool): 'True' for dark style, 'False' for light style.
            title (str): The title of the plot.
            show (bool): Whether to show the plot.
            save_path (str): If given, save the figure to this path.
        """
        if self.U0 is None:
            raise ValueError("Please calculate phase first by calling calculate_phase()")
        self._draw_map(self.parameter_map, 'R (nm)', "Structure parameter map",
                       cmap, fontsize, xlim, ylim, dark_style, title, show, save_path)

    def draw_phase_error(
        self,
        cmap: str = 'turbo',
        fontsize: int = 16,
        xlim: list = None,
        ylim: list = None,
        dark_style: bool = False,
        title: str = None,
        show: bool = True,
        save_path: str = None
    ):
        """
        Draw the wrapped phase-error map (rad).

        Args:
            cmap (str): The colormap to use.
            fontsize (int): The font size.
            xlim (list): The x-axis limits [xmin, xmax] (µm).
            ylim (list): The y-axis limits [ymin, ymax] (µm).
            dark_style (bool): 'True' for dark style, 'False' for light style.
            title (str): The title of the plot.
            show (bool): Whether to show the plot.
            save_path (str): If given, save the figure to this path.
        """
        if self.U0 is None:
            raise ValueError("Please calculate phase first by calling calculate_phase()")
        self._draw_map(self.phase_error, 'Phase error (rad)', "Phase error map",
                       cmap, fontsize, xlim, ylim, dark_style, title, show, save_path)

    def export_parameter_map(self, path: str):
        """
        Export the structure parameter map (R in nm per pixel) for fabrication.

        Args:
            path (str): Output file path. Saved as .npy if the path ends with
                '.npy', otherwise as comma-separated text (.csv).
        """
        if self.U0 is None:
            raise ValueError("Please calculate phase first by calling calculate_phase()")
        parameter_map = self.parameter_map.cpu().numpy()
        if path.endswith('.npy'):
            np.save(path, parameter_map)
        else:
            np.savetxt(path, parameter_map, delimiter=',', fmt='%.6g')
