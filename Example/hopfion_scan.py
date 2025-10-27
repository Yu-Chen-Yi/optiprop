import os
from tkinter import N
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import optiprop
import numpy as np
import os
import torch
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb

def polarization_color_map(Sx, Sy, Sz, *, sat_mode="transverse", val_mode="signed", clip=True):
    """
    Convert Sx,Sy,Sz (2D arrays) into an RGB image via HSV mapping.
    
    Parameters
    ----------
    Sx, Sy, Sz : 2D numpy arrays with the same shape.
    sat_mode : {"transverse","dop"}
        - "transverse": saturation = sqrt(Sx^2+Sy^2) / max(sqrt(Sx^2+Sy^2))
        - "dop": saturation = sqrt(Sx^2+Sy^2+Sz^2) / max(sqrt(Sx^2+Sy^2+Sz^2))
    val_mode : {"signed","intensity"}
        - "signed":  map Sz from [-max|Sz|, +max|Sz|] -> [0,1] (mid-grey at 0)
        - "intensity": map sqrt(Sx^2+Sy^2+Sz^2) to [0,1]
    clip : bool
        Clip H,S,V into [0,1] after normalization.
    """
    Sx = np.asarray(Sx, dtype=float)
    Sy = np.asarray(Sy, dtype=float)
    Sz = np.asarray(Sz, dtype=float)
    assert Sx.shape == Sy.shape == Sz.shape, "Sx, Sy, Sz must have the same shape"
    
    # Hue from atan2 in [0, 1]
    angle = np.arctan2(Sy, Sx)                     # [-π, π]
    H = (angle % (2*np.pi)) / (2*np.pi)            # [0,1)

    # Saturation
    if sat_mode == "dop":
        mag = np.sqrt(Sx*Sx + Sy*Sy + Sz*Sz)
    else:
        mag = np.sqrt(Sx*Sx + Sy*Sy)
    S = mag / (np.nanmax(mag) + 1e-12)

    # Value
    if val_mode == "intensity":
        val = np.sqrt(Sx*Sx + Sy*Sy + Sz*Sz)
        V = val / (np.nanmax(val) + 1e-12)
    else:  # "signed"
        m = np.nanmax(np.abs(Sz)) + 1e-12
        V = (Sz / m + 1.0) * 0.5  # [-m,m] -> [0,1]

    if clip:
        H = np.clip(H, 0, 1)
        S = np.clip(S, 0, 1)
        V = np.clip(V, 0, 1)
    
    HSV = np.stack([H, S, V], axis=-1)
    RGB = hsv_to_rgb(HSV)
    return RGB, (H, S, V)

def show_pcolor(Z, title):
    plt.figure()
    # pcolormesh with default grid (index-based). shading='auto' to match array shape
    pc = plt.pcolormesh(Z, shading='auto')
    plt.title(title)
    plt.colorbar(pc)
    plt.tight_layout()
    plt.show()

def show_rgb_image(RGB, title, save_path=None):
    plt.figure(figsize=(10, 10))
    plt.imshow(RGB, extent=[x.min(), x.max(), y.min(), y.max()], origin='lower', aspect='equal')
    plt.xlabel('x (µm)', fontweight='bold')
    plt.ylabel('y (µm)', fontweight='bold')
    plt.xlim(xlim)
    plt.ylim(ylim)
    plt.title(title, fontweight='bold')
    #plt.axis("off")
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

def point_like(
    PCE: float = 1.0,
    aperture_decenter_x: float = 0,
    aperture_decenter_y: float = 0,
    aperture_diameter: float = 40e-6,
):
    metalens_RL = 0
    metalens_LL = 0
    aperture = 0
    for metalens_L, metalens_R in zip(metalens_LCP, metalens_RCP):
        Metalens_element = optiprop.EqualPathPhase(Input_field)
        Metalens_element.calculate_phase(
            focal_length=metalens_focal_length,
                design_lambda=wavelength,
                lens_diameter=lens_diameter,
                lens_center=[metalens_L['cx'], metalens_L['cy']],
                aperture_type='circle',
                aperture_size=[lens_diameter],
                amplitude=metalens_L['amplitude']*PCE,
                phase_offset=0,
            )
        metalens_RL += Metalens_element.U0
        #noise
        Metalens_element = optiprop.EqualPathPhase(Input_field)
        Metalens_element.calculate_phase(
            focal_length=metalens_focal_length,
                design_lambda=-wavelength,
                lens_diameter=lens_diameter,
                lens_center=[metalens_R['cx'], metalens_R['cy']],
                aperture_type='circle',
                aperture_size=[lens_diameter],
                amplitude=metalens_R['amplitude']*PCE,
                phase_offset=0,
            )
        metalens_RL += Metalens_element.U0
        #co-polarized
        Metalens_element = optiprop.IncidentField(Input_field)
        Metalens_element.calculate_phase(
            incident_angle=[0, 0],
            n=1,
            design_lambda=wavelength,
            lens_center=[metalens_L['cx'], metalens_L['cy']],
            aperture_type='circle',
            aperture_size=[lens_diameter],
            amplitude=(1-abs(metalens_L['amplitude'])*PCE),
            phase_offset=0,
        )
        metalens_LL += Metalens_element.U0
        Metalens_element = optiprop.IncidentField(Input_field)
        Metalens_element.calculate_phase(
            incident_angle=[0, 0],
            n=1,
            design_lambda=wavelength,
            lens_center=[metalens_R['cx'], metalens_R['cy']],
            aperture_type='circle',
            aperture_size=[lens_diameter],
            amplitude=(1-abs(metalens_R['amplitude'])*PCE),
            phase_offset=0,
        )
        metalens_LL += Metalens_element.U0

        #aperture decenter
        Aperture_element = optiprop.IncidentField(Input_field)
        Aperture_element.calculate_phase(
            incident_angle=[0, 0],
            n=1,
            design_lambda=wavelength,
            lens_center=[metalens_L['cx']+aperture_decenter_x, metalens_L['cy']+aperture_decenter_y],
            aperture_type='circle',
            aperture_size=[aperture_diameter],
        )
        aperture += Aperture_element.U0
        Aperture_element.calculate_phase(
            incident_angle=[0, 0],
            n=1,
            design_lambda=wavelength,
            lens_center=[metalens_R['cx']+aperture_decenter_x, metalens_R['cy']+aperture_decenter_y],
            aperture_type='circle',
            aperture_size=[aperture_diameter],
        )
        aperture += Aperture_element.U0
    # # Register the field on the NearField helper (for convenient plotting)
    # Input_field.input_field(Metalens_element)

    # # Visualize the resulting field, amplitude and phase
    # Input_field.draw(
    #     metalens_RL,
    #     cmap='turbo',
    #     fontsize=12,
    #     xlim=None,
    #     ylim=None,
    #     selected_field='all',
    #     dark_style=True,
    #     show=True,
    #     title='RCP->LCP'
    # )

    # # Visualize the resulting field, amplitude and phase
    # Input_field.draw(
    #     metalens_LL,
    #     cmap='turbo',
    #     fontsize=12,
    #     xlim=None,
    #     ylim=None,
    #     selected_field='all',
    #     dark_style=True,
    #     show=True,
    #     title='LCP->LCP'
    # )

    # Create an Angular Spectrum Method (ASM) propagator
    # - propagation_wavelength: effective wavelength in the medium (m)
    # - propagation_distance: distance between near-field plane and observation plane (m)
    # - n: refractive index (1.0 for air)
    propagator = optiprop.ASMPropagation(
        propagation_wavelength=wavelength,
        propagation_distance=metalens_focal_length,
        n=1,
        device=device,
        dtype=dtype,
    )

    # Feed the combined near-field into the propagator
    propagator.set_input_field(
        u_in= metalens_RL + metalens_LL,
        pixel_size=NearField_pixel_size,
    )

    # Inspect the input field settings and visualization
    propagator.show_input_U(
        cmap='turbo',
        fontsize=14,
        xlim=None,
        ylim=None,
        selected_field='all',
        dark_style=True,
        show=False,
        save_path=ULCP_input_field_path
    )
    # Run ASM propagation to the focal plane and visualize complex field
    propagator.propagate()
    propagator.output_U = propagator.output_U * aperture
    propagator.show_output_U(
        cmap='turbo',
        fontsize=14,
        xlim=None,
        ylim=None,
        selected_field='all',
        dark_style=True,
        show=False,
        save_path=ULCP_output_field_path
    )

    propagator2 = optiprop.ASMPropagation(
        propagation_wavelength=wavelength,
        propagation_distance=metalens_focal_length*9,
        n=1,
        device=device,
        dtype=dtype,
    )

    propagator2.set_input_field(
        u_in=propagator.output_U,
        pixel_size=NearField_pixel_size,
    )
    propagator2.propagate()

    propagator2.show_output_U(
        cmap='turbo',
        fontsize=14,
        xlim=[-40, 40],
        ylim=[-40, 40],
        selected_field='all',
        dark_style=True,
        show=False,
        save_path=U_LCP_far_field_path
    )
    U_LCP = propagator2.output_U

    metalens_RL = 0
    metalens_LL = 0
    aperture = 0
    for metalens_R, metalens_L in zip(metalens_LCP, metalens_RCP):
        Metalens_element = optiprop.EqualPathPhase(Input_field)
        Metalens_element.calculate_phase(
            focal_length=metalens_focal_length,
                design_lambda=wavelength,
                lens_diameter=lens_diameter,
                lens_center=[metalens_L['cx'], metalens_L['cy']],
                aperture_type='circle',
                aperture_size=[lens_diameter],
                amplitude=metalens_L['amplitude']*PCE,
                phase_offset=0,
            )
        metalens_RL += Metalens_element.U0
        #noise
        Metalens_element = optiprop.EqualPathPhase(Input_field)
        Metalens_element.calculate_phase(
            focal_length=metalens_focal_length,
                design_lambda=-wavelength,
                lens_diameter=lens_diameter,
                lens_center=[metalens_R['cx'], metalens_R['cy']],
                aperture_type='circle',
                aperture_size=[lens_diameter],
                amplitude=metalens_R['amplitude']*PCE,
                phase_offset=0,
            )
        metalens_RL += Metalens_element.U0
        #co-polarized
        Metalens_element = optiprop.IncidentField(Input_field)
        Metalens_element.calculate_phase(
            incident_angle=[0, 0],
            n=1,
            design_lambda=wavelength,
            lens_center=[metalens_L['cx'], metalens_L['cy']],
            aperture_type='circle',
            aperture_size=[lens_diameter],
            amplitude=(1-abs(metalens_L['amplitude'])*PCE),
            phase_offset=0,
        )
        metalens_LL += Metalens_element.U0
        Metalens_element = optiprop.IncidentField(Input_field)
        Metalens_element.calculate_phase(
            incident_angle=[0, 0],
            n=1,
            design_lambda=wavelength,
            lens_center=[metalens_R['cx'], metalens_R['cy']],
            aperture_type='circle',
            aperture_size=[lens_diameter],
            amplitude=(1-abs(metalens_R['amplitude'])*PCE),
            phase_offset=0,
        )
        metalens_LL += Metalens_element.U0

        #aperture decenter
        Aperture_element = optiprop.IncidentField(Input_field)
        Aperture_element.calculate_phase(
            incident_angle=[0, 0],
            n=1,
            design_lambda=wavelength,
            lens_center=[metalens_L['cx']+aperture_decenter_x, metalens_L['cy']+aperture_decenter_y],
            aperture_type='circle',
            aperture_size=[aperture_diameter],
        )
        aperture += Aperture_element.U0
        Aperture_element.calculate_phase(
            incident_angle=[0, 0],
            n=1,
            design_lambda=wavelength,
            lens_center=[metalens_R['cx']+aperture_decenter_x, metalens_R['cy']+aperture_decenter_y],
            aperture_type='circle',
            aperture_size=[aperture_diameter],
        )
        aperture += Aperture_element.U0
    # # Register the field on the NearField helper (for convenient plotting)
    # Input_field.input_field(Metalens_element)

    # # Visualize the resulting field, amplitude and phase
    # Input_field.draw(
    #     metalens_RL,
    #     cmap='turbo',
    #     fontsize=12,
    #     xlim=None,
    #     ylim=None,
    #     selected_field='all',
    #     dark_style=True,
    #     show=True,
    #     title='RCP->LCP'
    # )

    # # Visualize the resulting field, amplitude and phase
    # Input_field.draw(
    #     metalens_LL,
    #     cmap='turbo',
    #     fontsize=12,
    #     xlim=None,
    #     ylim=None,
    #     selected_field='all',
    #     dark_style=True,
    #     show=True,
    #     title='LCP->LCP'
    # )

    # Create an Angular Spectrum Method (ASM) propagator
    # - propagation_wavelength: effective wavelength in the medium (m)
    # - propagation_distance: distance between near-field plane and observation plane (m)
    # - n: refractive index (1.0 for air)
    propagator = optiprop.ASMPropagation(
        propagation_wavelength=wavelength,
        propagation_distance=metalens_focal_length,
        n=1,
        device=device,
        dtype=dtype,
    )

    # Feed the combined near-field into the propagator
    propagator.set_input_field(
        u_in= metalens_RL + metalens_LL,
        pixel_size=NearField_pixel_size,
    )

    # Inspect the input field settings and visualization
    propagator.show_input_U(
        cmap='turbo',
        fontsize=14,
        xlim=None,
        ylim=None,
        selected_field='all',
        dark_style=True,
        show=False,
        save_path=U_RCP_input_field_path
    )
    # Run ASM propagation to the focal plane and visualize complex field
    propagator.propagate()
    propagator.output_U = propagator.output_U * aperture
    propagator.show_output_U(
        cmap='turbo',
        fontsize=14,
        xlim=None,
        ylim=None,
        selected_field='all',
        dark_style=True,
        show=False,
        save_path=U_RCP_output_field_path
    )

    propagator2 = optiprop.ASMPropagation(
        propagation_wavelength=wavelength,
        propagation_distance=metalens_focal_length*9,
        n=1,
        device=device,
        dtype=dtype,
    )

    propagator2.set_input_field(
        u_in=propagator.output_U,
        pixel_size=NearField_pixel_size,
    )
    propagator2.propagate()

    propagator2.show_output_U(
        cmap='turbo',
        fontsize=14,
        xlim=[-40, 40],
        ylim=[-40, 40],
        selected_field='all',
        dark_style=True,
        show=False,
        save_path=U_RCP_far_field_path
    )
    U_RCP = propagator2.output_U

    return U_LCP, U_RCP

if __name__ == '__main__':
    # Configuration
    # - device: 'cpu' or 'cuda' (if you have a GPU and CUDA-enabled PyTorch installed)
    # - dtype: float32 is memory-friendly; float64 increases precision
    device = 'cpu'
    dtype = torch.float32

    # Near-field grid definition
    # - NearField_pixel_size: sampling interval (m) on the near-field plane
    # - NearField_field_Lx/Ly: physical size (m) of the simulated field in X and Y
    NearField_pixel_size = 400e-9
    NearField_field_Lx, NearField_field_Ly = 1000e-6, 1000e-6


    # Create the near-field grid on which phase elements will be defined
    Input_field = optiprop.NearField(
        pixel_size=NearField_pixel_size,
        field_Lx=NearField_field_Lx,
        field_Ly=NearField_field_Ly,
        dtype=dtype,
        device=device,
    )

    # Pretty print grid details (size, pixel pitch, device, etc.)
    Input_field.rich_print()

    # Metalens array definition
    metalens_RCP = [
        {'cx': -200e-6, 'cy': 0e-6, 'amplitude': 1},
        {'cx': +200e-6, 'cy': 0e-6, 'amplitude': 1},
        {'cx': 0e-6, 'cy': -200e-6, 'amplitude': 1},
        {'cx': 0e-6, 'cy': 200e-6, 'amplitude': 1},
        {'cx': -400e-6, 'cy': 0e-6, 'amplitude': -1/2},
        {'cx': 400e-6, 'cy': 0e-6, 'amplitude': -1/2},
        {'cx': 0e-6, 'cy': -400e-6, 'amplitude': -1/2},
        {'cx': 0e-6, 'cy': 400e-6, 'amplitude': -1/2},
    ]
    metalens_LCP = [
        {'cx': -200e-6, 'cy': 100e-6, 'amplitude': 1j},
        {'cx': -200e-6, 'cy': -100e-6, 'amplitude': 1j},
        {'cx': 200e-6, 'cy': 100e-6, 'amplitude': -1j},
        {'cx': 200e-6, 'cy': -100e-6, 'amplitude': -1j},
        {'cx': -100e-6, 'cy': 200e-6, 'amplitude': 1},
        {'cx': 100e-6, 'cy': 200e-6, 'amplitude': 1},
        {'cx': -100e-6, 'cy': -200e-6, 'amplitude': -1},
        {'cx': 100e-6, 'cy': -200e-6, 'amplitude': -1},
    ]
    # Add a focusing metalens element to collimate/focus the VCSEL beam
    # - wavelength: design wavelength (m)
    # - lens_diameter: diameter of the metalens (m)
    # - metalens_focal_length: focal length of the metalens (m)
    wavelength = 1.31e-6
    lens_diameter = 100e-6
    metalens_focal_length = 575e-6 / 1.46

    PCE = 1.0
    for AP_DIA in [30e-6]:
        for DCT_DISTANCE in [0e-6, 10e-6, 20e-6, 30e-6]:
            for THETA in [0]:
                aperture_diameter = AP_DIA#50e-6
                aperture_decenter_distance = DCT_DISTANCE#20e-6
                aperture_decenter_angle = THETA#np.pi/2
                aperture_decenter_x = aperture_decenter_distance*np.cos(aperture_decenter_angle)
                aperture_decenter_y = aperture_decenter_distance*np.sin(aperture_decenter_angle)
                os.makedirs('result', exist_ok=True)
                output_folder = f'AP_{aperture_diameter*1e6}um_DCT_{aperture_decenter_distance*1e6}um_THETA_{round(aperture_decenter_angle*180/np.pi, 1)}deg'
                os.makedirs(os.path.join('result', output_folder), exist_ok=True)
                ULCP_input_field_path = os.path.join('result', output_folder, 'U_LCP_input_field.png')
                ULCP_output_field_path = os.path.join('result', output_folder, 'U_LCP_output_field.png')
                U_LCP_far_field_path = os.path.join('result', output_folder, 'U_LCP_far_field.png')
                U_RCP_input_field_path = os.path.join('result', output_folder, 'U_RCP_input_field.png')
                U_RCP_output_field_path = os.path.join('result', output_folder, 'U_RCP_output_field.png')
                U_RCP_far_field_path = os.path.join('result', output_folder, 'U_RCP_far_field.png')
                U_LCP, U_RCP = point_like(
                    PCE=PCE, 
                    aperture_decenter_x=aperture_decenter_x, 
                    aperture_decenter_y=aperture_decenter_y, 
                    aperture_diameter=aperture_diameter
                    )
                S0 = torch.abs(U_LCP) + torch.abs(U_RCP)
                LCP = U_LCP/S0
                RCP = U_RCP/S0
                Sx = 2*torch.real(torch.conj(LCP)*RCP)
                Sy = -2*torch.imag(torch.conj(LCP)*RCP)
                Sz = torch.abs(RCP)**2 - torch.abs(LCP)**2
                RGB, (H,S,V) = polarization_color_map(Sx, Sy, Sz, sat_mode="transverse", val_mode="signed")
                x = Input_field.X*1e6
                y = Input_field.Y*1e6
                xlim = [-40, 40]
                ylim = [-40, 40]
                # Show the composite pseudo-spin color map
                show_rgb_image(RGB, f'AP_{aperture_diameter*1e6}um_DCT_{aperture_decenter_distance*1e6}um_THETA_{round(aperture_decenter_angle*180/np.pi, 1)}deg', save_path=os.path.join('result', output_folder, 'Pseudo-spin color map.png'))