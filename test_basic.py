#!/usr/bin/env python3
"""
Basic functionality test script
Used to verify package installation and basic functionality
"""

def test_import():
    """Test import functionality"""
    try:
        import optiprop
        print("[OK] Successfully imported field_propagation")
        return True
    except ImportError as e:
        print(f"[FAIL] Import failed: {e}")
        return False

def test_basic_functionality():
    """Test basic functionality"""
    try:
        import optiprop
        import torch
        
        # Test creating NearField
        field = optiprop.NearField(
            pixel_size=1e-6,
            field_Lx=100e-6,
            field_Ly=100e-6,
            device='cpu'
        )
        print("[OK] Successfully created NearField object")
        
        # Test creating lens
        lens = optiprop.EqualPathPhase(field)
        U0 = lens.calculate_phase(
            focal_length=1000e-6,
            design_lambda=0.94e-6,
            lens_diameter=80e-6
        )
        print("[OK] Successfully created lens and calculated phase")
        
        # Test propagation
        prop = optiprop.FresnelPropagation(
            propagation_wavelength=0.94e-6,
            propagation_distance=1000e-6,
            device='cpu'
        )
        prop.set_input_field(U0, field.pixel_size)
        prop.propagate()
        print("[OK] Successfully executed Fresnel propagation")
        
        return True
        
    except Exception as e:
        print(f"[FAIL] Basic functionality test failed: {e}")
        return False

def test_metaatom_library():
    """Test meta-atom database lookup"""
    try:
        import os
        import torch
        import optiprop

        library_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'asia_1310.npy')
        if not os.path.exists(library_path):
            print("[SKIP] asia_1310.npy not found, skipping meta-atom test")
            return True

        library = optiprop.MetaAtomLibrary(library_path, device='cpu')
        print(f"[OK] Loaded meta-atom library ({library.n_atoms} atoms, "
              f"coverage {library.phase_coverage:.3f} rad)")

        target = torch.linspace(-torch.pi, torch.pi, 64).reshape(8, 8)
        index_map, phase_map, amplitude_map = library.lookup(target, alpha=0.3)
        assert index_map.shape == target.shape
        assert bool((amplitude_map > 0).all())
        error = torch.remainder(phase_map - target + torch.pi, 2 * torch.pi) - torch.pi
        assert float(error.abs().max()) <= library.worst_lookup_error + 0.35
        print("[OK] Nearest-phase lookup works and error is within bounds")
        return True

    except Exception as e:
        print(f"[FAIL] Meta-atom library test failed: {e}")
        return False

def test_info():
    """Test package information"""
    try:
        import optiprop
        print(f"[OK] Package version: {optiprop.__version__}")
        print(f"[OK] Package author: {optiprop.__author__}")
        optiprop.info()
        return True
    except Exception as e:
        print(f"[FAIL] Package info test failed: {e}")
        return False

if __name__ == "__main__":
    print("Field Propagation Package Test")
    print("=" * 40)
    
    # Run tests
    tests = [
        ("Import Test", test_import),
        ("Basic Functionality Test", test_basic_functionality),
        ("Meta-Atom Library Test", test_metaatom_library),
        ("Package Info Test", test_info)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n{test_name}:")
        if test_func():
            passed += 1
    
    print("\n" + "=" * 40)
    print(f"Test Results: {passed}/{total} passed")
    
    if passed == total:
        print("SUCCESS: All tests passed! Package installation successful!")
    else:
        print("ERROR: Some tests failed, please check installation")
