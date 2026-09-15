from uuid import uuid4

import pytest

from optiprop.project import (
    AssetIntegrityError,
    AssetReference,
    AssetStatus,
    UnsafeAssetPathError,
    require_valid_assets,
    validate_assets,
)


def test_relative_asset_digest_missing_and_changed_states(tmp_path):
    assets = tmp_path / "assets"
    assets.mkdir()
    field = assets / "source.npz"
    field.write_bytes(b"canonical field")
    reference = AssetReference.from_file(field, project_directory=tmp_path)

    check = validate_assets((reference,), tmp_path)[0]
    assert check.status is AssetStatus.OK
    assert check.resolved_path == field.resolve()

    field.write_bytes(b"changed")
    assert validate_assets((reference,), tmp_path)[0].status is AssetStatus.CHANGED
    with pytest.raises(AssetIntegrityError, match="changed"):
        require_valid_assets((reference,), tmp_path)

    field.unlink()
    assert validate_assets((reference,), tmp_path)[0].status is AssetStatus.MISSING


@pytest.mark.parametrize(
    "path", ["../secret.npz", "/root/file.npz", "C:/secret.npz", "assets/../x.npz"]
)
def test_asset_path_traversal_and_absolute_paths_are_rejected(path):
    with pytest.raises(UnsafeAssetPathError):
        AssetReference(path, "0" * 64, 0)


def test_asset_must_be_inside_project_directory(tmp_path):
    outside = tmp_path.parent / f"outside-{uuid4()}.bin"
    outside.write_bytes(b"x")
    try:
        with pytest.raises(UnsafeAssetPathError):
            AssetReference.from_file(outside, project_directory=tmp_path)
    finally:
        outside.unlink()


def test_asset_ids_use_independent_default_factories():
    first = AssetReference("a.bin", "0" * 64, 0)
    second = AssetReference("b.bin", "0" * 64, 0)
    assert first.id != second.id
