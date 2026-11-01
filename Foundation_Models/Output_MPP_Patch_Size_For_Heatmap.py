from pathlib import Path
from importlib import import_module
print('[Startup] Loading dependencies for step 1; Excel output only.', flush=True)
import numpy as np
import pandas as pd
from openslide import OpenSlide
from tqdm import tqdm

# ==================== Edit paths and historical extraction settings here ====================
data_dir = Path(r"D:\New_Nodes")                         # Original WSI directory
excel_path = Path(r"D:\Python_project\Macrosection\Lymph_Nodes\MPP_Storage\Slide_Info_Aligned.xlsx")
utils_module = 'utils.wsi_utils_forCamelyon16'  # Must match the module used to generate this batch of H5 files
# If this batch used the strict module, set utils_module = 'utils.wsi_utils_strict'

raw_patch_size = 256   # Initial patch_size passed to the original extraction code
target_mpp = 0.23      # Match the original setting; use None if it was originally None
tile_mag = 10         # tile_mag used during the original extraction
overlap = 0           # Original overlap, in pixels at the read level
# ====================================================================

def_slide_info = import_module(utils_module).def_slide_info


def get_slide_record(wsi_path):
    """Reproduce legacy sizing and rounding; record origin spacing and patch footprint separately."""
    _, tile_info, slide_info = def_slide_info(wsi_path, 1.25, tile_mag)
    legacy_downsample, tile_level, tile_factor = tile_info
    slide_mag, slide_mpp = slide_info
    if slide_mpp is None or not np.isfinite(slide_mpp) or slide_mpp <= 0:
        raise ValueError('Missing or invalid MPP.')
    requested_mpp = slide_mpp if target_mpp is None else target_mpp
    if raw_patch_size <= 0 or requested_mpp <= 0 or tile_factor <= 0 or overlap < 0:
        raise ValueError('Invalid extraction settings.')
    # Preserve both original branches, including the 0.25 constant and int rounding, to restore legacy feature positions.
    if tile_factor == 1.0:
        patch_size = int(raw_patch_size * (requested_mpp / tile_factor) / slide_mpp)
    else:
        patch_size = int(raw_patch_size * (requested_mpp / tile_factor) / 0.25)
    if patch_size <= overlap:
        raise ValueError('patch_size must be greater than overlap.')
    origin_downsample = int(legacy_downsample + 0.5)
    relative = wsi_path.relative_to(data_dir)
    with OpenSlide(str(wsi_path)) as slide:
        actual_downsample = float(slide.level_downsamples[tile_level])
        return {
            'file_name': wsi_path.stem,
            'slide_key': relative.with_suffix('').as_posix(),
            'width': slide.dimensions[0],             # Full slide width at level 0
            'height': slide.dimensions[1],            # Full slide height at level 0
            'mpp': slide_mpp,
            'patch_size': patch_size,                # Patch side length at the read level
            'tile_level': tile_level,
            'tile_downsample': origin_downsample,    # Integer factor used for origins by the legacy code
            'actual_downsample': actual_downsample,  # Actual floating-point factor used for patch footprints
            'overlap': overlap,
            'step_level0': (patch_size - overlap) * origin_downsample,
            'patch_size_level0': patch_size * actual_downsample,
            'read_width': slide.level_dimensions[tile_level][0],
            'read_height': slide.level_dimensions[tile_level][1],
            'raw_patch_size': raw_patch_size,
            'target_mpp': requested_mpp,
            'tile_mag': tile_mag,
            'utils_module': utils_module,
            'geometry_version': 'level0_grid_v2',
            'wsi_relative_path': relative.as_posix(),
            'wsi_path': str(wsi_path.resolve()),
            'status': 'OK',
            'error': '',
        }


def main():
    print(f'[Settings] utils={utils_module}; raw_patch_size={raw_patch_size}; target_mpp={target_mpp}; tile_mag={tile_mag}; overlap={overlap}', flush=True)
    print(f'[Scan] WSI directory: {data_dir}', flush=True)
    if not data_dir.is_dir():
        raise NotADirectoryError(f'Set data_dir at the top of the script to an accessible directory: {data_dir}')
    if excel_path.suffix.lower() != '.xlsx':
        raise ValueError('excel_path must end with .xlsx.')
    extensions = {'.svs', '.ndpi', '.tiff', '.tif', '.mrxs', '.scn', '.svslide', '.vms', '.vmu', '.bif'}
    slides = sorted(p for p in data_dir.rglob('*') if p.is_file() and p.suffix.lower() in extensions)
    if not slides:
        raise ValueError('No slides found under data_dir.')
    print(f'[Scan complete] Found {len(slides)} slides.', flush=True)
    # Require unique relative slide names so step 2 cannot match different file versions.
    keys = [p.relative_to(data_dir).with_suffix('').as_posix().casefold() for p in slides]
    if len(keys) != len(set(keys)):
        raise ValueError('Slides with the same stem and different extensions exist in the same directory. Keep only the intended version before proceeding.')
    records = []
    for number, wsi_path in enumerate(tqdm(slides, desc='Reading slide metadata'), start=1):
        try:
            print(f'[Metadata {number}/{len(slides)}] {wsi_path.name}', flush=True)
            records.append(get_slide_record(wsi_path))
        except Exception as error:
            records.append({'file_name': wsi_path.stem,
                            'slide_key': wsi_path.relative_to(data_dir).with_suffix('').as_posix(),
                            'wsi_relative_path': wsi_path.relative_to(data_dir).as_posix(),
                            'utils_module': utils_module,
                            'wsi_path': str(wsi_path.resolve()), 'status': 'ERROR',
                            'error': f'{type(error).__name__}: {error}'})
            print(f'\nFailed: {wsi_path.name}; {error}')
    frame = pd.DataFrame(records)
    # Keep the required columns and error details even if every slide fails.
    for name in ('width', 'height', 'step_level0', 'patch_size_level0', 'wsi_path'):
        if name not in frame:
            frame[name] = ''
    excel_path.parent.mkdir(parents=True, exist_ok=True)
    print(f'[Writing Excel] {excel_path}', flush=True)
    frame.to_excel(excel_path, index=False)
    print(f'\nComplete: {(frame["status"] == "OK").sum()} slides succeeded, {(frame["status"] != "OK").sum()} slides failed.')
    print('Excel:', excel_path)
    return {'ok': int((frame['status'] == 'OK').sum()), 'failed': int((frame['status'] != 'OK').sum())}


if __name__ == '__main__':
    summary = main()
    raise SystemExit(1 if summary['ok'] == 0 else 0)