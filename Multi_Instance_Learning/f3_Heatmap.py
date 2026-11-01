from pathlib import Path
import time
print('[Startup] Loading heatmap script dependencies.', flush=True)
import numpy as np
import pandas as pd
import cv2
from PIL import Image
import yaml
import h5py
import torch
from tqdm import tqdm
from model_AttentionMIL import AttentionNet


# ======================== Edit paths here ========================
yaml_path = Path(r"D:\Python_project\Macrosection\Lymph_Nodes\lymph_node_ABMIL_external_validation.yaml")
wsi_info_path = Path(r"D:\Python_project\Macrosection\Lymph_Nodes\MPP_Storage\Slide_Info_Aligned.xlsx")
thumbnail_dir = Path(r"D:\Digital_Pathology_Resources\Masks\Macrosection_Project\Lymph_Masks")
# Existing <slide_name>_original.png files in the original extraction thumb_dir; do not use the WSI directory or _segmented.png files.
output_dir = Path(r"D:\Python_project\Macrosection\Lymph_Nodes\Attention_Image_Aligned")
model_root = Path(r"D:\Python_project\Macrosection\Lymph_Nodes\Modeling_Results")
wanted_file = Path(r"D:\Python_project\Macrosection\Lymph_Nodes\Files\Review1026\all_for_heatmap.xlsx")
# wanted_file = None  # Replace the line above with this one to plot all feature files.

model_candidate = 2       # Same as original f8: third result-table row, indexed from zero; confirm the intended model
attention_class = 1       # Same as original f8: display class 1, as defined by the training labels
blend_alpha = 0.5         # Preserve the original f8 opacity
sigma = 3                # Convert the original f8 smoothing scale from a level-0/16 canvas to the current canvas
max_thumbnail_side = 2400 # Resize existing thumbnails proportionally in memory only; use None to retain their original size
save_coverage = False     # Save only the final fusion image by default; use True to inspect patch coverage
verbose = True          # Print processing stages for each slide without creating a log file
# ================================================================


def log(message):
    print(f'[{time.strftime("%H:%M:%S")}] {message}', flush=True)


def load_existing_thumbnail(slide_name):
    """Open the full-view original thumbnail saved during extraction, without a preliminary path check."""
    path = thumbnail_dir / (slide_name + '_original.png')
    if verbose:
        log(f'[{slide_name}] Reading existing thumbnail: {path}')
    with Image.open(path) as source:
        background = source.convert('RGB')
    if max_thumbnail_side is not None:
        background.thumbnail((max_thumbnail_side, max_thumbnail_side), Image.Resampling.LANCZOS)
    return background


def read_geometry(row):
    patch, overlap_px = float(row['patch_size']), float(row['overlap'])
    step, footprint = float(row['step_level0']), float(row['patch_size_level0'])
    origin_ds, actual_ds = float(row['tile_downsample']), float(row['actual_downsample'])
    values = [patch, overlap_px, step, footprint, origin_ds, actual_ds]
    if not np.isfinite(values).all() or min(patch, step, footprint, origin_ds, actual_ds) <= 0 or overlap_px < 0 or overlap_px >= patch:
        raise ValueError(f'Invalid geometry fields in Excel: {values}')
    expected_step = (patch-overlap_px)*origin_ds
    expected_footprint = patch*actual_ds
    if not np.isclose(step, expected_step, rtol=1e-10, atol=1e-8) or not np.isclose(footprint, expected_footprint, rtol=1e-10, atol=1e-8):
        raise ValueError(
            f'Inconsistent geometry fields in Excel: step_level0={step:.15g}, calculated={expected_step:.15g}; '
            f'patch_size_level0={footprint:.15g}, calculated={expected_footprint:.15g}; '
            f'patch_size={patch:.15g}, overlap={overlap_px:.15g}, tile_downsample={origin_ds:.15g}, '
            f'actual_downsample={actual_ds:.15g}; utils={row.get("utils_module", "not recorded in the legacy table")}. '
            'Check or provide this Excel row; do not simply disable validation.')
    return patch, overlap_px, step, footprint


def load_trained_model(args, device):
    customs = args['customs']
    log(f'Reading training manifest: {customs["manifest_file_path"]}')
    manifest = pd.read_excel(customs['manifest_file_path'])
    labels = sorted(manifest['label'].unique())
    if labels != list(range(len(labels))):
        raise ValueError('Training labels must be consecutive integers starting at 0.')
    log(f'Reading model results table: {customs["training_result_file"]}')
    result = pd.read_excel(customs['training_result_file']).iloc[model_candidate]
    epoch = float(result['Epoch'])
    if not np.isfinite(epoch) or epoch != int(epoch):
        raise ValueError('Epoch in the model results table must be an integer.')
    checkpoint = model_root / str(customs['train_run_id']) / str(result['Task_Name']) / f'{int(epoch)}_checkpoint.pt'
    log(f'Loading MIL weights: {checkpoint}')
    model = AttentionNet(input_feature_size=int(customs['feature_size']), n_classes=len(labels),
                         model_size=customs['model_size'], p_dropout_fc=customs['dropout_fc'],
                         p_dropout_atn=customs['dropout_atn'])
    model.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=True), strict=True)
    model.to(device).eval()
    log(f'MIL model ready; device={device}')
    return model


def get_featurecoord(path):
    with h5py.File(path, 'r') as source:
        features, coords = source['features'][:], source['coords'][:]
    if features.ndim != 2 or coords.shape != (len(features), 2) or len(features) == 0:
        raise ValueError('features must have shape N x D and coords must have shape N x 2, with N greater than 0.')
    if not np.isfinite(features).all() or not np.isfinite(coords).all():
        raise ValueError('Features or coordinates contain non-finite values.')
    if (coords < 0).any() or not np.equal(coords, np.rint(coords)).all():
        raise ValueError('This workflow requires the nonnegative integer grid indices saved by the original code.')
    return torch.from_numpy(np.asarray(features, dtype=np.float32)), coords.astype(np.int64)


def find_slide_row(info, relative_key, slide_name):
    matches = info[info['slide_key'].str.casefold() == relative_key.casefold()]
    if len(matches) == 0:
        matches = info[info['file_name'].str.casefold() == slide_name.casefold()]
    if len(matches) != 1:
        raise ValueError(f'Found {len(matches)} matching Excel rows. Preserve the corresponding subdirectories for slides with duplicate names; the match must be unambiguous.')
    row = matches.iloc[0]
    if row['status'] != 'OK':
        raise ValueError(f'Step 1 failed: {row["error"]}')
    return row


def generate_heatmap(width, height, coords, values, step, patch_size, thumbnail_size):
    """Draw on a thumbnail-sized canvas, transforming grid origins and patch footprints separately."""
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if len(values) != len(coords) or not np.isfinite(values).all():
        raise ValueError('Scores must match coords row for row and contain only finite values.')
    wh = np.array([width, height], dtype=np.float64)
    canvas = np.array(thumbnail_size, dtype=np.int64)
    if not np.isfinite([width, height, step, patch_size]).all() or min(width, height, step, patch_size) <= 0:
        raise ValueError('Invalid image dimensions, origin spacing, or patch footprint in Excel.')
    xy0 = coords * step
    if np.any(xy0 < 0) or np.any(xy0 >= wh):
        raise ValueError('Reconstructed patch origins fall outside the full slide. Check the historical extraction settings in step 1.')
    left = np.floor(xy0 * canvas / wh).astype(np.int64)
    right = np.ceil((xy0 + patch_size) * canvas / wh).astype(np.int64)
    left, right = np.clip(left, 0, canvas), np.clip(right, 0, canvas)
    canvas_width, canvas_height = map(int, canvas)
    totals = np.zeros((canvas_height + 1, canvas_width + 1), dtype=np.float64)
    counts = np.zeros_like(totals)
    # Accumulate rectangle differences efficiently; average overlaps without reordering patches.
    for target, weights in ((totals, values), (counts, np.ones(len(values)))):
        np.add.at(target, (left[:, 1], left[:, 0]), weights)
        np.add.at(target, (left[:, 1], right[:, 0]), -weights)
        np.add.at(target, (right[:, 1], left[:, 0]), -weights)
        np.add.at(target, (right[:, 1], right[:, 0]), weights)
    totals = totals.cumsum(0).cumsum(1)[:-1, :-1]
    counts = counts.cumsum(0).cumsum(1)[:-1, :-1]
    covered = counts > 0.5
    field = np.divide(totals, counts, out=np.zeros_like(totals), where=covered)
    return field, covered


def blend_heatmap(background, field, covered, wsi_width=None, wsi_height=None):
    """Reuse the original f8 OpenCV JET display workflow with the corrected coordinate mapping."""
    heatmap = np.asarray(field, dtype=np.float32)
    if sigma > 0:
        # Original f8 used sigma=3 on a level-0/16 canvas; preserve that smoothing scale in slide coordinates.
        sx = sigma if wsi_width is None else sigma * 16 * background.width / wsi_width
        sy = sigma if wsi_height is None else sigma * 16 * background.height / wsi_height
        heatmap = cv2.GaussianBlur(heatmap, (0, 0), sigmaX=sx, sigmaY=sy,
                                    borderType=cv2.BORDER_REFLECT)
    if np.ptp(heatmap) == 0:
        heatmap_norm = np.full(heatmap.shape, 128, dtype=np.uint8)
    else:
        heatmap_norm = cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    # Preserve the exact color bytes passed to PIL by original f8, including the 255-x mapping; do not swap channels.
    # Retain the original inversion and channel handling together to reproduce the displayed palette.
    colors = cv2.applyColorMap(255 - heatmap_norm, cv2.COLORMAP_JET)
    alpha = blend_alpha * covered[..., None]
    result = (1-alpha) * np.asarray(background, dtype=float) + alpha * colors
    return Image.fromarray(np.clip(result, 0, 255).astype(np.uint8))


def main():
    log('Existing thumbnail mode: original slides are not opened, and WSI paths are not read or checked.')
    log(f'Reading YAML: {yaml_path}')
    if not 0 <= blend_alpha <= 1 or not np.isfinite(sigma) or sigma < 0 or (max_thumbnail_side is not None and max_thumbnail_side <= 0):
        raise ValueError('blend_alpha must be between 0 and 1, sigma must be nonnegative, and max_thumbnail_side must be positive or None.')
    with yaml_path.open('r', encoding='utf-8-sig') as stream:
        args = yaml.safe_load(stream)
    feature_dir = Path(args['customs']['features_save_path'])  # Reuse the feature directory from the original YAML
    log(f'Reading slide information table: {wsi_info_path}')
    info = pd.read_excel(wsi_info_path, dtype=object, keep_default_na=False)
    required = {'slide_key', 'file_name', 'width', 'height', 'step_level0', 'patch_size_level0',
                'status', 'error',
                'patch_size', 'overlap', 'tile_downsample', 'actual_downsample', 'read_width', 'read_height'}
    if not required.issubset(info.columns):
        raise ValueError('Use Slide_Info_Aligned.xlsx generated by the updated step 1 script. The current Excel file is missing required fields.')
    log(f'Recursively scanning feature directory: {feature_dir}; scanning a server directory may take time.')
    features = sorted(p for p in feature_dir.rglob('*') if p.suffix.lower() in {'.h5', '.hdf5'} and p.is_file())
    log(f'Scan complete: found {len(features)} feature files.')
    keys = [p.relative_to(feature_dir).with_suffix('').as_posix().casefold() for p in features]
    if len(keys) != len(set(keys)):
        raise ValueError('Multiple feature file versions share the same relative stem. Keep only the intended version before plotting.')
    if wanted_file is not None:
        log(f'Reading plot selection list: {wanted_file}')
        wanted = pd.read_excel(wanted_file, dtype=str, keep_default_na=False).iloc[:, 0].str.strip().tolist()
        selected = []
        for name in wanted:
            if not name:
                continue
            name = name.replace('\\', '/')
            if Path(name).suffix.lower() in {'.h5', '.hdf5'}:
                name = Path(name).with_suffix('').as_posix()
            matches = [p for p in features if p.relative_to(feature_dir).with_suffix('').as_posix().casefold() == name.casefold()]
            if not matches:
                matches = [p for p in features if p.stem.casefold() == name.casefold()]
            if len(matches) == 0:
                print(f'No matching feature file for {name} in the selection list; skipping.')
                continue
            if len(matches) != 1:
                raise ValueError(f'Selection list entry {name} matches {len(matches)} feature files. Check the paths or use the full relative name.')
            if matches[0] not in selected:
                selected.append(matches[0])
        features = selected
    if not features:
        raise ValueError('No feature files selected for plotting. Check the YAML and wanted_file settings.')
    # Original extraction saved thumbnails in a flat directory by slide stem, so duplicate stems cannot be matched reliably.
    names = [p.stem.casefold() for p in features]
    if len(names) != len(set(names)):
        raise ValueError('Selected feature files have duplicate slide stems. Original thumbnails are stored in a flat directory by name; process each thumbnail directory as a separate batch to avoid mismatches.')
    log(f'Selected {len(features)} slides for plotting; existing thumbnail directory: {thumbnail_dir}')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = load_trained_model(args, device)
    output_dir.mkdir(parents=True, exist_ok=True)
    error_count = 0
    for number, feature_path in enumerate(tqdm(features, desc='Generating heatmaps'), start=1):
        key = feature_path.relative_to(feature_dir).with_suffix('')
        started = time.perf_counter()
        try:
            if verbose:
                log(f'[{number}/{len(features)} {key.as_posix()}] Validating Excel geometry fields')
            row = find_slide_row(info, key.as_posix(), feature_path.stem)
            patch, overlap_px, step, patch0 = read_geometry(row)
            width, height = float(row['width']), float(row['height'])
            background = load_existing_thumbnail(feature_path.stem)
            if verbose:
                log(f'[{key.as_posix()}] Reading H5: {feature_path}')
            tensor, coords = get_featurecoord(feature_path)
            if verbose:
                log(f'[{key.as_posix()}] H5 loaded: N={len(coords)}; starting MIL inference')
            if np.any(coords*(patch-overlap_px)+patch > np.array([float(row['read_width']), float(row['read_height'])])):
                raise ValueError('coords exceed the read-level bounds. The features may belong to a different slide or extraction configuration.')
            with torch.inference_mode():
                _, _, _, raw, _ = model(tensor.to(device))
                raw = raw.cpu().numpy()
            if raw.ndim != 2 or raw.shape[1] != len(coords) or not 0 <= attention_class < raw.shape[0]:
                raise ValueError('Attention class or patch count mismatch.')
            scores = np.asarray(raw[attention_class], dtype=np.float64)
            if not np.isfinite(scores).all():
                raise ValueError('Attention scores contain non-finite values.')
            span = np.ptp(scores)
            normalized = np.full_like(scores, 0.5) if span == 0 else (scores-scores.min())/span
            field, covered = generate_heatmap(width, height, coords, normalized,
                                               step, patch0, background.size)
            destination = output_dir / key.parent
            destination.mkdir(parents=True, exist_ok=True)
            if verbose:
                log(f'[{key.as_posix()}] Attention inference complete; rendering and saving PNG')
            blend_heatmap(background, field, covered, width, height).save(
                destination / (key.name+'_Heatmap_Fusion.png'))
            if save_coverage:
                coverage = np.asarray(background).copy()
                coverage[covered] = (0.7*coverage[covered]+0.3*np.array([0,220,0])).astype(np.uint8)
                Image.fromarray(coverage).save(destination / (key.name+'_Patch_Coverage.png'))
            if verbose:
                log(f'[{key.as_posix()}] Success; elapsed {time.perf_counter()-started:.1f} seconds')
        except Exception as error:
            error_count += 1
            log(f'Failed: {key.as_posix()}; {error}')
    ok_count = len(features)-error_count
    log(f'Complete: {ok_count} slides succeeded, {error_count} slides failed; output: {output_dir}')
    if ok_count == 0:
        log('No heatmaps were generated. Review the errors above and check the Excel file, H5 files, existing thumbnails, or MIL configuration/weights.')
    return {'ok': ok_count, 'failed': error_count}


if __name__ == '__main__':
    summary = main()
    raise SystemExit(1 if summary['ok'] == 0 else 0)