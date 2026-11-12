"""Calibrated analytic synthetic scenes for dense LSM validation.

Each image is independently rendered from camera rays, with 4x4 pixel-area
supersampling of a continuous world-space texture. No reference image is warped
to create a target. GT correspondence and visibility are evaluation-only fields.
Perturbed GT initializations verify the optimizer; they are never named RoMa A0.
"""
from pathlib import Path
import argparse
import hashlib
import json

import cv2
import numpy as np


SIZE = (192, 144)
BACKGROUND_Z = 4.0
FOREGROUND_Z = 3.0
FOREGROUND_RECT = (-0.12, 0.30, -0.50, 0.18)
WEAK_RECT = (-1.20, -0.25, -0.55, 0.55)


def _rotation(x=0., y=0., z=0.):
    cx, sx = np.cos(x), np.sin(x)
    cy, sy = np.cos(y), np.sin(y)
    cz, sz = np.cos(z), np.sin(z)
    rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return rz @ ry @ rx


def make_camera(camera_id, center, rotation=None, focal=(180., 176.), size=SIZE):
    width, height = size
    r = np.eye(3) if rotation is None else np.asarray(rotation, float)
    center = np.asarray(center, float)
    k = np.array([[focal[0], 0, (width-1)/2],
                  [0, focal[1], (height-1)/2], [0, 0, 1]], float)
    return dict(camera_id=camera_id, size=[width, height], K=k.tolist(),
                R=r.tolist(), t=(-r @ center).tolist(), center_world=center.tolist(),
                distortion_model='none', D=[0., 0., 0., 0., 0.],
                coordinate_system='ideal pinhole working-image pixels, x/y, integer pixel centers')


def pixel_grid(size=SIZE):
    width, height = size
    y, x = np.mgrid[:height, :width]
    return np.stack([x, y], axis=-1).astype(np.float64)


def rays(camera, uv):
    """Return world center and unit world rays for arbitrary [...,2] pixels."""
    uv = np.asarray(uv, float)
    k, r, t = (np.asarray(camera[name], float) for name in ['K', 'R', 't'])
    homogeneous = np.concatenate([uv, np.ones((*uv.shape[:-1], 1))], axis=-1)
    direction = homogeneous @ np.linalg.inv(k).T @ r
    direction /= np.linalg.norm(direction, axis=-1, keepdims=True)
    return -r.T @ t, direction


def project(camera, xyz):
    k, r, t = (np.asarray(camera[name], float) for name in ['K', 'R', 't'])
    point = np.asarray(xyz) @ r.T + t
    homogeneous = point @ k.T
    return homogeneous[..., :2] / homogeneous[..., 2, None], point[..., 2]


def _in_rect(xyz, rectangle, inset=0.):
    xmin, xmax, ymin, ymax = rectangle
    return ((xyz[..., 0] >= xmin+inset) & (xyz[..., 0] <= xmax-inset)
            & (xyz[..., 1] >= ymin+inset) & (xyz[..., 1] <= ymax-inset))


def intersect_scene(camera, uv, occluder=False):
    """Nearest intersection with z=4 plane and optional finite z=3 rectangle."""
    center, direction = rays(camera, uv)
    d_background = (BACKGROUND_Z-center[2]) / direction[..., 2]
    xyz = center + d_background[..., None] * direction
    surface = np.zeros(uv.shape[:-1], dtype=np.int8)
    depth = d_background.copy()
    if occluder:
        d_foreground = (FOREGROUND_Z-center[2]) / direction[..., 2]
        foreground_xyz = center + d_foreground[..., None] * direction
        hit = (_in_rect(foreground_xyz, FOREGROUND_RECT)
               & (d_foreground > 0) & (d_foreground < d_background))
        xyz = np.where(hit[..., None], foreground_xyz, xyz)
        depth = np.where(hit, d_foreground, depth)
        surface[hit] = 1
    valid = (depth > 0) & np.isfinite(xyz).all(axis=-1)
    return xyz, depth, surface, valid


def _texture_parameters(seed):
    rng = np.random.default_rng(seed)
    angle = rng.uniform(0, 2*np.pi, 18)
    frequency = np.r_[rng.uniform(2.5, 9.5, 14), rng.uniform(11, 16, 4)]
    vectors = frequency[:, None] * np.stack([np.cos(angle), np.sin(angle)], axis=-1)
    amplitude = np.r_[rng.uniform(.019, .026, 14), np.full(4, .008)]
    phase = rng.uniform(0, 2*np.pi, 18)
    return vectors, amplitude, phase


def texture(xyz, surface, seed, weak_texture=False):
    """Bounded, continuous texture sampled analytically in world XY."""
    vectors, amplitude, phase = _texture_parameters(seed)
    xy = xyz[..., :2]
    # A separate analytic texture on the foreground is spatially coherent in all views.
    offset = np.where(surface[..., None] == 1, np.array([.723, -.419]), 0.)
    waves = 2*np.pi * np.einsum('...d,kd->...k', xy+offset, vectors) + phase
    value = .5 + np.sum(amplitude*np.sin(waves), axis=-1)
    if weak_texture:
        weak = (surface == 0) & _in_rect(xyz, WEAK_RECT)
        smooth = .5 + .0015*np.sin(1.7*xyz[..., 0]+.8*xyz[..., 1])
        value = np.where(weak, smooth, value)
    return value


def render(camera, scene, gain=1., bias=0., noise_sigma=0., seed=0, samples_per_axis=4):
    """Render floating grayscale with quadrature; per-view noise follows rendering."""
    grid = pixel_grid(camera['size'])
    image = np.zeros(grid.shape[:-1], dtype=np.float64)
    offsets = (np.arange(samples_per_axis)+.5)/samples_per_axis-.5
    for y in offsets:
        for x in offsets:
            xyz, _, surface, valid = intersect_scene(camera, grid+[x, y], scene['occluder'])
            assert valid.all()
            image += texture(xyz, surface, scene['texture_seed'], scene['weak_texture'])
    image /= samples_per_axis**2
    image = gain*image+bias
    if noise_sigma:
        image += np.random.default_rng(seed).normal(0, noise_sigma, image.shape)
    clipping = float(np.mean((image < 0) | (image > 1)))
    return np.clip(image, 0, 1).astype(np.float32), clipping


def ground_truth(reference, target, scene):
    """Dense GT center correspondence, ray distance and target visibility."""
    uv = pixel_grid(reference['size'])
    xyz, distance, surface, source_valid = intersect_scene(reference, uv, scene['occluder'])
    target_uv, target_z = project(target, xyz)
    width, height = target['size']
    inside = ((target_uv[..., 0] >= 0) & (target_uv[..., 0] <= width-1)
              & (target_uv[..., 1] >= 0) & (target_uv[..., 1] <= height-1)
              & (target_z > 0) & np.isfinite(target_uv).all(axis=-1))
    # Intersect target rays independently; the queried world point must be the first hit.
    target_xyz, _, _, hit_valid = intersect_scene(target, target_uv, scene['occluder'])
    visible = np.linalg.norm(target_xyz-xyz, axis=-1) < 1e-7
    valid = source_valid & inside & hit_valid & visible
    occluded = source_valid & inside & hit_valid & ~visible
    return dict(uv=target_uv.astype(np.float32), valid=valid, occluded=occluded,
                inside=source_valid & inside, depth=distance.astype(np.float32),
                xyz=xyz.astype(np.float32), surface=surface)


def perturbed_initialization(uv_gt, magnitude, seed=0):
    """Known smooth local-center errors for optimizer unit experiments, never RoMa."""
    height, width = uv_gt.shape[:2]
    y, x = np.mgrid[:height, :width]
    phase = np.random.default_rng(seed).uniform(-np.pi, np.pi)
    angle = phase + .35*np.sin(x/29.)*np.cos(y/31.)
    displacement = magnitude*np.stack([np.cos(angle), np.sin(angle)], axis=-1)
    return (uv_gt.astype(float)+displacement).astype(np.float32)


def case_definition(name, seed=20261111):
    scene = dict(texture_seed=seed, background_z=BACKGROUND_Z,
                 foreground_z=FOREGROUND_Z, foreground_rectangle=list(FOREGROUND_RECT),
                 weak_rectangle=list(WEAK_RECT), occluder=False, weak_texture=False)
    reference = make_camera(0, [0, 0, 0])
    if name == 'translation':
        cameras = [reference, make_camera(1, [.237, -.063, 0]),
                   make_camera(2, [-.36, .045, 0])]
        gain, bias, sigma = [1., 1., 1.], [0., 0., 0.], [0., 0., 0.]
        description = 'Exact noninteger 2D translation induced by calibrated parallel cameras and a planar surface.'
    elif name == 'affine_brightness':
        cameras = [reference,
                   make_camera(1, [.25, -.04, 0], _rotation(z=.065), (188., 182.)),
                   make_camera(2, [-.34, .06, 0], _rotation(z=-.045), (172., 181.))]
        gain, bias, sigma = [1., 1.10, .91], [0., .025, -.015], [0., .002, .004]
        description = 'Exact plane-induced affine rotation and anisotropic scale, with per-target gain/bias and weak noise.'
    elif name == 'occlusion_weak_noise':
        scene.update(occluder=True, weak_texture=True)
        cameras = [reference,
                   make_camera(1, [.26, -.035, 0], _rotation(y=.018, z=.035), (182., 179.)),
                   make_camera(2, [-.48, .07, 0], _rotation(y=-.025, z=-.04), (177., 173.))]
        gain, bias, sigma = [1., 1.10, .90], [0., .02, -.015], [.002, .009, .018]
        description = 'Projective planar background plus a nearer finite static rectangle, true occlusion, weak texture and asymmetric sensor noise.'
    else:
        raise ValueError(name)
    return cameras, scene, gain, bias, sigma, description


def generate_case(name, out_dir, seed=20261111):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=False)
    cameras, scene, gains, biases, noise, description = case_definition(name, seed)
    images, clipping = [], []
    for i, camera in enumerate(cameras):
        image, clip_fraction = render(camera, scene, gains[i], biases[i], noise[i], seed+100+i)
        images.append(image); clipping.append(clip_fraction)
        filename = out_dir/f'cam{i:02d}.png'
        encoded = np.rint(image*255).astype(np.uint8)
        assert cv2.imwrite(str(filename), cv2.cvtColor(encoded, cv2.COLOR_GRAY2BGR))
        camera['image_path'] = str(filename.resolve())
    grid = pixel_grid()
    xyz, depth, surface, _ = intersect_scene(cameras[0], grid, scene['occluder'])
    weak = (surface == 0) & _in_rect(xyz, WEAK_RECT, inset=.08) if scene['weak_texture'] else np.zeros(SIZE[::-1], bool)
    regions = np.where(weak, 1, 0).astype(np.uint8)
    regions[surface == 1] = 2
    data = dict(images_float=np.stack(images), reference_uv=grid.astype(np.float32),
                depth_gt=depth.astype(np.float32), xyz_gt=xyz.astype(np.float32),
                region_labels=regions, geometry_applicable_mask=np.ones(SIZE[::-1], bool))
    visibility_counts = {}
    for a, b in [(0, 1), (1, 0), (0, 2), (2, 0)]:
        truth = ground_truth(cameras[a], cameras[b], scene)
        key = f'{a}{b}'
        data[f'uv_gt_{key}'] = truth['uv']
        data[f'valid_gt_{key}'] = truth['valid']
        data[f'occluded_gt_{key}'] = truth['occluded']
        data[f'inside_gt_{key}'] = truth['inside']
        visibility_counts[key] = dict(visible=int(truth['valid'].sum()),
                                      occluded=int(truth['occluded'].sum()),
                                      in_bounds=int(truth['inside'].sum()))
        for magnitude, tag in [(.5, '0p5'), (2., '2'), (6., '6')]:
            data[f'init_uv_{key}_e{tag}'] = perturbed_initialization(truth['uv'], magnitude, seed+10*a+b)
    np.savez_compressed(out_dir/'arrays.npz', **data)
    metadata = dict(case=name, description=description, mode='static', seed=seed,
        cameras=cameras, image_size=list(SIZE), reference_camera=0, target_cameras=[1, 2],
        images_float='arrays.npz: images_float [camera,y,x], floating grayscale in [0,1]',
        png_quantization='8-bit RGB replicas of grayscale; official RoMa may read these; LSM should record whether it uses float or PNG data',
        array_path=str((out_dir/'arrays.npz').resolve()), scene=scene,
        geometry_convention='X_camera=R@X_world+t; C_world=-R.T@t; depth_gt is positive distance along a UNIT reference world ray',
        units='synthetic metres', pixel_convention='x/y, integer pixel centers, ideal pinhole coordinates, no distortion',
        renderer='Independent analytic ray intersections + continuous world-space Fourier texture, 4x4 subpixel quadrature per image; no bilinear source-image warp',
        gain=gains, bias=biases, noise_sigma=noise, clipped_fraction=clipping,
        all_scene_geometry_static=True,
        region_labels={'0':'textured background','1':'weak-texture interior','2':'foreground rectangle'},
        visibility=visibility_counts,
        geometry_applicable_mask='All pixels are static; independent of GT visibility. Occlusion/valid_gt are evaluation-only, never optimizer acceptance or weight inputs.',
        initialization_keys='init_uv_AB_e0p5 / e2 / e6: controlled errors of 0.5/2/6 work pixels, optimizer verification ONLY; NOT actual RoMa output',
        initialization_magnitudes=[.5, 2., 6.],
        limitations=['Analytic textured planes are simpler than real surfaces and do not model BRDF variation, defocus or rolling shutter.',
                     'GT maps pixel-center surface points; pixel-area integration mixes surfaces at occlusion boundaries.',
                     'Repeated Fourier texture can cause local ambiguities; no claim of guaranteed matcher success.',
                     'Three-camera weighting is a synthetic verification; no real-data multiview claim follows.'])
    metadata['image_sha256'] = {Path(c['image_path']).name: hashlib.sha256(Path(c['image_path']).read_bytes()).hexdigest() for c in cameras}
    (out_dir/'metadata.json').write_text(json.dumps(metadata, indent=2)+'\n')
    return metadata


def generate_suite(out_dir, seed=20261111):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    names = ['translation', 'affine_brightness', 'occlusion_weak_noise']
    cases = [generate_case(name, out_dir/name, seed) for name in names]
    manifest = dict(seed=seed, cases=[str((out_dir/name/'metadata.json').resolve()) for name in names],
                    caution='Controlled initialization fields are synthetic optimizer verification, not RoMa predictions.')
    (out_dir/'suite.json').write_text(json.dumps(manifest, indent=2)+'\n')
    return cases


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=20261111)
    args = parser.parse_args()
    result = generate_suite(args.out, args.seed)
    print(json.dumps({c['case']: c['visibility'] for c in result}, indent=2))
