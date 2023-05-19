import numpy
import trimesh
import trimesh.sample
import trimesh.visual
import trimesh.proximity
import streamlit as st
import matplotlib.pyplot as plotlib


def get_bytes(x: str):
    import io, requests
    return io.BytesIO(requests.get(x).content)


def get_image(x: str):
    try:
        return plotlib.imread(get_bytes(x), 'auto')
    except Exception:
        raise ValueError("Invalid image", x)


def model_to_pc(mesh: trimesh.Trimesh, n_sample_points=10000):
    f32 = numpy.float32
    rad = numpy.sqrt(mesh.area / (3 * n_sample_points))
    for _ in range(24):
        pcd, face_idx = trimesh.sample.sample_surface_even(mesh, n_sample_points, rad)
        rad *= 0.85
        if len(pcd) == n_sample_points:
            break
    else:
        raise ValueError("Bad geometry, cannot finish sampling.", mesh.area)
    if isinstance(mesh.visual, trimesh.visual.ColorVisuals):
        rgba = mesh.visual.face_colors[face_idx]
    elif isinstance(mesh.visual, trimesh.visual.TextureVisuals):
        bc = trimesh.proximity.points_to_barycentric(mesh.triangles[face_idx], pcd)
        uv = numpy.einsum('ntc,nt->nc', mesh.visual.uv[mesh.faces[face_idx]], bc)
        rgba = trimesh.visual.uv_to_interpolated_color(uv, mesh.visual.material.image)
    if rgba.max() > 1:
        if rgba.max() > 255:
            rgba = rgba.astype(f32) / rgba.max()
        else:
            rgba = rgba.astype(f32) / 255.0
    return numpy.concatenate([numpy.array(pcd, f32), numpy.array(rgba, f32)[:, :3]], axis=-1)


def trimesh_to_pc(scene_or_mesh):
    if isinstance(scene_or_mesh, trimesh.Scene):
        meshes = [
            model_to_pc(trimesh.Trimesh(vertices=g.vertices, faces=g.faces), 10000 // len(scene_or_mesh.geometry))
            for g in scene_or_mesh.geometry.values()
            if isinstance(g, trimesh.Trimesh)
        ]
        if not len(meshes):
            return None
        return numpy.concatenate(meshes)
    else:
        assert isinstance(scene_or_mesh, trimesh.Trimesh)
        return model_to_pc(scene_or_mesh, 10000)
