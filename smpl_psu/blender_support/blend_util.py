import bpy
import mathutils
import math
import os 
import numpy as np
import glob
from tqdm import tqdm

import bpy
import os
from glob import glob

def create_animation(cfg):
    # Paths
    blend_dir  = cfg['dirs']['blend_dir']
    save_path  = os.path.join(cfg['dirs']['save_dir'], 'animation_shapekeys.blend')
    blend_files = sorted(glob(os.path.join(blend_dir, "frame_*.blend")))

    # — Start fresh —
    bpy.ops.wm.read_factory_settings(use_empty=True)
    z_offset = 0.0 if cfg['motion']['coordinate_system'] == 'global' else -0.85
    setup_floor(floor_size=cfg['env']['floor_size'], z_offset=z_offset)
    
    setup_light(cfg)
    setup_camera(cfg)

    # — 1) Append the very first mesh as our “master” object —
    with bpy.data.libraries.load(blend_files[0], link=False) as (src, dst):
        meshes = [name for name in src.meshes if name.startswith("frame_")]
        dst.meshes = meshes[:1]  # take the first

    mesh       = bpy.data.meshes[meshes[0]]
    master_obj = bpy.data.objects.new(mesh.name, mesh)
    # Rotate the **object** 90° around X
    master_obj.rotation_euler = (math.radians(90), 0.0, 0.0)
    # Link and activate
    bpy.context.collection.objects.link(master_obj)
    bpy.context.view_layer.objects.active = master_obj

    # — 2) Append each subsequent mesh as its own object —
    temp_objs = []
    for _, bf in tqdm(enumerate(blend_files[1:]), desc="Appending meshes", total=len(blend_files)-1):
        with bpy.data.libraries.load(bf, link=False) as (src, dst):
            mnames = [n for n in src.meshes if n.startswith("frame_")]
            dst.meshes = mnames[:1]
        mesh = bpy.data.meshes[mnames[0]]
        obj  = bpy.data.objects.new(mesh.name, mesh)
        # Rotate the temp object too
        obj.rotation_euler = (math.radians(90), 0.0, 0.0)
        bpy.context.collection.objects.link(obj)
        temp_objs.append(obj)

    # — 3) Convert all those meshes into shape-keys on master —
    master_obj.select_set(True)
    for o in temp_objs:
        o.select_set(True)
    # This bakes each rotated temp into a shape-key on the rotated master
    bpy.ops.object.join_shapes()

    # Remove the temps
    for o in temp_objs:
        bpy.data.objects.remove(o, do_unlink=True)

    # — 4) Animate each shape-key’s influence —
    key_blocks = master_obj.data.shape_keys.key_blocks
    for idx, key in tqdm(enumerate(key_blocks[1:], start=1), desc="Animating shape-keys", total=len(key_blocks)-1):
        frame = idx
        key.value = 0.0
        key.keyframe_insert("value", frame=frame-1)
        key.value = 1.0
        key.keyframe_insert("value", frame=frame)
        key.value = 0.0
        key.keyframe_insert("value", frame=frame+1)

    # — 5) Finalize and save —
    bpy.context.scene.frame_start = 1
    bpy.context.scene.frame_end   = len(blend_files)
    if os.path.exists(save_path):
        os.remove(save_path)
    bpy.ops.wm.save_as_mainfile(filepath=save_path)
    print(f"[INFO] Saved animation with shape-keys to: {save_path}")

def setup_light(cfg):
    # remove any existing lights
    for obj in list(bpy.data.objects):
        if obj.type == 'LIGHT':
            bpy.data.objects.remove(obj, do_unlink=True)

    # add a Sun lamp
    bpy.ops.object.light_add(type='SUN', location=(0, 0, 10))
    sun = bpy.context.active_object

    # aim it down at roughly 45° toward the origin
    sun.rotation_euler = (
        math.radians(0),   # tilt down
        math.radians(0),
        math.radians(180)    # rotate around Z
    )

    # energy controls brightness
    if cfg['motion']['coordinate_system'] == 'local':
        sun.data.energy = 3.0
    else:
        sun.data.energy = 3.0

    # angle controls softness of shadows (bigger = softer)
    sun.data.angle = math.radians(5)   # try 5°–10° for pleasing penumbra
    return sun

def setup_camera(cfg):
    if cfg['motion']['coordinate_system'] == 'local':
        tgt_z = -0.30
        tgt_loc = mathutils.Vector((0.0, 0.0, tgt_z))
 
        distance = 7.0
        height = 2.0
        cam_pos = mathutils.Vector((0.0, -distance, height))

        # Create camera and target
        bpy.ops.object.camera_add(location=cam_pos)
        cam = bpy.context.active_object
        cam.name = "TrackingCamera"

        tgt = bpy.data.objects.new("CamTarget", None)
        bpy.context.scene.collection.objects.link(tgt)
        tgt.location = tgt_loc

        # Apply tracking constraint
        constraint = cam.constraints.new('TRACK_TO')
        constraint.target = tgt
        constraint.track_axis = 'TRACK_NEGATIVE_Z'
        constraint.up_axis = 'UP_Y'

        bpy.context.scene.camera = cam
        return cam

    else:
        square_half_size = cfg['env']['floor_size'] / 2.0
        margin = 1.9
        height = 3.5
        target_z = -0.5  # slightly above ground (e.g., pelvis or torso)

        camera_view = cfg['render']['camera_view']
        if camera_view == 1:
            cam_x = -square_half_size * margin
            cam_y =  square_half_size * margin
        elif camera_view == 2:
            cam_x =  square_half_size * margin
            cam_y =  square_half_size * margin
        else:
            raise ValueError("Unsupported camera_view: should be 1 or 2")

        cam_pos = mathutils.Vector((cam_x, cam_y, height))
        tgt_pos = mathutils.Vector((0.0, 0.0, target_z))
        direction = (tgt_pos - cam_pos).normalized()

        # Create camera and aim it
        bpy.ops.object.camera_add(location=cam_pos)
        cam = bpy.context.active_object
        cam.name = "TrackingCamera"

        # Point camera toward target
        cam_direction = -direction
        up = mathutils.Vector((0, 0, 1))
        right = up.cross(cam_direction).normalized()
        real_up = cam_direction.cross(right).normalized()

        rot_matrix = mathutils.Matrix((right, real_up, cam_direction)).transposed()
        cam.rotation_euler = rot_matrix.to_euler()
        # Set as scene camera
        bpy.context.scene.camera = cam
        return cam

def setup_floor(floor_size=4, z_offset=-0.85):
    # 1) Create or grab the plane object
    if "CheckeredFloor" in bpy.data.objects:
        floor = bpy.data.objects["CheckeredFloor"]
    else:
        bpy.ops.mesh.primitive_plane_add(size=floor_size, location=(0, 0, z_offset))
        floor = bpy.context.active_object
        floor.name = "CheckeredFloor"

    # 2) Create the checker material if needed
    if "CheckeredMaterial" not in bpy.data.materials:
        mat = bpy.data.materials.new(name="CheckeredMaterial")
        mat.use_nodes = True
        nodes = mat.node_tree.nodes
        links = mat.node_tree.links

        # clear default nodes
        for n in nodes:
            nodes.remove(n)

        # build the node graph
        out_node     = nodes.new(type='ShaderNodeOutputMaterial')
        bsdf_node    = nodes.new(type='ShaderNodeBsdfPrincipled')
        checker_node = nodes.new(type='ShaderNodeTexChecker')
        coord_node   = nodes.new(type='ShaderNodeTexCoord')
        mapping_node = nodes.new(type='ShaderNodeMapping')

        # configure checker
        checker_node.inputs['Color1'].default_value = (0.8, 0.8, 0.8, 1.0)
        checker_node.inputs['Color2'].default_value = (0.2, 0.2, 0.2, 1.0)
        mapping_node.inputs['Scale'].default_value   = (10.0, 10.0, 10.0)

        # wire it up
        links.new(coord_node.outputs['UV'],          mapping_node.inputs['Vector'])
        links.new(mapping_node.outputs['Vector'],    checker_node.inputs['Vector'])
        links.new(checker_node.outputs['Color'],     bsdf_node.inputs['Base Color'])
        links.new(bsdf_node.outputs['BSDF'],         out_node.inputs['Surface'])

        # kill all reflections
        bsdf_node.inputs['Roughness'].default_value = 1.0
        if "Specular" in bsdf_node.inputs:
            bsdf_node.inputs["Specular"].default_value = 0.0
        # If using Blender 4.x: also try this
        else:
            bsdf_node.inputs["Specular IOR Level"].default_value = 0.0

    # 3) Assign the material
    floor.data.materials.clear()
    floor.data.materials.append(bpy.data.materials["CheckeredMaterial"])
    return floor

def apply_render_settings(cfg, res_multiplier=1.0, png_compression=20, use_jpeg=False, jpeg_quality=90):
    sc = bpy.context.scene
    sc.render.engine = cfg['render']['engine']
    w, h = cfg['render']['resolution']
    sc.render.resolution_x = int(w * res_multiplier)
    sc.render.resolution_y = int(h * res_multiplier)
    sc.render.resolution_percentage = 100  # always interpret resolution_x/y as final pixels

    if cfg['render']['engine'] == 'CYCLES':
        sc.cycles.device = cfg['render']['device']
        sc.cycles.samples = cfg['render']['cycles_samples']
        sc.cycles.use_denoising = cfg['render']['cycles_denoise']
    else:
        ee = sc.eevee
        ee.use_ssr         = cfg['render']['eevee_ssr']
        ee.use_gtao        = cfg['render']['eevee_gtao']
        ee.use_bloom       = cfg['render']['eevee_bloom']
        ee.shadow_cube_size = cfg['render']['eevee_shadow_size']
        
    if use_jpeg:
        sc.render.image_settings.file_format = 'JPEG'
        sc.render.image_settings.quality     = jpeg_quality  # 0–100
    else:
        sc.render.image_settings.file_format  = 'PNG'
        sc.render.image_settings.compression   = png_compression  # 0 (no) – 100 (max)

    sc.render.image_settings.color_mode = 'RGBA'  # keep alpha if you need it