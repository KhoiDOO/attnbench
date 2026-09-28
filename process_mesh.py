#!/usr/bin/env python3
"""Import an OBJ file into Blender collection 'main', name it 'mesh',
stand it upright facing front, normalize it to center at origin (0, 0, 0)
and scale to [-5, -5, -5] to [5, 5, 5] (a box), apply crystal-clear luminous glass material,
configure a light, bright studio environment with scaled edge-defining lighting,
and export to PLY, GLB, USDZ, OBJ+MTL, and a high-resolution render.

Usage:
    blender main.blend --background --python process_mesh.py -- <path_to_obj> <output_ply_or_dir>
"""

import math
import os
import sys
import bpy


def parse_arguments():
    """Parse command line arguments passed after '--' or directly."""
    if "--" in sys.argv:
        args = sys.argv[sys.argv.index("--") + 1 :]
    else:
        script_name = os.path.basename(__file__)
        args = []
        for idx, arg in enumerate(sys.argv):
            if arg.endswith(script_name):
                args = sys.argv[idx + 1 :]
                break
        if not args:
            args = sys.argv[1:]

    if len(args) < 2:
        print(
            "Error: Missing required arguments.\n"
            "Usage: blender [blend_file] --background --python process_mesh.py -- <path_to_obj> <output_ply_or_dir>"
        )
        sys.exit(1)

    obj_path = os.path.abspath(args[0])
    raw_output_path = args[1]

    if not os.path.isfile(obj_path):
        raise FileNotFoundError(f"OBJ file not found: {obj_path}")

    # Determine target PLY file path and output directory
    obj_stem = os.path.splitext(os.path.basename(obj_path))[0]
    if os.path.isdir(raw_output_path) or raw_output_path.endswith(("/", "\\")):
        out_dir = os.path.abspath(raw_output_path)
        ply_path = os.path.join(out_dir, f"{obj_stem}.ply")
    elif not raw_output_path.lower().endswith(".ply"):
        if os.path.isdir(raw_output_path):
            out_dir = os.path.abspath(raw_output_path)
            ply_path = os.path.join(out_dir, f"{obj_stem}.ply")
        else:
            ply_path = os.path.abspath(f"{raw_output_path}.ply")
            out_dir = os.path.dirname(ply_path)
    else:
        ply_path = os.path.abspath(raw_output_path)
        out_dir = os.path.dirname(ply_path)

    os.makedirs(out_dir, exist_ok=True)
    return obj_path, ply_path, out_dir


def create_clear_crystal_glass_material(name="Clear_Crystal_Glass"):
    """Create a physically-based, luminous crystal clear glass material."""
    mat = bpy.data.materials.get(name)
    if mat is None:
        mat = bpy.data.materials.new(name=name)

    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    nodes.clear()

    # Principled BSDF configured for clear crystal glass
    bsdf = nodes.new(type="ShaderNodeBsdfPrincipled")
    bsdf.location = (0, 0)
    output = nodes.new(type="ShaderNodeOutputMaterial")
    output.location = (300, 0)
    mat.node_tree.links.new(bsdf.outputs["BSDF"], output.inputs["Surface"])

    # Pure optical glass clarity
    bsdf.inputs["Base Color"].default_value = (1.0, 1.0, 1.0, 1.0)
    # 100% Light transmission
    if "Transmission Weight" in bsdf.inputs:
        bsdf.inputs["Transmission Weight"].default_value = 1.0
    elif "Transmission" in bsdf.inputs:
        bsdf.inputs["Transmission"].default_value = 1.0
    # Crown glass optical IOR
    if "IOR" in bsdf.inputs:
        bsdf.inputs["IOR"].default_value = 1.50
    # Polished crystal smoothness
    if "Roughness" in bsdf.inputs:
        bsdf.inputs["Roughness"].default_value = 0.008
    # Specular reflection
    if "Specular IOR Level" in bsdf.inputs:
        bsdf.inputs["Specular IOR Level"].default_value = 0.55

    # Subtle prism sparkle on edges
    if "Thin Film Thickness" in bsdf.inputs:
        bsdf.inputs["Thin Film Thickness"].default_value = 140.0
        bsdf.inputs["Thin Film IOR"].default_value = 1.33

    # Viewport settings
    if hasattr(mat, "surface_render_method"):
        mat.surface_render_method = "DITHERED"
    if hasattr(mat, "use_raytrace_refraction"):
        mat.use_raytrace_refraction = True
    if hasattr(mat, "use_screen_refraction"):
        mat.use_screen_refraction = True
    if hasattr(mat, "diffuse_color"):
        mat.diffuse_color = (0.95, 0.98, 1.0, 0.15)

    return mat


def setup_light_studio_environment(min_z=-5.0):
    """Create a bright, light studio environment scaled and positioned for a normalized mesh (centered at origin, bounds in [-5, 5])."""
    scene = bpy.context.scene

    # 1. Studio Collection
    studio_col = bpy.data.collections.get("Studio")
    if studio_col is None:
        studio_col = bpy.data.collections.new("Studio")
        scene.collection.children.link(studio_col)

    # Clean existing studio objects
    for obj in list(studio_col.objects):
        bpy.data.objects.remove(obj, do_unlink=True)

    # 2. Light Studio World
    world = bpy.data.worlds.get("LightStudioWorld")
    if world is None:
        world = bpy.data.worlds.new("LightStudioWorld")
    scene.world = world
    world.use_nodes = True
    bg = world.node_tree.nodes.get("Background")
    if bg:
        bg.inputs["Color"].default_value = (0.84, 0.85, 0.88, 1.0)
        bg.inputs["Strength"].default_value = 0.85

    # 3. Light Pedestal with soft reflection (top surface at min_z)
    ped_depth = 0.6
    bpy.ops.mesh.primitive_cylinder_add(radius=12.0, depth=ped_depth, location=(0, 0, min_z - ped_depth / 2.0))
    ped = bpy.context.active_object
    ped.name = "Pedestal"
    studio_col.objects.link(ped)
    scene.collection.objects.unlink(ped)
    ped_mat = bpy.data.materials.new("LightPedestalMat")
    ped_mat.use_nodes = True
    pbsdf = ped_mat.node_tree.nodes.get("Principled BSDF")
    if pbsdf:
        pbsdf.inputs["Base Color"].default_value = (0.80, 0.82, 0.85, 1.0)
        pbsdf.inputs["Roughness"].default_value = 0.15
    ped.data.materials.append(ped_mat)

    # 4. Light Backdrop (centered at Z=0 behind the mesh)
    bpy.ops.mesh.primitive_plane_add(size=80.0, location=(0, 14.0, 0.0), rotation=(math.radians(90), 0, 0))
    back = bpy.context.active_object
    back.name = "Backdrop"
    studio_col.objects.link(back)
    scene.collection.objects.unlink(back)
    bmat = bpy.data.materials.new("LightBackdropMat")
    bmat.use_nodes = True
    bbsdf = bmat.node_tree.nodes.get("Principled BSDF")
    if bbsdf:
        bbsdf.inputs["Base Color"].default_value = (0.86, 0.87, 0.90, 1.0)
        bbsdf.inputs["Roughness"].default_value = 0.6
    back.data.materials.append(bmat)

    # 5. Studio Lights (scaled to normalized ~10 unit scene)
    # Key Light (front left)
    key = bpy.data.lights.new("KeySoftbox", "AREA")
    key.energy = 22.0
    key.size = 12.0
    key.size_y = 8.0
    key_obj = bpy.data.objects.new("KeySoftbox", key)
    studio_col.objects.link(key_obj)
    key_obj.location = (-10.5, -12.0, 5.0)
    key_obj.rotation_euler = (math.radians(50), math.radians(10), math.radians(-40))

    # Fill Light (front right)
    fill = bpy.data.lights.new("FillSoftbox", "AREA")
    fill.energy = 13.0
    fill.size = 12.0
    fill.size_y = 8.0
    fill_obj = bpy.data.objects.new("FillSoftbox", fill)
    studio_col.objects.link(fill_obj)
    fill_obj.location = (10.5, -12.0, 3.5)
    fill_obj.rotation_euler = (math.radians(50), math.radians(-10), math.radians(40))

    # Left & Right Edge Strips (give glass clear, crisp outlines)
    left_edge = bpy.data.lights.new("LeftEdge", "AREA")
    left_edge.energy = 26.0
    left_edge.size = 2.6
    left_edge.size_y = 18.5
    left_edge_obj = bpy.data.objects.new("LeftEdge", left_edge)
    studio_col.objects.link(left_edge_obj)
    left_edge_obj.location = (-10.0, -2.6, 2.3)
    left_edge_obj.rotation_euler = (math.radians(15), math.radians(5), math.radians(-75))

    right_edge = bpy.data.lights.new("RightEdge", "AREA")
    right_edge.energy = 26.0
    right_edge.size = 2.6
    right_edge.size_y = 18.5
    right_edge_obj = bpy.data.objects.new("RightEdge", right_edge)
    studio_col.objects.link(right_edge_obj)
    right_edge_obj.location = (10.0, -2.6, 2.3)
    right_edge_obj.rotation_euler = (math.radians(15), math.radians(-5), math.radians(75))

    # Top Rim Light
    rim = bpy.data.lights.new("TopRim", "AREA")
    rim.energy = 22.0
    rim.size = 12.0
    rim.size_y = 5.3
    rim_obj = bpy.data.objects.new("TopRim", rim)
    studio_col.objects.link(rim_obj)
    rim_obj.location = (0, 4.6, 12.2)
    rim_obj.rotation_euler = (math.radians(-30), 0, math.radians(180))

    # Dark contour cards (far on sides, out of camera view, carving out clean edge definition)
    cmat = bpy.data.materials.new("DarkCardMat")
    cmat.use_nodes = True
    cmat.node_tree.nodes["Principled BSDF"].inputs["Base Color"].default_value = (0.05, 0.05, 0.05, 1.0)

    bpy.ops.mesh.primitive_plane_add(size=20.0, location=(-13.5, 0, 0), rotation=(0, math.radians(90), 0))
    card_l = bpy.context.active_object
    card_l.name = "DarkCardLeft"
    studio_col.objects.link(card_l)
    scene.collection.objects.unlink(card_l)
    card_l.data.materials.append(cmat)

    bpy.ops.mesh.primitive_plane_add(size=20.0, location=(13.5, 0, 0), rotation=(0, math.radians(90), 0))
    card_r = bpy.context.active_object
    card_r.name = "DarkCardRight"
    studio_col.objects.link(card_r)
    scene.collection.objects.unlink(card_r)
    card_r.data.materials.append(cmat)

    # 6. Hero Camera (Framing full upright figure centered at origin)
    cam_data = bpy.data.cameras.new("StudioCamera")
    cam_obj = bpy.data.objects.new("StudioCamera", cam_data)
    studio_col.objects.link(cam_obj)
    scene.camera = cam_obj

    cam_obj.location = (-1.3, -19.2, 1.1)
    cam_obj.rotation_euler = (math.radians(86.7), 0, math.radians(-3.9))
    cam_data.lens = 65.0
    cam_data.clip_start = 0.1
    cam_data.clip_end = 1000.0


def configure_render_engine():
    """Configure Cycles path tracer with Apple Metal GPU acceleration."""
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.cycles.use_denoising = True
    scene.cycles.samples = 64

    try:
        cycles_prefs = bpy.context.preferences.addons["cycles"].preferences
        device_types = cycles_prefs.get_device_types(bpy.context)
        metal_available = any(dt[0] == "METAL" for dt in device_types)
        if metal_available:
            cycles_prefs.compute_device_type = "METAL"
            for dev in cycles_prefs.devices:
                if dev.type == "METAL":
                    dev.use = True
            scene.cycles.device = "GPU"
            print("Enabled Cycles Metal GPU acceleration.")
    except Exception as e:
        print(f"Cycles device notice: {e}")


def process_mesh(obj_path, ply_path, out_dir):
    print(f"Processing OBJ: {obj_path}")
    print(f"Target PLY:     {ply_path}")
    obj_stem = os.path.splitext(os.path.basename(obj_path))[0]

    # 1. Ensure collection 'main' exists and is linked
    main_col = bpy.data.collections.get("main")
    if main_col is None:
        main_col = bpy.data.collections.new("main")
        bpy.context.scene.collection.children.link(main_col)
        print("Created collection 'main'.")
    else:
        if main_col.name not in bpy.context.scene.collection.children:
            bpy.context.scene.collection.children.link(main_col)

    # 2. Clean up any existing object named 'mesh'
    existing_mesh_obj = bpy.data.objects.get("mesh")
    if existing_mesh_obj:
        bpy.data.objects.remove(existing_mesh_obj, do_unlink=True)

    # Purge unused mesh data and materials
    for m in list(bpy.data.meshes):
        if m.users == 0:
            bpy.data.meshes.remove(m)
    for mat in list(bpy.data.materials):
        if mat.users == 0:
            bpy.data.materials.remove(mat)

    # 3. Import the OBJ file
    objects_before = set(bpy.data.objects)
    if hasattr(bpy.ops.wm, "obj_import"):
        bpy.ops.wm.obj_import(filepath=obj_path)
    elif hasattr(bpy.ops.import_scene, "obj"):
        bpy.ops.import_scene.obj(filepath=obj_path)
    else:
        raise RuntimeError("No OBJ import operator found in Blender.")

    new_objects = [o for o in bpy.data.objects if o not in objects_before]
    if not new_objects:
        raise RuntimeError(f"No objects imported from {obj_path}")

    mesh_objects = [o for o in new_objects if o.type == "MESH"]
    target_obj = mesh_objects[0] if mesh_objects else new_objects[0]

    # 4. Rename object and mesh data to 'mesh'
    target_obj.name = "mesh"
    if target_obj.data:
        target_obj.data.name = "mesh"

    # Stand upright (+90X) and face forward (180Z)
    target_obj.rotation_euler = (math.radians(90), 0, math.radians(180))
    bpy.ops.object.transform_apply(rotation=True)

    # Enable smooth shading
    if target_obj.data:
        for poly in target_obj.data.polygons:
            poly.use_smooth = True

    # Normalize: center at origin (0, 0, 0) and scale into [-5, -5, -5] to [5, 5, 5] (a box)
    coords = [v.co for v in target_obj.data.vertices]
    min_x = min(v.x for v in coords)
    max_x = max(v.x for v in coords)
    min_y = min(v.y for v in coords)
    max_y = max(v.y for v in coords)
    min_z = min(v.z for v in coords)
    max_z = max(v.z for v in coords)

    center_x = (min_x + max_x) / 2.0
    center_y = (min_y + max_y) / 2.0
    center_z = (min_z + max_z) / 2.0

    max_extent = max(max_x - min_x, max_y - min_y, max_z - min_z)
    scale_factor = 10.0 / max_extent if max_extent > 0 else 1.0

    target_obj.location = (-center_x, -center_y, -center_z)
    bpy.ops.object.transform_apply(location=True)
    target_obj.scale = (scale_factor, scale_factor, scale_factor)
    bpy.ops.object.transform_apply(scale=True)

    norm_coords = [v.co for v in target_obj.data.vertices]
    norm_min_z = min(v.z for v in norm_coords)
    print(f"Normalized mesh: centered at origin (0, 0, 0), scaled by factor {scale_factor:.6f} to fit inside [-5, 5]^3 box.")

    # 5. Move object exclusively into collection 'main'
    if target_obj.name not in main_col.objects:
        main_col.objects.link(target_obj)

    for col in list(target_obj.users_collection):
        if col != main_col:
            col.objects.unlink(target_obj)
    print(f"Linked '{target_obj.name}' upright in collection 'main'.")

    # 6. Create and assign clear crystal glass material
    glass_mat = create_clear_crystal_glass_material("Clear_Crystal_Glass")
    if target_obj.data:
        target_obj.data.materials.clear()
        target_obj.data.materials.append(glass_mat)
    print("Created and assigned clear crystal glass material.")

    # Select target object
    bpy.ops.object.select_all(action="DESELECT")
    target_obj.select_set(True)
    bpy.context.view_layer.objects.active = target_obj

    # 7. Export mesh to PLY
    if hasattr(bpy.ops.wm, "ply_export"):
        bpy.ops.wm.ply_export(filepath=ply_path, export_selected_objects=True)
    elif hasattr(bpy.ops.export_mesh, "ply"):
        bpy.ops.export_mesh.ply(filepath=ply_path, use_selection=True)
    print(f"Exported mesh to PLY: {ply_path}")

    # 8. Export to formats preserving clear glass:
    # A) GLB with KHR_materials_transmission
    glb_path = os.path.join(out_dir, f"{obj_stem}_transparent.glb")
    if hasattr(bpy.ops.export_scene, "gltf"):
        bpy.ops.export_scene.gltf(
            filepath=glb_path,
            export_format="GLB",
            use_selection=True,
        )
        print(f"Exported material-preserved mesh to GLB: {glb_path}")

    # B) USDZ (Apple QuickLook optical glass)
    usdz_path = os.path.join(out_dir, f"{obj_stem}_transparent.usdz")
    if hasattr(bpy.ops.wm, "usd_export"):
        try:
            bpy.ops.wm.usd_export(
                filepath=usdz_path,
                selected_objects_only=True,
            )
            print(f"Exported material-preserved mesh to USDZ: {usdz_path}")
        except Exception as e:
            print(f"USDZ export notice: {e}")

    # C) OBJ + MTL with optical density
    obj_out_path = os.path.join(out_dir, f"{obj_stem}_transparent.obj")
    if hasattr(bpy.ops.wm, "obj_export"):
        bpy.ops.wm.obj_export(
            filepath=obj_out_path,
            export_selected_objects=True,
        )
        print(f"Exported material-preserved mesh to OBJ+MTL: {obj_out_path}")

    # 9. Setup light studio environment and lighting scaled to normalized bounds
    setup_light_studio_environment(min_z=norm_min_z)

    # 10. Configure Cycles Metal GPU
    configure_render_engine()

    # 11. Render hero image
    render_path = os.path.join(out_dir, f"{obj_stem}_render.png")
    scene = bpy.context.scene
    scene.render.resolution_x = 1200
    scene.render.resolution_y = 1200
    scene.render.filepath = render_path
    print(f"Rendering normalized hero image to: {render_path}")
    bpy.ops.render.render(write_still=True)
    print(f"Render complete: {render_path}")

    # 12. Save the blend file
    if bpy.data.filepath:
        bpy.ops.wm.save_mainfile()
        print(f"Saved blend file to: {bpy.data.filepath}")


def main():
    obj_path, ply_path, out_dir = parse_arguments()
    process_mesh(obj_path, ply_path, out_dir)


if __name__ == "__main__":
    main()
