#!/usr/bin/env python3
"""Import an OBJ file into Blender collection 'main', name it 'mesh',

apply a realistic physical crystal glass material (pure refraction, Fresnel, caustics,
zero tree/environment reflections), configure studio softbox lighting, set viewport to
real-time Cycles Metal GPU rendering, and export to PLY, GLB, USDZ, and OBJ+MTL.

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


def create_realistic_glass_material(name="Realistic_Glass"):
    """Create a physically-based crystal glass material with 100% transmission and optical IOR."""
    mat = bpy.data.materials.get(name)
    if mat is None:
        mat = bpy.data.materials.new(name=name)

    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    nodes.clear()

    # Principled BSDF configured for physical glass
    bsdf = nodes.new(type="ShaderNodeBsdfPrincipled")
    bsdf.location = (0, 0)
    output = nodes.new(type="ShaderNodeOutputMaterial")
    output.location = (300, 0)
    mat.node_tree.links.new(bsdf.outputs["BSDF"], output.inputs["Surface"])

    # Pure clear optical glass parameters
    bsdf.inputs["Base Color"].default_value = (1.0, 1.0, 1.0, 1.0)
    # 100% Optical light transmission
    if "Transmission Weight" in bsdf.inputs:
        bsdf.inputs["Transmission Weight"].default_value = 1.0
    elif "Transmission" in bsdf.inputs:
        bsdf.inputs["Transmission"].default_value = 1.0
    # Optical glass Index of Refraction (standard crown glass is 1.50 - 1.52)
    if "IOR" in bsdf.inputs:
        bsdf.inputs["IOR"].default_value = 1.50
    # Ultra-smooth polished glass surface
    if "Roughness" in bsdf.inputs:
        bsdf.inputs["Roughness"].default_value = 0.01
    # Physical Fresnel specular reflection
    if "Specular IOR Level" in bsdf.inputs:
        bsdf.inputs["Specular IOR Level"].default_value = 0.5

    # EEVEE Next & Viewport properties
    if hasattr(mat, "surface_render_method"):
        mat.surface_render_method = "DITHERED"
    if hasattr(mat, "use_raytrace_refraction"):
        mat.use_raytrace_refraction = True
    if hasattr(mat, "use_screen_refraction"):
        mat.use_screen_refraction = True
    if hasattr(mat, "refraction_depth"):
        mat.refraction_depth = 15.0
    if hasattr(mat, "diffuse_color"):
        mat.diffuse_color = (0.95, 0.98, 1.0, 0.15)

    return mat


def setup_studio_environment():
    """Create a neutral photography studio environment with softbox lights and neutral backdrop (no trees)."""
    scene = bpy.context.scene

    # 1. Studio Collection
    studio_col = bpy.data.collections.get("Studio")
    if studio_col is None:
        studio_col = bpy.data.collections.new("Studio")
        scene.collection.children.link(studio_col)

    # Clean existing studio objects
    for obj in list(studio_col.objects):
        bpy.data.objects.remove(obj, do_unlink=True)

    # 2. Neutral Studio World
    world = bpy.data.worlds.get("StudioWorld")
    if world is None:
        world = bpy.data.worlds.new("StudioWorld")
    scene.world = world
    world.use_nodes = True
    bg = world.node_tree.nodes.get("Background")
    if bg:
        bg.inputs["Color"].default_value = (0.85, 0.85, 0.88, 1.0)
        bg.inputs["Strength"].default_value = 0.8

    # 3. Softbox Lights
    # Key Softbox (left front)
    key_light = bpy.data.lights.new("KeySoftbox", "AREA")
    key_light.energy = 8000.0
    key_light.size = 200.0
    key_light.size_y = 100.0
    key_obj = bpy.data.objects.new("KeySoftbox", key_light)
    studio_col.objects.link(key_obj)
    key_obj.location = (-150, -180, 150)
    key_obj.rotation_euler = (math.radians(50), math.radians(10), math.radians(-35))

    # Fill Softbox (right front)
    fill_light = bpy.data.lights.new("FillSoftbox", "AREA")
    fill_light.energy = 3000.0
    fill_light.size = 180.0
    fill_obj = bpy.data.objects.new("FillSoftbox", fill_light)
    studio_col.objects.link(fill_obj)
    fill_obj.location = (160, -140, 80)
    fill_obj.rotation_euler = (math.radians(60), 0, math.radians(45))

    # Rim Softbox (behind and above - creates crystal edge gleam)
    rim_light = bpy.data.lights.new("RimSoftbox", "AREA")
    rim_light.energy = 6000.0
    rim_light.size = 150.0
    rim_obj = bpy.data.objects.new("RimSoftbox", rim_light)
    studio_col.objects.link(rim_obj)
    rim_obj.location = (0, 160, 180)
    rim_obj.rotation_euler = (math.radians(-50), 0, math.radians(180))

    # 4. Neutral Studio Backdrop Plane
    backdrop_mesh = bpy.data.meshes.new("StudioBackdrop")
    backdrop_obj = bpy.data.objects.new("StudioBackdrop", backdrop_mesh)
    studio_col.objects.link(backdrop_obj)

    # Create plane geometry
    s = 300.0
    verts = [(-s, -s + 50, -50), (s, -s + 50, -50), (s, s + 50, -50), (-s, s + 50, -50)]
    faces = [(0, 1, 2, 3)]
    backdrop_mesh.from_pydata(verts, [], faces)
    backdrop_mesh.update()

    backdrop_mat = bpy.data.materials.get("StudioBackdropMat")
    if backdrop_mat is None:
        backdrop_mat = bpy.data.materials.new("StudioBackdropMat")
    backdrop_mat.use_nodes = True
    bsdf = backdrop_mat.node_tree.nodes.get("Principled BSDF")
    if bsdf:
        bsdf.inputs["Base Color"].default_value = (0.7, 0.72, 0.75, 1.0)
        bsdf.inputs["Roughness"].default_value = 0.4
    backdrop_obj.data.materials.append(backdrop_mat)

    # 5. Studio Camera
    cam_data = bpy.data.cameras.get("StudioCamera")
    if cam_data is None:
        cam_data = bpy.data.cameras.new("StudioCamera")
    cam_obj = bpy.data.objects.get("StudioCamera")
    if cam_obj is None:
        cam_obj = bpy.data.objects.new("StudioCamera", cam_data)
        studio_col.objects.link(cam_obj)
    scene.camera = cam_obj
    cam_obj.location = (0, -220, 20)
    cam_obj.rotation_euler = (math.radians(83), 0, 0)


def configure_render_engine_and_viewport():
    """Configure Cycles with Apple Metal GPU and set viewport to real-time Rendered mode."""
    scene = bpy.context.scene

    # Set Cycles path tracing engine
    scene.render.engine = "CYCLES"
    scene.cycles.use_denoising = True
    scene.cycles.samples = 64

    # Enable Apple Silicon Metal GPU acceleration if available
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

    # Set 3D Viewports to RENDERED mode for real-time ray-traced glass
    for screen in bpy.data.screens:
        for area in screen.areas:
            if area.type == "VIEW_3D":
                for space in area.spaces:
                    if space.type == "VIEW_3D":
                        space.shading.type = "RENDERED"
                        space.shading.use_scene_lights = True
                        space.shading.use_scene_world = True


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

    # Find the mesh object
    mesh_objects = [o for o in new_objects if o.type == "MESH"]
    target_obj = mesh_objects[0] if mesh_objects else new_objects[0]

    # 4. Rename object and mesh data to 'mesh'
    target_obj.name = "mesh"
    if target_obj.data:
        target_obj.data.name = "mesh"
    print(f"Renamed imported object to: {target_obj.name}")

    # Enable smooth shading
    if target_obj.data:
        for poly in target_obj.data.polygons:
            poly.use_smooth = True

    # 5. Move object exclusively into collection 'main'
    if target_obj.name not in main_col.objects:
        main_col.objects.link(target_obj)

    for col in list(target_obj.users_collection):
        if col != main_col:
            col.objects.unlink(target_obj)
    print(f"Linked '{target_obj.name}' to collection 'main'.")

    # 6. Create and assign realistic physical glass material
    glass_mat = create_realistic_glass_material("Realistic_Glass")
    if target_obj.data:
        target_obj.data.materials.clear()
        target_obj.data.materials.append(glass_mat)
    print("Created and assigned realistic physical glass material.")

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

    # 8. Export to formats that preserve physical glass material:
    # A) GLB (glTF 2.0 Binary) with KHR_materials_transmission & KHR_materials_ior
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

    # C) OBJ + MTL with optical density (Ni 1.5, d 0.05)
    obj_out_path = os.path.join(out_dir, f"{obj_stem}_transparent.obj")
    if hasattr(bpy.ops.wm, "obj_export"):
        bpy.ops.wm.obj_export(
            filepath=obj_out_path,
            export_selected_objects=True,
        )
        print(f"Exported material-preserved mesh to OBJ+MTL: {obj_out_path}")

    # 9. Setup studio softbox lighting and neutral background
    setup_studio_environment()

    # 10. Configure Cycles GPU (Metal) and real-time Rendered viewport
    configure_render_engine_and_viewport()

    # 11. Save the blend file
    if bpy.data.filepath:
        bpy.ops.wm.save_mainfile()
        print(f"Saved blend file to: {bpy.data.filepath}")


def main():
    obj_path, ply_path, out_dir = parse_arguments()
    process_mesh(obj_path, ply_path, out_dir)


if __name__ == "__main__":
    main()
