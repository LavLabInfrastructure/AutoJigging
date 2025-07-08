import time
import logging
import json
import yaml
from typing import cast, Any
import argparse
import tempfile

import nibabel as nib
import numpy as np

import vtk
import cadquery as cq

from stl.mesh import Mesh
import pstats
from pstats import SortKey

profiles = {
    "prostate": {
        "organ": "prostate",
        "jig_translate_y": 15,
        "slicing_z": 1.2, 
        "x_wall": 5, 
        "y_wall": 3,
        "z_wall": 3, 
        "pre_knife_space": 37,
        "post_knife_space": 3,
        "label": 1,
        "iterations": 10000,
        "reduction": 0.97,
        "scale": (1.02, 1.02, 1.02),
        "tumor_laterality": "L"
    },

    "brain": {
        "organ": "brain",
        "jig_translate_y": 15,
        "slicing_z": 2.5,
        "x_wall": 15,
        "y_wall": 5,
        "z_wall": 5,
        "pre_knife_space": 50,
        "post_knife_space": 5,
        "label": 1,
        "iterations": 10000,
        "reduction": 0.8,
        "scale": (1.02, 1.02, 1.02),
        "tumor_laterality": "L"
    }
}

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def log_time_taken(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        logger.info(f'Function {func.__name__} took {end_time - start_time:.2f} seconds to complete.')
        return result
    return wrapper

def create_vtk_obj(obj_class, connect_port=None):
    obj_instance = obj_class()
    if connect_port is not None:
        obj_instance.SetInputConnection(connect_port)
    return obj_instance

def new_reader(nifti_path):
    reader = create_vtk_obj(vtk.vtkNIFTIImageReader, None)
    reader.SetFileName(nifti_path)
    reader.Update()
    return reader

def new_surface_extractor(connect_port, label: int = 1):
    surf = create_vtk_obj(vtk.vtkDiscreteMarchingCubes, connect_port)
    surf.SetValue(0, int(label))
    surf.Update()
    return surf

def new_smoother(connect_port, iterations: int = 10000):
    smoother = create_vtk_obj(vtk.vtkSmoothPolyDataFilter, connect_port)
    smoother.SetNumberOfIterations(iterations)
    smoother.Update()
    return smoother

def new_decimator(connect_port, target_reduction: float = 0.97):
    decimate = create_vtk_obj(vtk.vtkDecimatePro, connect_port)
    decimate.SetTargetReduction(target_reduction)
    decimate.Update()
    return decimate

def new_scaler(connect_port, scaling_factors: tuple = (1.02, 1.02, 1.02)):
    set_scaler = vtk.vtkTransform()
    set_scaler.Scale(scaling_factors)
    scaler = create_vtk_obj(vtk.vtkTransformPolyDataFilter, connect_port)
    scaler.SetTransform(set_scaler)
    scaler.Update()
    return scaler

def new_translator(connect_port, translation):
    translation_transform = vtk.vtkTransform()
    translation_transform.Translate(translation)
    translation_filter = create_vtk_obj(vtk.vtkTransformPolyDataFilter, connect_port)
    translation_filter.SetTransform(translation_transform)
    translation_filter.Update()
    return translation_filter

def new_rotator(connect_port):
    rotate_transform = vtk.vtkTransform()
    rotate_transform.RotateZ(90)
    rotate_filter = create_vtk_obj(vtk.vtkTransformPolyDataFilter, connect_port)
    rotate_filter.SetTransform(rotate_transform)
    rotate_filter.Update()
    return rotate_filter

def new_connector(connect_port):
    connector = create_vtk_obj(vtk.vtkConnectivityFilter, connect_port)
    connector.SetExtractionModeToLargestRegion()
    connector.Update()
    return connector

def find_mold_bounds(poly_data: vtk.vtkPolyData) -> tuple:
    """Finds the boundaries of given vtk polydata

    Parameters
    ----------
    poly_data : vtk.vtkPolyData
        Input vtkPolyData object

    Returns
    -------
    tuple
       6 index tuple containing the boundaries of the polydata.
        (xmin, xmax, ymin, ymax, zmin, zmax)
    """
    return poly_data.GetBounds()

def find_translation(bounds):
    """Calculates the translation needed to center the polydata based on its boundaries

    Parameters
    ----------
    final_mold_bounds : tuple
        6 index tuple containing the boundaries of the polydata.
        (xmin, xmax, ymin, ymax, zmin, zmax)

    Returns
    -------
    tuple
        3 index tuple representing the translation vector to move the center of the object to 0,0,0
        (x, y, z) 

    """
    center_x = (bounds[0] + bounds[1]) / 2
    center_y = (bounds[2] + bounds[3]) / 2
    center_z = (bounds[4] + bounds[5]) / 2
    translation = (-center_x, -center_y, -center_z)
    return translation

@log_time_taken
def prep_mold(nifti_path, label, iterations, reduction, scale, organ):
    reader = new_reader(nifti_path)
    surf = new_surface_extractor(reader.GetOutputPort(),label)
    print(organ)
    if organ == 'brain':
        decimate = new_decimator(surf.GetOutputPort(), .5)
        smoother = new_smoother(decimate.GetOutputPort(), iterations)
        connector = new_connector(smoother.GetOutputPort())
        decimate = new_decimator(connector.GetOutputPort(), reduction)
        smoother = new_smoother(decimate.GetOutputPort(), 1000)
        scaler = new_scaler(smoother.GetOutputPort(), scale)
        rotator = new_rotator(scaler.GetOutputPort())
        mold_poly = rotator.GetOutput()
        mold_bounds = find_mold_bounds(mold_poly)
        translation = find_translation(mold_bounds)
        translator = new_translator(rotator.GetOutputPort(), translation)
    else:
        smoother = new_smoother(surf.GetOutputPort(), iterations)
        decimate = new_decimator(smoother.GetOutputPort(), reduction)
        scaler = new_scaler(decimate.GetOutputPort(), scale)
        mold_poly = scaler.GetOutput()
        mold_bounds = find_mold_bounds(mold_poly)
        translation = find_translation(mold_bounds)
        translator = new_translator(scaler.GetOutputPort(), translation)
    mold_poly = translator.GetOutput()
    return mold_poly, mold_bounds

@log_time_taken
def prep_jig(mold_bounds, slice_thickness, jig_translate_y, x_wall, y_wall, z_wall, pre_knife_space, post_knife_space):
    jig_modifiers = find_jig_modifiers(slice_thickness, x_wall, y_wall, z_wall, pre_knife_space, post_knife_space)
    jig_bounds = find_jig_bounds(mold_bounds, jig_modifiers)
    jig_size = find_jig_size(jig_bounds)
    jig = (
        cq.Workplane("XY")
        .box(*jig_size)
        .translate((0, jig_translate_y, 0))
        )
    return jig, jig_size, jig_modifiers, jig_bounds


def find_jig_modifiers(slice_thickness, x_wall, y_wall, z_wall, pre_knife_space, post_knife_space) -> tuple:
    """Determines modifiers for creating a jig based on parameters extracted from the nifti

    Parameters
    ----------
    nifti_path : str
        File path to the specified nifti

    Returns
    -------
    tuple
        6 index tuple of modifiers used for adjusting jig dimensions
        (-x, +x, -y, +y, -z, +z) 
    """
    jig_modifiers = (
        -x_wall,
        x_wall,
        -(y_wall + post_knife_space),
        y_wall + pre_knife_space,
        -((slice_thickness) + z_wall),
        ((slice_thickness) + z_wall),
    )

    return jig_modifiers

def find_jig_bounds(mold_bounds, jig_modifiers) -> tuple:
    """Calculates the bounds of a jig based on a NIfTI file.

    Parameters
    ----------
    nifti_path : str
        File path to the specified nifti

    Returns
    -------
    tuple
        6 index tuple representing the boundaries of the jig
        (xmin, xmax, ymin, ymax, zmin, zmax)
    """
    jig_bounds = (
        mold_bounds[0] + jig_modifiers[0],
        mold_bounds[1] + jig_modifiers[1],
        mold_bounds[2] + jig_modifiers[2],
        mold_bounds[3] + jig_modifiers[3],
        mold_bounds[4] + jig_modifiers[4],
        mold_bounds[5] + jig_modifiers[5],
    )
    return jig_bounds

def find_jig_size(jig_bounds) -> tuple:
    """Determines the size dimensions of a jig from the nifti

    Parameters
    ----------
    nifti_path : str
        File path to the specified nifti

    Returns
    -------
    tuple
        3 index tuple representing the dimensions of the jig
        (x, y, z) 
    """
    jigsizex = jig_bounds[1] - jig_bounds[0]
    jigsizey = jig_bounds[3] - jig_bounds[2]
    jigsizez = jig_bounds[5] - jig_bounds[4]
    jig_size = (jigsizex, jigsizey, jigsizez)
    return jig_size

@log_time_taken
def prep_slicer(jig_size, slice_thickness, jig_translate_y, slicing_z, x_wall, y_wall):
    """Creates a CAD object slicer to slice the jig

    Parameters
    ----------
    nifti_path : str
        File path to the specified nifti

    Returns
    -------
    cq.Workplane
        Slicer as a CADquery Workplane object
    """
    slicing_x = jig_size[0] + (2 * x_wall)
    slicing_y = jig_size[1] - (2 * y_wall)

    slicer = (
        cq.Workplane("XY")
        .box(slicing_x, slicing_y, slicing_z)
        .translate((0, jig_translate_y, -((jig_size[2] / 2) - slice_thickness)))
    )

    return slicer


def find_slice_thickness(nifti: nib.nifti1.Nifti1Image) -> float:
    return np.round(nifti.header["pixdim"][3], 3)

def load_profile(file_path: str, profile_key) -> dict [str, Any]:
    if file_path.endswith(".json"):
        with open(file_path, "r") as f:
            profiles = json.load(f)
    elif file_path.endswith(".yaml") or file_path.endswith(".yml"):
        with open(file_path, "r") as f:
            profiles = yaml.safe_load(f)
    elif file_path.endswith(".py"):
        import importlib.util
        spec = importlib.util.spec_from_file_location("profiles_module", file_path)
        profiles_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(profiles_module)
        profiles = profiles_module.profiles

    if profile_key not in profiles:
        raise KeyError(f"Profile '{profile_key}' not found in the file.")
    return profiles[profile_key]


@log_time_taken
def write_poly(poly_data: vtk.vtkPolyData, mold_stl_path: str) -> str:
    """Writes a vtkPolyData object to an STL file

    Parameters
    ----------
    poly_data : vtk.vtkPolyData
        Input vtkPolyData object of the surface mesh to be saved
    mold_stl_path : str
        File path to where the STL file will be saved

    Returns
    -------
    str
        File path of the saved STL file
    """
    writer = create_vtk_obj(vtk.vtkSTLWriter, None)
    writer.SetInputData(poly_data)
    writer.SetFileTypeToASCII()
    writer.SetFileName(mold_stl_path)
    writer.Write()
    return mold_stl_path

def cq_more_make_polyhedron(points: tuple, faces: list) -> cq.Solid:  # from cqmore
    """Creates a polyhedral solid from given points and faces
    From cqMore https://github.com/JustinSDK/cqMore/blob/a490603a18e87e65d9a8c5d52ea0b0ce402d261c/cqmore/_solid.py#L17
    
    Parameters
    ----------
    points : tuple
        Tuple of points representing the points in the object
    faces : list
        List of lists with each sublist containing the vectors of each face

    Returns
    -------
    cq.Solid
        CADQuery Solid object created with the provided points and faces.
    """
    vectors = np.array(cq_more_to_vectors(points))

    return cq.Solid.makeSolid(
        cq.Shell.makeShell(
            cq.Face.makeFromWires(
                cq.Wire.assembleEdges(
                    cq.Edge.makeLine(*vts[[-1 + i, i]]) for i in range(vts.size)
                )
            )
            for vts in (vectors[list(face)] for face in faces)
        )
    )

def cq_more_to_vectors(points: tuple) -> tuple:  # from cqmore
    """Converts the given tuple of points to a tuple of CADQuery vectors

    Parameters
    ----------
    points : tuple
        Tuple of points representing the points in the object


    Returns
    -------
    tuple
        Tuple of CADQuery vectors, with size determined by the amount of points
    """
    if isinstance(next(iter(points)), cq.Vector):
        return cast(tuple[cq.Vector], list(points))

    return cast(tuple[cq.Vector], tuple(cq.Vector(*p) for p in points))

@log_time_taken
def poly_2_stl(mold_stl_path):
    """Imports an STL file as a mesh for CADquery

    Parameters
    ----------
    stl_path : str
        File path to the specified nifti

    Returns
    -------
    cq.Solid
        CADquery Solid object representing the geometry from the STL file

    """
    vectors = Mesh.from_file(mold_stl_path, remove_duplicate_polygons=True, remove_empty_areas=True).vectors
    points = tuple(map(tuple, vectors.reshape((vectors.shape[0] * vectors.shape[1], 3))))
    faces = [(i, i + 1, i + 2) for i in range(0, len(points), 3)]
    return cq_more_make_polyhedron(points, faces)

@log_time_taken
def make_mold_stl(mold_poly, mold_stl_path):
    if not mold_stl_path:
        with tempfile.NamedTemporaryFile(suffix=".stl", delete=True) as tmp_file:
            mold_stl_path = tmp_file.name
            write_poly(mold_poly, mold_stl_path)
            logger.info("The poly has been written")
    else:
        write_poly(mold_poly, mold_stl_path)
        logger.info("The poly has been written")

    logger.info("The poly is being converted into an stl")
    mold_stl = poly_2_stl(mold_stl_path)

    return mold_stl

@log_time_taken
def slicer_2_jig(jig, slicer, slice_thickness, jig_size, z_wall):

    assembly = cq.Workplane("XY").add(jig)

    end_z = int(np.round(jig_size[2] - (slice_thickness + z_wall), 2) * 100)
    step_z = int(np.round(slice_thickness, 3) * 100)

    slice_time = 0
    for slots in range(0, end_z, step_z):
        slice_start = time.time()
        assembly = assembly.cut(slicer.translate((0, 0, slots / 100)))
        logging.error(time.time()-slice_start)
        slice_time += time.time()-slice_start

    # print(f"Time to slicer jig: {slice_time} sec")

    compound_slicer = cq.Workplane("XY")
    for slots in range(0, end_z, step_z):
        compound_slicer = compound_slicer.union(slicer.translate((0, 0, slots / 100)))
    assembly = assembly.cut(compound_slicer)



    sliced_jig = assembly

    return sliced_jig

@log_time_taken
def mold_2_jig(mold_stl, sliced_jig, jig_size, jig_translate_y, laterality):
    scale_time = time.time()

    end_y = int(np.round((jig_size[1]) - jig_translate_y))
    assembly = sliced_jig
    mold_stl = mold_stl.scale(0.01)
    assembly = assembly.val().scale(0.01)
    if laterality == 'R':
        mold_stl = mold_stl.rotate((0, 0, 0), (0, 0, 1), 180)
    logging.info(f"Scale time {time.time()-scale_time}")


    slice_time = 0
    logging.info(f"steps to take {len(list(range(0, end_y, 2)))}")
    for i in range(0, end_y, 2):
        mold_start = time.time()
        assembly = assembly.cut(mold_stl.translate((0, i/ 100, 0)))
        percentage = (i/end_y)*100
        print(percentage)
        logging.error(time.time()-mold_start)
        slice_time += time.time()-mold_start

    # print(f"Time to cut mold: {slice_time} sec")
    assembly = assembly.scale(100)
    final_jig = assembly

    return final_jig

@log_time_taken
def assemble_jig(jig, slicer, slice_thickness, jig_size, z_wall, jig_translate_y, mold_stl, laterality):
    sliced_jig = slicer_2_jig(jig, slicer, slice_thickness, jig_size, z_wall)
    assembled_jig = mold_2_jig(mold_stl, sliced_jig, jig_size, jig_translate_y, laterality)

    return assembled_jig

# def unpack_profile(profile):
#     organ = profile["organ"]
#     jig_translate_y = profile['jig_translate_y']
#     x_wall = profile['x_wall']
#     y_wall = profile['y_wall']
#     z_wall = profile['z_wall']
#     slicing_z = profile['slicing_z']
#     pre_knife_space = profile['pre_knife_space']
#     post_knife_space = profile['post_knife_space']
#     label = profile['label']
#     iterations = profile['iterations']
#     reduction = profile['reduction']
#     scale = profile['scale']
#     if organ == 'brain':
#         laterality = profile['tumor_laterality']

#     return organ, jig_translate_y, x_wall, y_wall, z_wall, slicing_z, pre_knife_space, post_knife_space, label, iterations, reduction, scale, laterality

    

def main(nifti_path: str, mold_stl_path: str, jig_stl_path: str, profile) -> None:
    logger.info("Loading nifti and finding slice thickness.")
    nifti = nib.load(nifti_path)
    logger.info("Loading profile.")
    organ, jig_translate_y, x_wall, y_wall, z_wall, slicing_z, pre_knife_space, post_knife_space, label, iterations, reduction, scale, laterality = profile.values()

    if organ == 'brain':
        slice_thickness = find_slice_thickness(nifti)
        print(slice_thickness)
        if slice_thickness < 5:
            slice_thickness = slice_thickness * 2
    else:
        slice_thickness = find_slice_thickness(nifti)

    mold_poly, mold_bounds = prep_mold(nifti_path, label, iterations, reduction, scale, organ)
    mold_stl = make_mold_stl(mold_poly, mold_stl_path)
    jig, jig_size, _, _  = prep_jig(mold_bounds, slice_thickness, jig_translate_y, x_wall, y_wall, z_wall, pre_knife_space, post_knife_space)
    slicer = prep_slicer(jig_size, slice_thickness, jig_translate_y, slicing_z, x_wall, y_wall)
    final_jig = assemble_jig(jig, slicer, slice_thickness, jig_size, z_wall, jig_translate_y, mold_stl, laterality)
    final_jig.exportStl(jig_stl_path)


if __name__ == "__main__":

    # def these_args():

    # parser = argparse.ArgumentParser(description="Process NIfTI file and output STLs.")
    # parser.add_argument(
    #     '-i','--nifti',
    #     required=True,
    #     type=str,
    #     help="Path to the input NIfTI file",
    # )
    # parser.add_argument(
    #     '-m','--mold_stl_path',
    #     nargs="?",
    #     type=str,
    #     default= None,
    #     help="Path to the output STL file (optional)",
    # )
    # parser.add_argument(
    #     '-o','--jig_stl_path',
    #     required=True,
    #     type=str,
    #     help="Path to the output STL file",
    # )
    # parser.add_argument(
    #     '-p','--profile',
    #     choices=profiles.keys(), #load from file
    #     default='prostate',
    #     help="Profile to use for processesing. Defaults to prostate"
    # )
    
    # args = parser.parse_args()

    # if args.profile in profiles:
    #     profile = profiles[args.profile]
    # else:
    #     profile_key = args.profile.split("/")[-1].split(".")[0]
    #     profile = load_profile(args.profile, profile_key)

    #brain test args
    nifti_test = '/Volumes/Siren/Brain_data/1.PatientDirectory/263/MRI/Processed/20250529_Mr_Brain_Rcbv_WO+W_Cont/brainmask.nii.gz'
    mold_test = '/Volumes/Siren/Brain_data/1.PatientDirectory/263/MRI/Processed/20250529_Mr_Brain_Rcbv_WO+W_Cont/test_brain.stl'
    jig_test = '/Volumes/Siren/Brain_data/1.PatientDirectory/263/MRI/Processed/20250529_Mr_Brain_Rcbv_WO+W_Cont/test_jig.stl'
    prof = profiles['brain']
    #profile test
    # prof = load_profile('/Users/fkyereme/Code_Blue/projects/AutoJigging/profiles/brain_profile.yaml', 'brain')
    #252 L 3.5
    # #254 R 4
    # GR 256 R 4
    # SE 257 L 3.99
    # JR 258 L 3.96
    #prostate test args
    # nifti_test = '/Volumes/Siren/Prostate_data/648/MRI/Processed/prostate_mask.nii.gz'
    # mold_test = '/Volumes/Siren/Prostate_data/648/MRI/Processed/mold.stl'
    # jig_test = '/Volumes/Siren/Prostate_data/648/MRI/Processed/jig.stl'
    # prof = profiles['prostate']

    main(nifti_test, mold_test, jig_test, prof)


    # TODO:
    #DONE - flush out profile/add profile file support 
    #DONE - add tumor laterality 
    # - finish slice by slice logic
    #DONE - flush out args | file profiles, default profile, 
    # - ???
    # - lint
    # - finish docstrings
    # - fix blob jig
    # - visualization
    # - fix slicing logic
