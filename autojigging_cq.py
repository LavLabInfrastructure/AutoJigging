import nibabel as nib
import numpy as np
import vtk
import SimpleITK as sitk
import cadquery as cq
import argparse
from typing import cast
from stl.mesh import Mesh

JIG_TRANSLATEY = 15
SLICINGZ = 1 #constant for now, args/custom later
X_WALL = 5
Y_WALL = 3
Z_WALL = 3

def find_slice_thickness(nifti_path: str):
    loaded_nifti = nib.load(nifti_path)
    slice_thickness = np.round(loaded_nifti.header["pixdim"][3],1)

    return slice_thickness

def create_reader(nifti_path: str):
    reader = vtk.vtkNIFTIImageReader()
    reader.SetFileName(nifti_path)
    reader.Update()

    return reader

def create_surface_extractor(reader_connect, label=1):
    surf = vtk.vtkDiscreteMarchingCubes()
    surf.SetInputConnection(reader_connect)
    surf.SetValue(0, int(label))
    surf.Update()

    return surf

def create_smoother(surfer_connect, iterations=10000):
    smoother = vtk.vtkSmoothPolyDataFilter()
    smoother.SetInputConnection(surfer_connect)
    smoother.SetNumberOfIterations(iterations)
    smoother.Update()

    return smoother

def create_decimator(smoother_connect, target_reduction=.93):
    decimate = vtk.vtkDecimatePro()
    decimate.SetInputConnection(smoother_connect)
    decimate.SetTargetReduction(target_reduction)
    decimate.Update()

    return decimate

def create_scaler(decimate_connect, scaling_factors = (1.02, 1.02, 1.02)): 
    set_scaler = vtk.vtkTransform()
    set_scaler.Scale(scaling_factors)

    scaler = vtk.vtkTransformPolyDataFilter()
    scaler.SetTransform(set_scaler)
    scaler.SetInputConnection(decimate_connect)
    scaler.Update()

    return scaler

def find_bounds(poly_data):
    final_obj_bounds = poly_data.GetBounds()

    return final_obj_bounds

def find_translation(final_obj_bounds):
    center_x = (final_obj_bounds[0] + final_obj_bounds[1]) / 2
    center_y = (final_obj_bounds[2] + final_obj_bounds[3]) / 2
    center_z = (final_obj_bounds[4] + final_obj_bounds[5]) / 2

    translation = (-center_x, -center_y, -center_z)

    return translation

def create_translator(scaler_connect, translation):
    translation_transform = vtk.vtkTransform()
    translation_transform.Translate(translation)

    translation_filter = vtk.vtkTransformPolyDataFilter()
    translation_filter.SetTransform(translation_transform)
    translation_filter.SetInputConnection(scaler_connect)
    translation_filter.Update()

    return translation_filter

def prep_object(nifti_path: str) -> vtk.vtkPolyData:
    reader = create_reader(nifti_path)
    surf = create_surface_extractor(reader.GetOutputPort())
    smoother = create_smoother(surf.GetOutputPort())
    decimate = create_decimator(smoother.GetOutputPort())
    scaler = create_scaler(decimate.GetOutputPort())

    poly_data = scaler.GetOutput()

    final_obj_bounds = find_bounds(poly_data)

    translation = find_translation(final_obj_bounds)
    translation_filter = create_translator(scaler.GetOutputPort(), translation)

    poly_data = translation_filter.GetOutput()
    
    return poly_data

def write_stl(poly_data, output_stl_path):
    writer = vtk.vtkSTLWriter()
    writer.SetInputData(poly_data)
    writer.SetFileTypeToASCII()
    writer.SetFileName(output_stl_path)
    writer.Write()
    return output_stl_path

def toVectors(points):
    if isinstance(next(iter(points)), cq.Vector):
        return cast(tuple[cq.Vector], list(points))
    
    return cast(tuple[cq.Vector], tuple(cq.Vector(*p) for p in points))

def makePolyhedron(points, faces) -> cq.Solid:
    vectors = np.array(toVectors(points))

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

def find_jig_bounds(nifti_path):
    poly_data = prep_object(nifti_path)
    final_obj_bounds = find_bounds(poly_data)
    jig_modifiers = find_jig_modifiers(nifti_path)

    jig_bounds = (
        final_obj_bounds[0] + jig_modifiers[0],
        final_obj_bounds[1] + jig_modifiers[1],
        final_obj_bounds[2] + jig_modifiers[2],
        final_obj_bounds[3] + jig_modifiers[3],
        final_obj_bounds[4] + jig_modifiers[4],
        final_obj_bounds[5] + jig_modifiers[5]
    )

    return jig_bounds

def find_jig_size(nifti_path):
    jig_bounds = find_jig_bounds(nifti_path)

    jigsizex = jig_bounds[1] - jig_bounds[0]
    jigsizey = jig_bounds[3] - jig_bounds[2]
    jigsizez = jig_bounds[5] - jig_bounds[4]

    jig_size = (jigsizex, jigsizey, jigsizez)

    return jig_size

def find_jig_modifiers(nifti_path):
    slice_thickness = find_slice_thickness(nifti_path)

    pre_knife_slot = 37
    post_knife_slot = 3

    jig_modifiers = (
        -X_WALL,
         X_WALL,
        -(Y_WALL+post_knife_slot),
         Y_WALL+pre_knife_slot,
         -((slice_thickness) +Z_WALL),
         ((slice_thickness) +Z_WALL)
    )

    return jig_modifiers
    
def process_jig(nifti_path):

    jig_size = find_jig_size(nifti_path)

    result = (
        cq.Workplane("XY")
        .box(*jig_size)
        .translate((0,JIG_TRANSLATEY,0))
        )

    cad_jig = result
    return cad_jig

def process_slicer(nifti_path):
    jig_size = find_jig_size(nifti_path)
    slicingx = jig_size[0] + 2*X_WALL
    slicingy = jig_size[1] - 2*Y_WALL

    result = (
        cq.Workplane("XY")
        .box(slicingx, slicingy, SLICINGZ)
        .translate((0, JIG_TRANSLATEY, -((jig_size[2] / 2) - 3)))
        )
    
    cad_slicer = result
    return cad_slicer

def import_stl(stl_path:str):
    vectors = Mesh.from_file(stl_path).vectors
    points = tuple(map(tuple, vectors.reshape((vectors.shape[0] * vectors.shape[1], 3))))
    faces = [(i, i + 1, i + 2) for i in range(0, len(points), 3)]
    return makePolyhedron(points, faces)

def process_object(nifti_path, output_stl_path):
    poly_data = prep_object(nifti_path)
    stl_path = write_stl(poly_data, output_stl_path)
    final_obj_bounds = find_bounds(poly_data)
    final_obj_y = final_obj_bounds[3] - final_obj_bounds[2]
    stl_object = import_stl(stl_path)
    return stl_object, final_obj_y

def assemble_jig(nifti_path, output_stl_path):
    jig = process_jig(nifti_path)
    slicer = process_slicer(nifti_path)
    object, final_obj_y = process_object(nifti_path, output_stl_path)

    slice_thickness = find_slice_thickness(nifti_path)
    jig_size = find_jig_size(nifti_path)
    z_gap = slice_thickness - SLICINGZ
    
    assembly = cq.Workplane("XY").add(jig)

    start_z = int(np.round(find_jig_modifiers(nifti_path)[4], 2) * 100)
    end_z = int(np.round(jig_size[2] - find_jig_modifiers(nifti_path)[5], 2) * 100)
    step_z = int(np.round(z_gap, 2) * 100)

    for slots in range(start_z, end_z, step_z):
        if slots > 0:
            slicer = slicer.translate((0,0, slots/100))
            assembly = assembly.cut(slicer)

    start_y = 0
    end_y = int(np.round((jig_size[1]+final_obj_y), 2) * 100)

    for i in range(start_y, end_y, 100):
        object = object.translate((0,1,0))
        assembly = assembly.cut(object)
       
    return assembly

def main(nifti, mold_stl_path, jig_stl_path):
    nifti = '/Volumes/Siren/Prostate_data/573/MRI/Processed/prostate_mask.nii.gz'
    polydata = prep_object(nifti)
    write_stl(polydata, mold_stl_path)
    resulting_assembly = assemble_jig(nifti, '/tmp/stl.stl')
    resulting_assembly.val().exportStl(jig_stl_path)

if __name__ == '__main__':    
    default_nifti = '/Volumes/Siren/Prostate_data/573/MRI/Processed/prostate_mask.nii.gz'
    default_mold_stl_path = '/Volumes/Siren/Prostate_data/573/MRI/Processed/autojigger_mold.stl'
    default_jig_stl_path = '/Volumes/Siren/Prostate_data/573/MRI/Processed/autojigger_slicer.stl'

    parser = argparse.ArgumentParser(description='Process NIfTI file and output STLs.')
    parser.add_argument('nifti', nargs='?', type=str, default=default_nifti, help='Path to the input NIfTI file')
    parser.add_argument('mold_stl_path', nargs='?', type=str, default=default_mold_stl_path, help='Path to the output STL file')
    parser.add_argument('jig_stl_path', nargs='?', type=str, default=default_jig_stl_path, help='Path to the output STL file')

    args = parser.parse_args() 
    main(args.nifti, args.mold_stl_path, args.jig_stl_path)
