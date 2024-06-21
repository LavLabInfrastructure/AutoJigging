import nibabel as nib
import numpy as np
import vtk
from solid import scad_render_to_file, cube, translate, difference
from solid.objects import import_stl
import argparse

def do_the_jig(nifti, output_stl_path):
    loaded_nifti = nib.load(nifti)
    slice_thickness = np.round(loaded_nifti.header["pixdim"][3],1)

    # read the file
    reader = vtk.vtkNIFTIImageReader()
    reader.SetFileName(nifti)
    reader.Update()

    # apply marching cube surface generation
    surf = vtk.vtkDiscreteMarchingCubes()
    surf.SetInputConnection(reader.GetOutputPort())
    surf.SetValue(0, 1) 
    surf.Update()

    #smoothing the mesh
    smoother = vtk.vtkSmoothPolyDataFilter()
    smoother.SetInputConnection(surf.GetOutputPort())
    smoother.SetNumberOfIterations(10000) #need test this more
    smoother.Update()

    #decimation
    decimate = vtk.vtkDecimatePro()
    decimate.SetInputConnection(smoother.GetOutputPort())
    decimate.SetTargetReduction(0.93)
    decimate.Update()

    #scaling the prostate
    scaling_factors = (1.02, 1.02, 1.02) 
    set_scaler = vtk.vtkTransform()
    set_scaler.Scale(scaling_factors)

    scaler = vtk.vtkTransformPolyDataFilter()
    scaler.SetTransform(set_scaler)
    scaler.SetInputConnection(decimate.GetOutputPort())
    scaler.Update()

    #determinging the boundaries of the prostate
    finalbounds = scaler.GetOutput().GetBounds()
    finalybound = finalbounds[3] - finalbounds[2]

    # calculate the center of the bounds
    center_x = (finalbounds[0] + finalbounds[1]) / 2
    center_y = (finalbounds[2] + finalbounds[3]) / 2
    center_z = (finalbounds[4] + finalbounds[5]) / 2

    # Compute the translation vector to move the center to (0, 0, 0)
    translation = (-center_x, -center_y, -center_z)

    # Create a vtkTransform to perform translation
    translation_transform = vtk.vtkTransform()
    translation_transform.Translate(translation)

    # Apply the translation to the scaled polydata
    translation_filter = vtk.vtkTransformPolyDataFilter()
    translation_filter.SetTransform(translation_transform)
    translation_filter.SetInputConnection(scaler.GetOutputPort())
    translation_filter.Update()

    # Get the translated polydata
    translated_poly_data = translation_filter.GetOutput()

    #finding and setting the boundaries of the slicing jig
    jig_modifiers = (-5, 5, -6, 40, -((slice_thickness) +3), ((slice_thickness) +3)) #may need to tweak modifiers

    jig_bounds = (
        finalbounds[0] + jig_modifiers[0],
        finalbounds[1] + jig_modifiers[1],
        finalbounds[2] + jig_modifiers[2],
        finalbounds[3] + jig_modifiers[3],
        finalbounds[4] + jig_modifiers[4],
        finalbounds[5] + jig_modifiers[5]
    )

    jigsizex = jig_bounds[1] - jig_bounds[0]
    jigsizey = jig_bounds[3] - jig_bounds[2]
    jigsizez = jig_bounds[5] - jig_bounds[4]

    jig_size = (jigsizex, jigsizey, jigsizez)

    # save the output
    writer = vtk.vtkSTLWriter()
    writer.SetInputData(translated_poly_data)
    writer.SetFileTypeToASCII()
    writer.SetFileName(output_stl_path)
    writer.Write()

    return output_stl_path, jig_size, jigsizex,jigsizey, jigsizez, finalybound,  slice_thickness 

def main(nifti, output_stl_path):
    output_stl_path, jig_size, jigsizex,jigsizey, jigsizez, finalybound,  slice_thickness = do_the_jig(nifti, output_stl_path)
    
    stl_object = import_stl(output_stl_path)
    stl_translatey = int((jigsizey - (finalybound))+(finalybound/2)) #some calculated value based off jigsizey 

    combined_stl = None

    for i in range(stl_translatey):
        translated_stl = translate([0, i * 1, 0])(stl_object)
        if combined_stl is None:
            combined_stl = translated_stl
        else:
            combined_stl += translated_stl

    jig_translatey = 15 #maybe jigsizey/2

    jig = cube(size=jig_size, center=True)
    translated_jig = translate([0, jig_translatey, 0])(jig) #putting the jig in the correct position

    slicingx = jigsizex+10 #~infinite
    slicingy = jigsizey-6 #-3 and +3 from edges
    slicingz = 1 # z + z_gap = slicethickness
    z_gap = slice_thickness - slicingz
    num_slots = int((np.round(jigsizez-6))/2)
    
    slicer = cube(size=[slicingx, slicingy, slicingz], center=True) 
    positioned_slicer = translate([0,jig_translatey,-((jigsizez/2)-3),])(slicer) 

    combined_slicer = None

    for i in range(num_slots):
        translated_slice = translate([0,0,i * z_gap])(positioned_slicer)
        if combined_slicer is None:
            combined_slicer = translated_slice
        else:
            combined_slicer += translated_slice

    final_jig = difference()(translated_jig, combined_slicer)
    final_jig -= final_jig, combined_stl

    scad_render_to_file(final_jig, '/Users/fkyereme/Documents/scripts/openscad/output.scad')

if __name__ == '__main__':
    default_nifti = '/Volumes/Siren/Prostate_data/1454/MRI/Processed/prostate_mask.nii.gz'
    default_output_stl_path = '/Volumes/Siren/Prostate_data/1454/MRI/Processed/autojigger_mold.stl'

    parser = argparse.ArgumentParser(description='Process NIfTI file and output STL.')
    parser.add_argument('nifti', nargs='?', type=str, default=default_nifti, help='Path to the input NIfTI file')
    parser.add_argument('output_stl_path', nargs='?', type=str, default=default_output_stl_path, help='Path to the output STL file')
    args = parser.parse_args()

    main(args.nifti, args.output_stl_path)
