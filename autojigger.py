import nibabel as nib
import numpy as np

import vtk

from solid import scad_render_to_file, cube, translate, color, difference, intersection
from solid.objects import import_stl


nifti = '/Volumes/Siren/Prostate_data/1454/MRI/Processed/prostate_mask.nii.gz'
output_stl_path = '/Volumes/Siren/Prostate_data/1454/MRI/Processed/prostate_mold_vtk_smooth_12_scaled_decimate_93_py.stl'

loaded_nifti = nib.load(nifti)

slice_thickness = np.round(loaded_nifti.header["pixdim"][3],1)

# can be done in a loop if you have multiple files to be processed, speed is guaranteed if GPU is used:)
filename_nii =  nifti
filename = filename_nii.split(".")[0]

label = 1

# read the file
reader = vtk.vtkNIFTIImageReader()
reader.SetFileName(filename_nii)
reader.Update()

vol = reader.GetOutput()

# apply marching cube surface generation
surf = vtk.vtkDiscreteMarchingCubes()
surf.SetInputConnection(reader.GetOutputPort())
surf.SetValue(0, int(label)) # use surf.GenerateValues function if more than one contour is available in the file
surf.Update()

#smoothing the mesh
smoother= vtk.vtkSmoothPolyDataFilter()
smoother.SetInputConnection(surf.GetOutputPort())

# increase this integer set number of iterations if smoother surface wanted
smoother.SetNumberOfIterations(10000)
smoother.Update()

decimate = vtk.vtkDecimatePro()
decimate.SetInputConnection(smoother.GetOutputPort())
decimate.SetTargetReduction(0.93)
decimate.Update()
decimated_poly_data = decimate.GetOutput()


scaling_factors = (1.02, 1.02, 1.02) # or (1/scale_x, 1/scale_y, 1/scale_z) but slightly big


set_scaler = vtk.vtkTransform()
set_scaler.Scale(scaling_factors)

scaler = vtk.vtkTransformPolyDataFilter()
scaler.SetTransform(set_scaler)
scaler.SetInputConnection(decimate.GetOutputPort())
scaler.Update()

scaled_poly_data = scaler.GetOutput()

finalbounds = scaler.GetOutput().GetBounds()
finalxbound = finalbounds[1] - finalbounds[0]
finalybound = finalbounds[3] - finalbounds[2]
finalzbound = finalbounds[5] - finalbounds[4]


# Calculate the center of the bounds
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


bounding_box_modifiers = (-5, 5, -6, 40, -((slice_thickness) +3), ((slice_thickness) +3))

box_bounds = (
    finalbounds[0] + bounding_box_modifiers[0],
    finalbounds[1] + bounding_box_modifiers[1],
    finalbounds[2] + bounding_box_modifiers[2],
    finalbounds[3] + bounding_box_modifiers[3],
    finalbounds[4] + bounding_box_modifiers[4],
    finalbounds[5] + bounding_box_modifiers[5]
)

boxsizex = box_bounds[1] - box_bounds[0]
boxsizey = box_bounds[3] - box_bounds[2]
boxsizez = box_bounds[5] - box_bounds[4]

box_size = (boxsizex, boxsizey, boxsizez)

# save the output
writer = vtk.vtkSTLWriter()
writer.SetInputData(translated_poly_data)
writer.SetFileTypeToASCII()

# file name need to be changed
# save as the .stl file, can be changed to other surface mesh file
writer.SetFileName(output_stl_path)
writer.Write()


def main():
       
    #stl proc - color, translate, combine
    stl_object = import_stl(output_stl_path)

    color_the_stl = color("red")
    colored_stl = color_the_stl(stl_object)

    stl_translatey = int((boxsizey - (finalybound))+(finalybound/2)) #some calculated value based off boxsizey 

    combined_stl = None

    #13 min for copy all then translate
    for i in range(stl_translatey):
        translated_stl = translate([0, i * 1, 0])(colored_stl)
        if combined_stl is None:
            combined_stl = translated_stl
        else:
            combined_stl += translated_stl

###############################################################################################################################################

    # box proc - color, translate
    box_translatey = 15 #maybe boxsizey/2

    bounding_cube = cube(size=box_size, center=True)
    
    color_the_box = color("blue", 0.6)
    colored_box = color_the_box(bounding_cube)

    
    translated_box = translate([0, box_translatey, 0])(colored_box) #putting the box in the correct position

 ###############################################################################################################################################
    #slicer proc - translate, combine
    slicingx = boxsizex+10 #~infinite
    slicingy = boxsizey-6 #-3 and +3 from edges
    slicingz = 1 # z + z_gap = slicethickness

    z_gap = slice_thickness - slicingz #
    num_slots = int((np.round(boxsizez-6))/2)

    slicer = cube(size=[slicingx, slicingy, slicingz], center=True)  #create slicer

    positioned_slicer = translate([0,box_translatey,-((boxsizez/2)-3),])(slicer) #align slicer with cube

    combined_slicer = None

    for i in range(num_slots):
        translated_slice = translate([0,0,i * z_gap])(positioned_slicer)
        if combined_slicer is None:
            combined_slicer = translated_slice
        else:
            combined_slicer += translated_slice


###############################################################################################################################################   
    #doing the jig
    combined_object = difference()(translated_box, combined_slicer)
    combined_object -= combined_object, combined_stl

    scad_render_to_file(combined_object, '/Users/fkyereme/Documents/scripts/openscad/output.scad')

if __name__ == '__main__':
    main()
