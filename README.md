# Reverse

Add-In for Autodesk Fusion360.    
Reconstructs BRep Surfaces from mesh points.    
Generates geometry very close to the original source file, generally within 1e-6mm.

# Use

![img1](https://user-images.githubusercontent.com/30301307/90188610-1094d180-ddbc-11ea-89c6-c5ea7fb4536e.jpg)

1. Import the mesh to be reconstructed
2. Activate the command for the desired feature type. Select the mesh and select mesh points belonging to that feature.
  - All points within the selection radius around the click position will be selected, even hidden ones.
  - Hold shift to deselect points.
  - Adjust the selection radius to select just the points you want.
  - Points not belonging to the feature you are reconstructing significantly degrade the accuracy. Keep an eye on the selection count to make sure you are not accidentally selecting undesired points.  
  - Press OK to create the surface
3. Repeat step 2 for all features of the part
4. Use Boundary fill to turn the enclosed volume into a solid

# Supported Features
- Cylinders
- Planes

# Installation

* Download the Project as ZIP and extract it somewhere you can find again, but won't bother you. (or use git to clone it there)
* Open Fusion360 and press ADD-INS > Scripts and Add-ins
* Select the tab Add-Ins and click the green plus symbol next to "My Add-Ins"
* Navigate to the extracted Project folder and hit open
* The Add-in should now appear in the "My Add-Ins" list. Select it in the list. If desired check the "Run on Startup" checkbox and hit run.
* The Commands will appear as SURFACE > CREATE

# Changelog

## 1.0 Plane & Cylinder
- Added support for planes and cylinders
