#Author-Nico Schlüter
#Description-An Addin for reconstructing surfaces from meshes

import adsk.core, adsk.fusion, adsk.cam, traceback
import time
import inspect
import os
import sys


# Enables experimental features
DEV = True



# ============================== Imports NumPy & SciPy  ==============================


script_path = os.path.abspath(inspect.getfile(inspect.currentframe()))
script_name = os.path.splitext(os.path.basename(script_path))[0]
script_dir = os.path.dirname(script_path)

sys.path.append(script_dir + "/Modules")

try:
    import numpy as np
    import scipy
    from scipy import optimize
    import math
finally:
    del sys.path[-1]



_handlers = []











# ============================== Addin Start & Stop  ==============================
# Responsible for createing and cleaning up commands & UI stuff


def run(context):
    try:
        app = adsk.core.Application.get()
        ui = app.userInterface

        '''
        defs = ui.commandDefinitions
        for i in range(defs.count):
            print(defs.item(i).id)

        '''
        
        commandDefinitions = ui.commandDefinitions

        tabReverse = ui.allToolbarTabs.itemById("tabReverse")
        if tabReverse:
            tabReverse.deleteMe()

        tabReverse = ui.workspaces.itemById("FusionSolidEnvironment").toolbarTabs.add("tabReverse", "Reverse Engineer")

        # Setup
        panelSetup = ui.allToolbarPanels.itemById("panelReverseSetup")
        if panelSetup:
            panelSetup.deleteMe()
        panelSetup = tabReverse.toolbarPanels.add("panelReverseSetup", "Setup")

        # Create
        panelCreate = ui.allToolbarPanels.itemById("panelReverseCreate")
        if panelCreate:
            panelCreate.deleteMe()
        panelCreate = tabReverse.toolbarPanels.add("panelReverseCreate", "Create")

        # Insert
        panelInsert = ui.allToolbarPanels.itemById("panelReverseInsert")
        if panelInsert:
            panelInsert.deleteMe()
        panelInsert = tabReverse.toolbarPanels.add("panelReverseInsert", "Insert")

        
        # Place Command
        cmdDef = commandDefinitions.itemById("commandReversePlace")
        if cmdDef:
            cmdDef.deleteMe()
        if(DEV):
            cmdDef = commandDefinitions.addButtonDefinition("commandReversePlace", "[WIP] Place",
                                                            "Places Mesh on XY Plane", 'Resources/Placeholder')

        onCommandCreated = CommandPlaceCreatedHandler()
        cmdDef.commandCreated.add(onCommandCreated)
        _handlers.append(onCommandCreated)

        panelSetup.controls.addCommand(cmdDef).isPromoted = True


        # Cylinder Command
        cmdDef = commandDefinitions.itemById("commandReverseCylinder")
        if cmdDef:
            cmdDef.deleteMe()
        if(DEV):
            cmdDef = commandDefinitions.addButtonDefinition("commandReverseCylinder", "Cylinder",
                                                            "Reconstructs a cylindrical face", 'Resources/Placeholder')

        onCommandCreated = CommandCylinderCreatedHandler()
        cmdDef.commandCreated.add(onCommandCreated)
        _handlers.append(onCommandCreated)

        panelCreate.controls.addCommand(cmdDef).isPromoted = True

        panelInsert.controls.addCommand(ui.commandDefinitions.itemById("InsertMeshCommand")).isPromoted = True



    except:
        print(traceback.format_exc())


def stop(context):
    try:
        app = adsk.core.Application.get()
        ui = app.userInterface
    except:
        print(traceback.format_exc())










# ============================== Place command ==============================


# Fires when the CommandDefinition gets executed.
# Responsible for adding commandInputs to the command &
# registering the other command handlers.
class CommandPlaceCreatedHandler(adsk.core.CommandCreatedEventHandler):
    def __init__(self):
        super().__init__()
    def notify(self, args):
        try:
            # Get the command that was created.
            cmd = args.command

            #import .commands.VertexSelectionInput
            vsi = VertexSelectionInput(args)

            # Registers the CommandDestryHandler
            onExecute = CommandPlaceExecuteHandler(vsi)
            cmd.execute.add(onExecute)
            _handlers.append(onExecute)  

        except:
            print(traceback.format_exc())


#Fires when the User executes the Command
#Responsible for doing the changes to the document
class CommandPlaceExecuteHandler(adsk.core.CommandEventHandler):
    def __init__(self, vsi):
        self.vsi = vsi
        super().__init__()
    def notify(self, args):
        try:
            if self.vsi.selected_points:
                
                # Gets actual coordinates from selected indexes
                crds = self.vsi.mesh_points[ list(self.vsi.selected_points) ]

                # Fits a plane to the set of coordinates
                # result contains metadata res.x is actual result
                res = fitPlaneToPoints( crds , seed=np.concatenate((crds[0], np.cross(crds[0]-crds[-1], crds[1]-crds[-1]) )))

                print(res)

                app = adsk.core.Application.get()
                des = app.activeProduct
                root = des.rootComponent
                bodies = root.bRepBodies

                des.designType = 0
                
                bodies = adsk.core.ObjectCollection.create()
                bodies.add(self.vsi.selectionInput.selection(0).entity)

                transform = adsk.core.Matrix3D.create()
                transform.setToRotateTo(adsk.core.Vector3D.create(0,0,1), adsk.core.Vector3D.create(res.x[3], res.x[4], res.x[5]) )

                # Create a move feature
                moveFeats = root.features.moveFeatures
                moveFeatureInput = moveFeats.createInput(bodies, transform)
                moveFeats.add(moveFeatureInput)
            
        except:
            print(traceback.format_exc())










# ============================== Cylinder command ==============================


# Fires when the CommandDefinition gets executed.
# Responsible for adding commandInputs to the command &
# registering the other command handlers.
class CommandCylinderCreatedHandler(adsk.core.CommandCreatedEventHandler):
    def __init__(self):
        super().__init__()
    def notify(self, args):
        try:
            # Get the command that was created.
            cmd = args.command

            #import .commands.VertexSelectionInput
            vsi = VertexSelectionInput(args)

            # Registers the CommandDestryHandler
            onExecute = CommandCylinderExecuteHandler(vsi)
            cmd.execute.add(onExecute)
            _handlers.append(onExecute)  

        except:
            print(traceback.format_exc())


#Fires when the User executes the Command
#Responsible for doing the changes to the document
class CommandCylinderExecuteHandler(adsk.core.CommandEventHandler):
    def __init__(self, vsi):
        self.vsi = vsi
        super().__init__()
    def notify(self, args):
        try:
            if self.vsi.selected_points:
                
                # Gets actual coordinates from selected indexes
                crds = self.vsi.mesh_points[ list(self.vsi.selected_points) ]

                # Fits a plane to the set of coordinates
                # result contains metadata res.x is actual result
                # avgCrds = np.average(crds, axis=0)
                # res = fitCylinderToPonts(crds, seed = np.array([avgCrds[0], avgCrds[1], avgCrds[2], 1, 1, 1, 2.5]))
                res = fitCylinderToPoints(crds)

                if(DEV):
                    print(res)
                
                # Bounds of cylinder as scalar
                bounds = cylinderBounds(crds, np.array(res.x[0:3]), np.array(res.x[3:6]))

                #Origin Vector
                o = np.array(res.x[0:3])

                #Normalized Normal Vector
                n = np.array(res.x[3:6]) / np.linalg.norm(np.array(res.x[3:6]))

                # Start and End Points
                p1 = o + n * bounds[0]
                p2 = o + n * bounds[1]

                app = adsk.core.Application.get()
                des = app.activeProduct
                root = des.rootComponent
                bodies = root.bRepBodies

                des.designType = 0
                
                tbm = adsk.fusion.TemporaryBRepManager.get()

                tempBRepBodies = []

                cylinder = tbm.createCylinderOrCone(adsk.core.Point3D.create(p2[0], p2[1], p2[2]),
                                            res.x[6],
                                            adsk.core.Point3D.create(p1[0], p1[1], p1[2]),
                                            res.x[6])
                tempBRepBodies.append(cylinder)

                for b in tempBRepBodies:
                    bodies.add(b)
                
                
            
        except:
            print(traceback.format_exc())




                
                
                




# p = Point a,b = Line
def distPtToLine(p, a, b):
    return np.linalg.norm( np.cross(b-a, a-p), axis=1) / np.linalg.norm(b-a)


# p = Point o = Plane Origin n = Plane normal
def distPtToPlane(p, o, n):
    return np.dot( (p-o), n ) / np.linalg.norm(n)


def isPointInvisiblePerspecive(points, cameraPos, tris):
    return [np.any( doesLineIntersectTriangle(np.repeat(np.array([[p, cameraPos]]), len(tris), axis=0) , tris) ) for p in points]


#Tales Array of lines (-1,2,3) and array of triangles and checks for intersection line by line
def doesLineIntersectTriangle(line, triangle):
    
    a = np.sign( spv(line[:,0], line[:,1], triangle[:,0], triangle[:,1]) )
    b = np.sign( spv(line[:,0], line[:,1], triangle[:,1], triangle[:,2]) )
    c = np.sign( spv(line[:,0], line[:,1], triangle[:,2], triangle[:,0]) )

    return np.logical_and(np.not_equal(np.sign(spv(line[:,0], triangle[:,0], triangle[:,1], triangle[:,2]) ), np.sign(spv(line[:,1], triangle[:,0], triangle[:,1], triangle[:,2]) )), np.logical_and(np.equal(a, b), np.equal(b, c)))
            

#Signed volume of a Parallelepiped, equal to 6 times the signed volume of a tetrahedron
def spv(a, b, c, d):
    #Einsum seems to be used for row-wise dot products
    return np.einsum('ij,ij->i', d-a, np.cross(b-a, c-a))


#Returns point and vector [px, py, pz, vx, vy, vz]
def fitLineToPoints(pts, seed=np.array([1,1,1,1,1,1])):
    return scipy.optimize.minimize(lambda x: np.sum(distPtToLine(pts, np.array([x[0], x[1], x[2]]), np.array([x[0] + x[3], x[1] + x[4], x[2]+ x[5]]))**2), seed , method = 'Powell')
    

def fitPlaneToPoints(pts, seed=np.array([1,1,1,1,1,1])):
    return scipy.optimize.minimize(lambda x: np.sum(

        distPtToPlane(pts, np.array([x[0], x[1], x[2]]), np.array([x[3], x[4], x[5]]))**2
        
        ), seed , method = 'Powell')

#Returns point, vector and radius [px, py, pz, vx, vy, vz, r]
def fitCylinderToPonts(pts, seed=np.array([1,1,1,1,1,1,1])):
    return scipy.optimize.minimize(lambda x: np.sum((distPtToLine(pts, np.array([x[0], x[1], x[2]]), np.array([x[0] + x[3], x[1] + x[4], x[2]+ x[5]]))-x[6])**2) , seed , method = "Powell")
  

# Generates a 3D unit vector given two spherical angles
#  Inputs:
#    ang: an array containing a (polar) angle from the Z axis and an (azimuth) angle in the X-Y plane from the X axis, in radians
#  Outputs:
#    a 3D unit vector defined by the input angles (input = [0,0] -> output = [0,0,1])
def sphericalToDirection(ang):
    """Construct a unit vector from spherical coordinates."""
    return [math.cos(ang[0]) * math.sin(ang[1]), math.sin(ang[0]) * math.sin(ang[1]), math.cos(ang[1])]

# Simple solution definiton
class Soln():
    pass


# Solves for the parameters of an infinite cylinder given a set of 3D cartesian points
#  Inputs:
#    pts: a Nx3 array of points on the cylinder, ideally well-distributed radially with some axial variation
#  Outputs a solution object containing members:
#    x: estimated cylinder origin, axis, and radius parameters [ px, py, pz, ax, ay, az, r ]
#    fun: non-dimensional residual of the fit to be used as a quality metric
#  Method:
#  - The general approach is a hybrid search where several parameters are handled using iterative optimization and the rest are directly solved for
#  - The outer search is performed over possible cylinder orientations (represented by two angles)
#    - A huge advantage is that these parameters are bounded, making search space coverage tractable despite the multiple local minima
#    - Reducing the iterative search space to two parameters dramatically helps iteration time as well
#    - To help ensure the global minimum is found, a coarse grid method is used over the bounded parameters
#    - A gradient method is used to refine the found coarse global minimum
#  - For each orientation, a direct (ie, non-iterative) LSQ solution is used for the other 3 paremeters
#    - This can be visualized as checking how "circular" the set of points appears when looking along the expected axis of the cylinder
#  - Note that no initial guess is needed since whole orientation is grid-searched and the other parameters are found directly without iteration
def fitCylinderToPoints(pts):
    """Solve for 3D parameters of an infinite cylinder given pts that lie on the cylinder surface."""
    # Create search grid for orientation angles
    # (note, may need to increase number of grid points in cases of poorly defined point sets)
    ranges = (slice(0, 2*math.pi, math.pi/4), slice(0, math.pi, math.pi/4))

    # Perform brute force grid search for approximate global minimum location followed by iterative fine convergence
    # (note, this can probably be replaced with a different global search method, ie simulated annealing, 
    #   but the grid search should work reasonably well given that the search space is bounded)
    sol = scipy.optimize.brute(lambda x: fitCylinderOnAxis(pts, sphericalToDirection(x))[1], ranges, finish=scipy.optimize.fmin)

    # Calculate desired parameters from the found cylinder orientation
    axis = sphericalToDirection(sol)    
    result = fitCylinderOnAxis(pts, axis)
    circle_params = result[0]    
    resid = result[1]
    cylinder_params = [circle_params[0], circle_params[1], circle_params[2], axis[0], axis[1], axis[2], circle_params[3]]

    # Mimic return elements in scipy.optimize.minimize functions
    res = Soln()
    setattr(res, 'x', cylinder_params)
    setattr(res, 'fun', resid)
    return res


# Solves for some parameters of an infinite cylinder given a set of 3D cartesian points and an axis for the cylinder
#  Inputs:
#    pts: a Nx3 array of points on the cylinder, ideally well-distributed radially with some axial variation
#    axis: a vector containing the central axis direction of the cylinder
#  Outputs a tuple containing:
#    pest: estimated cylinder origin and radius parameters [ px, py, pz, r ]
#    resid: non-dimensional residual of the fit to be used as a quality metric
#  Method:
#    - Generates a set of orthonormal basis vectors based on the input cylinder axis
#    - Forms a direction cosine matrix from the basis vectors and rotates the points into the cylinder frame
#    - Collapses the points along the cylinder axis and runs 2D circle estimation to get the lateral offset and radius
#    - Along-axis offset is meaningless for an infinite cylinder, so the mean of the input points in that direction is arbitrarily used
#    - Maps the offsets back to the original coordinate system
#    - Note that the returned residual is the 2D circular fit metric
def fitCylinderOnAxis(pts, axis=np.array([0,0,1])):
    """Solve for offset and radius parameters of a cylinder given its central axis and pts that lie on the cylinder surface."""
    # Create basis vectors for transformed coordinate system
    w = axis
    u = np.cross(w,np.array([w[1],w[2],w[0]]))
    u = u / np.linalg.norm(u)
    v = np.cross(w,u)
    
    # Construct DCM and rotate points into cylinder frame
    C = np.array([u, v, w]).transpose()
    pts3d = np.array(pts)
    N = len(pts3d)
    pts3drot = pts3d.dot(C)

    # Convert to 2D circle problem and solve for fit
    pts2d = pts3drot[0:N,0:2]
    result = fitCircle2D(pts2d)
    x2d = result[0]
    resid = result[1]

    # Compute mean axial offset and map back into original frame
    # (note, may better to use midpoint of min/max rather than the mean)
    x3d = C.dot(np.array([x2d[0], x2d[1], np.sum(pts3drot[0:N,2])/N]))
    pest = np.append(x3d, x2d[2])    
    return (pest, resid)


# Solves for the 2D parameters of a circle given a set of 2D (x,y) points
#  Inputs:
#    pts: a Nx3 array of points on the cylinder, 3 points minimum, ideally well-distributed
#  Outputs a tuple containing:
#    pest: estimated 2D parameters [ px, py, r ]
#    resid: non-dimensional residual of the fit to be used as a quality metric
#  Method:
#    - Reparameterizes the problem into a nondimensional linear form through a change of variables
#    - 2D parameters are solved for directly using linear least squares rather than an iterative method
def fitCircle2D(pts):
    """Solve for parameters of a 2D circle given pts that lie on the circle perimeter."""
    N = len(pts)

    # build LSQ model matrix and solve for non-dimensional parameters
    ym = -np.ones((N, 1))
    H = [[xi**2 + yi**2, xi, yi] for (xi,yi) in pts]
    result = np.linalg.lstsq(H, ym, rcond=None)
    xe = result[0].flatten()

    # extract desired circle parameters
    pest = [0,0,0]
    # origin
    pest[0] = -xe[1] / 2 / xe[0]
    pest[1] = -xe[2] / 2 / xe[0]
    # radius
    rsq = np.sum((pts[0:N,0] - pest[0])**2 + (pts[0:N,1] - pest[1])**2)/N
    pest[2] = math.sqrt(rsq)

    # return parameters and fit residual for optimization feedback
    resid = result[1][0]
    return (pest, resid)



#Returns point, radius [px, py, pz, r]
def fitSphereToPoints(pts, seed=np.array([1,1,1,1])):
    return scipy.optimize.minimize(lambda x: np.sum((np.linalg.norm(pts-np.array([x[0], x[1], x[2]]), axis=1)-x[3])**2) , seed , method = "Powell")
  

def cylinderBounds(pts, o, n):
    d = distPtToPlane(pts, o, n)
    return np.array([min(d), max(d)])


def GetRootMatrix(comp):
    comp = adsk.fusion.Component.cast(comp)
    des = adsk.fusion.Design.cast(comp.parentDesign)
    root = des.rootComponent

    mat = adsk.core.Matrix3D.create()
  
    if comp == root:
        return mat

    occs = root.allOccurrencesByComponent(comp)
    if len(occs) < 1:
        return mat

    occ = occs[0]
    occ_names = occ.fullPathName.split('+')
    occs = [root.allOccurrences.itemByName(name) 
                for name in occ_names]
    mat3ds = [occ.transform for occ in occs]
    mat3ds.reverse() #important!!
    for mat3d in mat3ds:
        mat.transformBy(mat3d)

    return mat










# ============================== Selection Input for Vertexes ==============================# 


class VertexSelectionInput:
    handlers = []
    mesh_points = None
    mesh_tris = None
    selected_points = set()


    def __init__(self, args):
        print("init")
        self.selected_points = set()
        
        cmd = args.command
        inputs = cmd.commandInputs

        onClick = self.VertexSelectionClickEventHandler(self)
        cmd.mouseClick.add(onClick)
        self.handlers.append(onClick)

        onInputChanged = self.VertexSelectionInputChangedEventHandler(self)
        cmd.inputChanged.add(onInputChanged)
        self.handlers.append(onInputChanged)

        onExecutePreview = self.VertexSelectionInputExecutePreviewHandler(self)
        cmd.executePreview.add(onExecutePreview)
        self.handlers.append(onExecutePreview)

        self.selectionInput = inputs.addSelectionInput('vertexSelectionMesh', 'Mesh', 'Select Mesh')
        self.selectionInput.addSelectionFilter('MeshBodies')
        self.selectionInput.setSelectionLimits(0, 1)

        self.floatSpinner = inputs.addFloatSpinnerCommandInput('vertexSelectionRadius', 'Selection Radius', '', 0, 10000, 5, 5)

        self.boolValue = inputs.addBoolValueInput('vertexSelectionThrough', 'Select Through', True)
        self.boolValue.isEnabled = False
        self.boolValue.value = True
        self.boolValue.tooltip = "Placeholder, Disabled due to performance issues"


    class VertexSelectionClickEventHandler(adsk.core.MouseEventHandler):
        def __init__(self, parent):
            super().__init__()
            self.parent = parent

        def notify(self, args):
            try:    
                if self.parent.mesh_points is not None:
                    isPerspective = args.viewport.camera.cameraType == 1

                    clickPos3D = np.array(args.viewport.viewToModelSpace(args.viewportPosition).asArray())
                    cameraPos = np.array(args.viewport.camera.eye.asArray())

                    app = adsk.core.Application.get()
                    design = app.activeProduct
                    rootComp = design.rootComponent

                    d = distPtToLine(self.parent.mesh_points, clickPos3D, cameraPos)
                    '''
                    print("start")
                    t = time.time()

                    v = isPointInvisiblePerspecive(self.parent.mesh_points,cameraPos, self.parent.mesh_tris)

                    print()
                    print("Time:")
                    print(time.time()-t)
                    print()
                    '''
                    for i, j in enumerate(self.parent.mesh_points):
                        if d[i] < self.parent.floatSpinner.value/10:
                            self.parent.selected_points.add(i)

                    #Update UI
                    self.parent.boolValue.value = False
                    self.parent.boolValue.value = True
            except:
                print(traceback.format_exc())

        
    class VertexSelectionInputChangedEventHandler(adsk.core.InputChangedEventHandler):
        def __init__(self, parent):
            super().__init__()
            self.parent = parent

        def notify(self, args):
            try:
                inputChanged = args.input
                if inputChanged.id == "vertexSelectionMesh":
                    if inputChanged.selectionCount == 1:
                        inputChanged.hasFocus = False
                        #TO-DO use nodeCoordinatesAsDouble (returns 1D list that needs to be seperated but is probably faster)
                        nc = inputChanged.selection(0).entity.displayMesh.nodeCoordinates

                        mat = GetRootMatrix(inputChanged.selection(0).entity.parentComponent)

                        for i in nc:
                            i.transformBy(mat)
                        self.parent.mesh_points = np.array( [ [i.x, i.y, i.z] for i in nc] )

                        self.parent.mesh_tris = self.parent.mesh_points[np.array(inputChanged.selection(0).entity.displayMesh.nodeIndices)].reshape(-1,3,3)

            except:
                print(traceback.format_exc())


    class VertexSelectionInputExecutePreviewHandler(adsk.core.CommandEventHandler):
        def __init__(self, parent):
            super().__init__()
            self.parent = parent

        def notify(self, args):
            try:
                if self.parent.mesh_points is not None:
                    pts = self.parent.mesh_points[list(self.parent.selected_points)]

                    cgg = adsk.core.Application.get().activeProduct.rootComponent.customGraphicsGroups.add()

                    print(np.asarray(pts.reshape(-1), dtype='d'))

                    coords = adsk.fusion.CustomGraphicsCoordinates.create(np.asarray(pts.reshape(-1), dtype='d'))
                    
                    cgg.addPointSet(coords, range(len(pts)), 0, 'TestPoint.png')
            except:
                print(traceback.format_exc())


  