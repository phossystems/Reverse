#Author-Nico Schlüter
#Description-An Addin for reconstructing surfaces from meshes

import adsk.core, adsk.fusion, adsk.cam, traceback
import time
import inspect
import os
import sys


# ============================== Imports NumPy & SciPy  ==============================


script_path = os.path.abspath(inspect.getfile(inspect.currentframe()))
script_name = os.path.splitext(os.path.basename(script_path))[0]
script_dir = os.path.dirname(script_path)

if os.name == "posix":  
    sys.path.append(script_dir + "/ModulesMac")
else:
    sys.path.append(script_dir + "/ModulesWin")

try:
    import numpy as np
    import scipy
    from scipy import optimize
    from scipy.spatial import ConvexHull
    import math
finally:
    del sys.path[-1]


# Initial persistence Dict
pers = {
    'viExpansion': 0.1,
    'fsViRadius': 2
}


_handlers = []










# ============================== Addin Start & Stop  ==============================
# Responsible for createing and cleaning up commands & UI stuff


def run(context):
    try:
        
        app = adsk.core.Application.get()
        ui = app.userInterface
        
        commandDefinitions = ui.commandDefinitions
        #check the command exists or not
        cmdDefCylinder = commandDefinitions.itemById("commandReverseCylinder")
        cmdDefPlane = commandDefinitions.itemById("commandReversePlane")
        if not cmdDefCylinder:
            cmdDefCylinder = commandDefinitions.addButtonDefinition("commandReverseCylinder", "Cylinder", "Reconstructs a cylindrical face", 'Resources/Cylinder')
        if not cmdDefPlane:
            cmdDefPlane = commandDefinitions.addButtonDefinition("commandReversePlane", "Plane", "Reconstructs a planar face", 'Resources/Plane')
        #Adds the commandDefinition to the toolbar
        for panel in ["SurfaceCreatePanel"]:
            ui.allToolbarPanels.itemById(panel).controls.addCommand(cmdDefCylinder)
            ui.allToolbarPanels.itemById(panel).controls.addCommand(cmdDefPlane)
        
        onCommandCylinderCreated = CommandCylinderCreatedHandler()
        cmdDefCylinder.commandCreated.add(onCommandCylinderCreated)
        _handlers.append(onCommandCylinderCreated)

        onCommandPlaneCreated = CommandPlaneCreatedHandler()
        cmdDefPlane.commandCreated.add(onCommandPlaneCreated)
        _handlers.append(onCommandPlaneCreated)
    except:
        print(traceback.format_exc())


def stop(context):
    try:
        app = adsk.core.Application.get()
        ui = app.userInterface
                
        #Removes the commandDefinition from the toolbar
        for panel in ["SurfaceCreatePanel"]:
            p = ui.allToolbarPanels.itemById(panel).controls.itemById("commandReverseCylinder")
            if p:
                p.deleteMe()
            p = ui.allToolbarPanels.itemById(panel).controls.itemById("commandReversePlane")
            if p:
                p.deleteMe()
        
        #Deletes the commandDefinition
        ui.commandDefinitions.itemById("commandReverseCylinder").deleteMe()
        ui.commandDefinitions.itemById("commandReversePlane").deleteMe()            
    except:
        print(traceback.format_exc())









# ============================== Plane command ==============================


# Fires when the CommandDefinition gets executed.
# Responsible for adding commandInputs to the command &
# registering the other command handlers.
class CommandPlaneCreatedHandler(adsk.core.CommandCreatedEventHandler):
    def __init__(self):
        super().__init__()
    def notify(self, args):
        try:
            # Get the command that was created.
            cmd = args.command

            #import .commands.VertexSelectionInput
            vsi = VertexSelectionInput(args)

            # Registers the CommandExecuteHandler
            onExecute = CommandPlaneExecuteHandler(vsi)
            cmd.execute.add(onExecute)
            _handlers.append(onExecute)  

            # Registers the CommandDestryHandler
            onExecutePreview = CommandPlaneExecutePreviewHandler(vsi)
            cmd.executePreview.add(onExecutePreview)
            _handlers.append(onExecutePreview)

            # Registers the CommandInputChangedHandler          
            onInputChanged = CommandPlaneInputChangedHandler()
            cmd.inputChanged.add(onInputChanged)
            _handlers.append(onInputChanged) 

            global pers

            viExpansion = cmd.commandInputs.addValueInput("viExpansion", "Expansion", "mm", adsk.core.ValueInput.createByReal(pers["viExpansion"]))

        except:
            print(traceback.format_exc())


#Fires when the User executes the Command
#Responsible for doing the changes to the document
# Almost identical to the ExecutePreviewEventHandler, but that one also adds custom graphics.
# There are better ways to do this, this is sutpid
class CommandPlaneExecuteHandler(adsk.core.CommandEventHandler):
    def __init__(self, vsi):
        self.vsi = vsi
        super().__init__()
    def notify(self, args):
        try:
            if self.vsi.selected_points and len(self.vsi.selected_points) >= 3:
                # Gets actual coordinates from selected indexes
                crds = self.vsi.mesh_points[ list(self.vsi.selected_points) ]

                # Fits a plane to the set of coordinates
                # result contains metadata res.x is actual result
                avgCrds = np.average(crds, axis=0)
                res = fitPlaneToPoints2(crds)

                # Rejects bad results (by looking for extreme values)
                if(max(res.x) > 100000 or min(res.x) < -100000):
                    return
                
                #Normalized Normal Vector
                n = np.array(sphericalToDirection(res.x[:2]))

                #Origin Vector
                o = np.array(sphericalToDirection(res.x[:2])) * res.x[2]

                app = adsk.core.Application.get()
                des = app.activeProduct
                root = des.rootComponent
                bodies = root.bRepBodies

                # Creates a base feature when in parametric design mode
                if des.designType:
                    baseFeature = root.features.baseFeatures.add()
                    baseFeature.startEdit()
                else:
                    baseFeature = None
                
                # Gets the TemporaryBRepManager
                tbm = adsk.fusion.TemporaryBRepManager.get()

                # Computes the convex hull and turns it into Line3D objects
                hullLines = [adsk.core.Line3D.create(point3d(i[0]), point3d(i[1])) for i in getConvexHull(crds, res.x)]

                # Constructs a BRepWire inside a BRepBody from the hull lines
                wireBody, _ = tbm.createWireFromCurves(hullLines)
                
                # Computes the normal of the resulting surface. This is not n as the direction the resulting face is facing is essentially random
                tempSurface = tbm.createFaceFromPlanarWires([wireBody])
                _, faceNormal = tempSurface.faces[0].evaluator.getNormalAtParameter(adsk.core.Point2D.create(0,0))

                # offsets the BRepWire for expansion
                offsetWireBody = wireBody.wires[0].offsetPlanarWire(
                    faceNormal,
                    args.command.commandInputs.itemById("viExpansion").value,
                    2
                )

                # creates the actual face
                surface = tbm.createFaceFromPlanarWires([offsetWireBody])
                
                # Adds face and optionally finishes the baseFeature
                if(baseFeature):
                    realSurface = bodies.add(surface, baseFeature)
                    baseFeature.finishEdit()
                else:
                    realSurface = bodies.add(surface)
        except:
            print(traceback.format_exc())


#Fires when the User executes the Command
#Responsible for doing the changes to the document
class CommandPlaneExecutePreviewHandler(adsk.core.CommandEventHandler):
    def __init__(self, vsi):
        self.vsi = vsi
        super().__init__()
    def notify(self, args):
        try:
            if self.vsi.selected_points and len(self.vsi.selected_points) >= 3:
                
                # Gets actual coordinates from selected indexes
                crds = self.vsi.mesh_points[ list(self.vsi.selected_points) ]

                # Fits a plane to the set of coordinates
                # result contains metadata res.x is actual result
                avgCrds = np.average(crds, axis=0)
                res = fitPlaneToPoints2(crds)

                # Rejects bad results (by looking for extreme values)
                if(max(res.x) > 100000 or min(res.x) < -100000):
                    return
                
                #Normalized Normal Vector
                n = np.array(sphericalToDirection(res.x[:2]))

                #Origin Vector
                o = np.array(sphericalToDirection(res.x[:2])) * res.x[2]

                app = adsk.core.Application.get()
                des = app.activeProduct
                root = des.rootComponent
                bodies = root.bRepBodies

                # Creates a base feature when in parametric design mode
                if des.designType:
                    baseFeature = root.features.baseFeatures.add()
                    baseFeature.startEdit()
                else:
                    baseFeature = None
                
                # Gets the TemporaryBRepManager
                tbm = adsk.fusion.TemporaryBRepManager.get()

                # Computes the convex hull and turns it into Line3D objects
                hullLines = [adsk.core.Line3D.create(point3d(i[0]), point3d(i[1])) for i in getConvexHull(crds, res.x)]

                # Constructs a BRepWire inside a BRepBody from the hull lines
                wireBody, _ = tbm.createWireFromCurves(hullLines)
                
                
                offsetWireBody = wireBody.wires[0].offsetPlanarWire(
                    vector3d(n),
                    -args.command.commandInputs.itemById("viExpansion").value,
                    2
                )
                
                # creates the actual face
                surface = tbm.createFaceFromPlanarWires([offsetWireBody])

                # Adds face and optionally finishes the baseFeature
                if(baseFeature):
                    realSurface = bodies.add(surface, baseFeature)
                    realSurface.opacity = 0.7
                    baseFeature.finishEdit()
                else:
                    realSurface = bodies.add(surface)
                    realSurface.opacity = 0.7

                args.isValidResult = False  
        except:
            print(traceback.format_exc())


# Fires when CommandInputs are changed
# Responsible for dynamically updating other Command Inputs
class CommandPlaneInputChangedHandler(adsk.core.InputChangedEventHandler):
    def __init__(self):
        super().__init__()
    def notify(self, args):
        try:
            global pers
            if args.input.id == "viExpansion":
                pers["viExpansion"] = args.input.value
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

            # Registers the CommandExecuteHandler
            onExecute = CommandCylinderExecuteHandler(vsi)
            cmd.execute.add(onExecute)
            _handlers.append(onExecute)  

            # Registers the CommandDestryHandler
            onExecutePreview = CommandCylinderExecutePreviewHandler(vsi)
            cmd.executePreview.add(onExecutePreview)
            _handlers.append(onExecutePreview)

            # Registers the CommandInputChangedHandler          
            onInputChanged = CommandCylinderInputChangedHandler()
            cmd.inputChanged.add(onInputChanged)
            _handlers.append(onInputChanged) 

            global pers

            viExpansion = cmd.commandInputs.addValueInput("viExpansion", "Expansion", "mm", adsk.core.ValueInput.createByReal(pers["viExpansion"]))

        except:
            print(traceback.format_exc())


#Fires when the User executes the Command
#Responsible for doing the changes to the document
# Almost identical to the ExecutePreviewEventHandler, but that one also adds custom graphics.
# There are better ways to do this, this is sutpid
class CommandCylinderExecuteHandler(adsk.core.CommandEventHandler):
    def __init__(self, vsi):
        self.vsi = vsi
        super().__init__()
    def notify(self, args):
        try:
            if self.vsi.selected_points:
                cylinderExecuteStuff(self, args)
        except:
            print(traceback.format_exc())


#Fires when the User executes the Command
#Responsible for doing the changes to the document
class CommandCylinderExecutePreviewHandler(adsk.core.CommandEventHandler):
    def __init__(self, vsi):
        self.vsi = vsi
        super().__init__()
    def notify(self, args):
        try:
            if self.vsi.selected_points:
                cylinderExecuteStuff(self, args).opacity = 0.7
                args.isValidResult = False
        except:
            print(traceback.format_exc())


# Fires when CommandInputs are changed
# Responsible for dynamically updating other Command Inputs
class CommandCylinderInputChangedHandler(adsk.core.InputChangedEventHandler):
    def __init__(self):
        super().__init__()
    def notify(self, args):
        try:
            global pers
            if args.input.id == "viExpansion":
                pers["viExpansion"] = args.input.value
        except:
            print(traceback.format_exc())


def cylinderExecuteStuff(handler, args):            
    # Gets actual coordinates from selected indexes
    crds = handler.vsi.mesh_points[ list(handler.vsi.selected_points) ]

    # Fits a plane to the set of coordinates
    # result contains metadata res.x is actual result
    # avgCrds = np.average(crds, axis=0)
    try:
        res = fitCylinderToPoints(crds)
    except:
        return

    # Rejects bad results (by looking for extreme values)
    if(max(res.x) > 100000 or min(res.x) < -100000):
        return
    
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

    if des.designType:
        baseFeature = root.features.baseFeatures.add()
        baseFeature.startEdit()
    else:
        baseFeature = None
    
    tbm = adsk.fusion.TemporaryBRepManager.get()

    tempBRepBodies = []

    circle1 = adsk.core.Circle3D.createByCenter(
        point3d(p1 - n * args.command.commandInputs.itemById("viExpansion").value),
        vector3d(n),
        res.x[6]
        )

    circle2 = adsk.core.Circle3D.createByCenter(
        point3d(p2 + n * args.command.commandInputs.itemById("viExpansion").value),
        vector3d(n),
        res.x[6]
        )

    wireBody1, _ = tbm.createWireFromCurves([circle1])
    wireBody2, _ = tbm.createWireFromCurves([circle2])

    surface = tbm.createRuledSurface(wireBody1.wires.item(0), wireBody2.wires.item(0))
    
    if(baseFeature):
        realSurface = bodies.add(surface, baseFeature)
        baseFeature.finishEdit()
    else:
        realSurface = bodies.add(surface)

    return realSurface

      









# p = Point a,b = Line
def distPtToLine(p, a, b):
    """Gets the distance between an array of points and a line

    Parameters
    ----------
    pts : List or np array of shape (-1, 3)
        List of points on the cylindrical surface
    a : List or np array of shape (3)
        a point on the line
    b : List or np array of shape (3)
        another point on the line

    Returns
    -------
    np array of shape (-1)
        array of distances for each point to the line
    """
    return np.linalg.norm( np.cross(b-a, a-p), axis=1) / np.linalg.norm(b-a)


def distPtToPlane(p, o, n):
    """Gets the distance between an array of points and a plane

    Parameters
    ----------
    pts : List or np array of shape (-1, 3)
        List of points on the cylindrical surface
    o : List or np array of shape (3)
        Vector to origin of cylinder
    n : List or np array of shape (3)
        Normal vector of cylinder

    Returns
    -------
    np array of shape (-1)
        array of distances for each point to the plane
    """
    return np.dot( (p-o), n ) / np.linalg.norm(n)


def sphericalToDirection(ang):
    """Generates a 3D unit vector given two spherical angles

    Parameters
    ----------
        ang : an array containing a (polar) angle from the Z axis and an (azimuth) angle in the X-Y plane from the X axis, in radians

    Returns
    -------
    a 3D unit vector defined by the input angles (input = [0,0] -> output = [0,0,1])
    """
    return [math.cos(ang[0]) * math.sin(ang[1]), math.sin(ang[0]) * math.sin(ang[1]), math.cos(ang[1])]

# Simple solution definiton
class Soln():
    pass


def fitCylinderToPoints(pts):
    """Solves for the parameters of an infinite cylinder given a set of 3D cartesian points

    Parameters
    ----------
        pts: a Nx3 array of points on the cylinder, ideally well-distributed radially with some axial variation

    Returns
    -------
    Outputs a solution object containing members:
        x: estimated cylinder origin, axis, and radius parameters [ px, py, pz, ax, ay, az, r ]
        fun: non-dimensional residual of the fit to be used as a quality metric

    Note
    -------
    - The general approach is a hybrid search where several parameters are handled using iterative optimization and the rest are directly solved for
    - The outer search is performed over possible cylinder orientations (represented by two angles)
        - A huge advantage is that these parameters are bounded, making search space coverage tractable despite the multiple local minima
        - Reducing the iterative search space to two parameters dramatically helps iteration time as well
        - To help ensure the global minimum is found, a coarse grid method is used over the bounded parameters
        - A gradient method is used to refine the found coarse global minimum
    - For each orientation, a direct (ie, non-iterative) LSQ solution is used for the other 3 paremeters
        - This can be visualized as checking how "circular" the set of points appears when looking along the expected axis of the cylinder
    - Note that no initial guess is needed since whole orientation is grid-searched and the other parameters are found directly without iteration
    """


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


def fitCylinderOnAxis(pts, axis=np.array([0,0,1])):
    """Solves for some parameters of an infinite cylinder given a set of 3D cartesian points and an axis for the cylinder

    Parameters
    ----------
        pts: a Nx3 array of points on the cylinder, ideally well-distributed radially with some axial variation
        axis: a vector containing the central axis direction of the cylinder

    Returns
    -------
    Outputs a tuple containing:
        pest: estimated cylinder origin and radius parameters [ px, py, pz, r ]
        resid: non-dimensional residual of the fit to be used as a quality metric

    Note
    -------
    - Generates a set of orthonormal basis vectors based on the input cylinder axis
    - Forms a direction cosine matrix from the basis vectors and rotates the points into the cylinder frame
    - Collapses the points along the cylinder axis and runs 2D circle estimation to get the lateral offset and radius
    - Along-axis offset is meaningless for an infinite cylinder, so the mean of the input points in that direction is arbitrarily used
    - Maps the offsets back to the original coordinate system
    - Note that the returned residual is the 2D circular fit metric
    """
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


def fitCircle2D(pts):
    """Solve for parameters of a 2D circle given pts that lie on the circle perimeter.
    
    Parameters
    ----------
    pts : a Nx3 array of points on the cylinder, 3 points minimum, ideally well-distributed

    Returns
    -------
    Outputs a tuple containing:
        pest: estimated 2D parameters [ px, py, r ]
        resid: non-dimensional residual of the fit to be used as a quality metric

    Note
    -------
    - Reparameterizes the problem into a nondimensional linear form through a change of variables
    - 2D parameters are solved for directly using linear least squares rather than an iterative method
    """
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


def fitPlaneToPoints2(pts):
    """Solve for 3D parameters of an infinite plane given pts that lie on the plane surface.
    
    Parameters
    ----------
    pts : List or np array of shape (-1, 3)
        List of points

    Returns
    -------
    list of shape (2)
        Soln object containing
            x : list of shape (3)
                [angle1, angle2, z-offset]
            fun : number
                remaining error function (distance squared)
    """
    # Create search grid for orientation angles
    # (note, may need to increase number of grid points in cases of poorly defined point sets)
    ranges = (slice(0, 2*math.pi, math.pi/4), slice(0, math.pi, math.pi/4))

    # Perform brute force grid search for approximate global minimum location followed by iterative fine convergence
    # (note, this can probably be replaced with a different global search method, ie simulated annealing, 
    #   but the grid search should work reasonably well given that the search space is bounded)
    sol = scipy.optimize.brute(lambda x: fitPlaneOnAxis(pts, sphericalToDirection(x))[1], ranges, finish=scipy.optimize.fmin)

    result = fitPlaneOnAxis(pts, sphericalToDirection(sol))

    # Mimic return elements in scipy.optimize.minimize functions
    res = Soln()
    setattr(res, 'x', [sol[0], sol[1], result[0]])
    setattr(res, 'fun', result[1])
    return res


def fitPlaneOnAxis(pts, pln=np.array([0,0,1])):
    """Solve for offset parameters of a plane given its normal and pts that lie on the plane surface.

    Parameters
    ----------
    pts : List or np array of shape (-1, 3)
        List of points
    pln : List or np array of shape (3)
        Plane normal vector

    Returns
    -------
    list of shape (2)
        [z-offset, distance_squared_error]
    """
    # Create basis vectors for transformed coordinate system
    w = pln
    u = np.cross(w,np.array([w[1],w[2],w[0]]))
    u = u / np.linalg.norm(u)
    v = np.cross(w,u)
    
    # Construct DCM and rotate points into plane frame
    C = np.array([u, v, w]).transpose()
    pts3d = np.array(pts)
    N = len(pts3d)
    pts3drot = pts3d.dot(C)

    z = pts3drot[:, 2]

    zo = np.average(z)

    return (zo, np.sum((z-zo)**2))


def getConvexHull(pts, pln):
    """Gets the 2D ConvexHull of a set of 3D points projected onto a plane and makes them coplanar

    Parameters
    ----------
    pts : List or np array of shape (-1, 3)
        List of points
    pln : List or np array of shape (3)
        Plane normal vector

    Returns
    -------
    list of shape (-1, 2, 3)
        -1 line segments, [startPoint, endPoint], [x, y, z]
    """
    # Create basis vectors for transformed coordinate system
    w = sphericalToDirection(pln)
    u = np.cross(w,np.array([w[1],w[2],w[0]]))
    u = u / np.linalg.norm(u)
    v = np.cross(w,u)

    # Construct DCM and rotate points into plane frame
    C = np.array([u, v, w]).transpose()
    pts3d = np.array(pts)
    pts3drot = pts3d.dot(C)

    # Computes convex hull on xy of points
    hull = ConvexHull(pts3drot[:,:2])

    # Sorts indices
    hullIndices = sortSimplex2D(hull.simplices.tolist())
    
    # Makes all points coplanar
    pts3drot[:, 2] = np.average(pts3drot[:, 2])

    # Makes loop clockwise
    if not isLoopClockwise([pts3drot[i] for i in hullIndices]):
        for i in hullIndices:
            i.reverse()
        hullIndices.reverse()

    # Rotates points back into oritignal frame
    Ci = np.linalg.inv(C)
    pts3dflat = pts3drot.dot(Ci)
    return np.array([pts3dflat[i] for i in hullIndices])


def sortSimplex2D(x):
    """Orders a list of line segments to align end to end

    Parameters
    ----------
    x : List of shape (-1, 2)
        -1 number of line segments, [startPointIndex, endPointIndex]

    Returns
    -------
    list of shape (-1, 2)
    """
    # Go through elements one by one
    for i in range(len(x)-1):
        # Look for the end index of the current elements in the remaining elements
        for j in range(i+1, len(x)):
            # If the end index is found as the start index of another element move it after the cureent element
            if x[j][0] == x[i][1]:
                x.insert(i+1, x.pop(j))
                break
            # If the end index is found as the end index of another element flip it and move it after the cureent element
            if x[j][1] == x[i][1]:
                x[j].reverse()
                x.insert(i+1, x.pop(j))
                break
    return x


def isLoopClockwise(loop):
    """Gets if a loop of line segments is clockwise

    Parameters
    ----------
    loop : List or np array of shape (-1, 2, 2)
        -1 number of line segments, [startPoint, endPoint], [x,y]

    Returns
    -------
    bool

    Note
    -------
    https://stackoverflow.com/questions/1165647/how-to-determine-if-a-list-of-polygon-points-are-in-clockwise-order
    """
    s = 0
    for i in loop:
        s += (i[1][0] - i[0][0]) * (i[1][1] + i[0][1])
    return s > 0


def cylinderBounds(pts, o, n):
    """Gets bounds of cylinder along its normal

    Parameters
    ----------
    pts : List or np array of shape (-1, 3)
        List of points on the cylindrical surface
    o : List or np array of shape (3)
        Vector to origin of cylinder
    n : List or np array of shape (3)
        Normal vector of cylinder

    Returns
    -------
    np array of shape (2) [min, max]
        array containing the miniumum and maximum extend of the cylinder along its normal
    """
    d = distPtToPlane(pts, o, n)
    return np.array([min(d), max(d)])


def GetRootMatrix(comp):
    """Gets the transformation matrix to tranform coordinates from component space to root space

    Parameters
    ----------
    comp : Component

    Returns
    -------
    adsk.core.Matrix3D
    """
    
    # Gets the root component 
    root = comp.parentDesign.rootComponent

    # Creates an emty matrix
    mat = adsk.core.Matrix3D.create()
  
    # If the component is the root component, return the emty matrix
    if comp == root:
        return mat

    # If there is no occurrence of the component, return the emty matrix
    occs = root.allOccurrencesByComponent(comp)
    if len(occs) < 1:
        return mat

    # Take the first occurence
    occ = occs[0]
    # Split its path
    occ_names = occ.fullPathName.split('+')
    # Get all occurences in the path
    occs = [root.allOccurrences.itemByName(name)for name in occ_names]
    # Get their transforms
    mat3ds = [occ.transform for occ in occs]
    # Reverse the order (importnat for some reason)
    mat3ds.reverse() #important!!
    # Transform the emty matrix by all of them
    for mat3d in mat3ds:
        mat.transformBy(mat3d)
    # Return the finished matrix
    return mat


def point3d(p):
    """Converts list of np array to fusion360 point3d object

    Parameters
    ----------
    p : List or np array of shape (3)

    Returns
    -------
    adsk.core.Point3D
    """
    return adsk.core.Point3D.create(p[0], p[1], p[2])


def vector3d(v):
    """Converts list of np array to fusion360 vector3d object

    Parameters
    ----------
    v : List or np array of shape (3)

    Returns
    -------
    adsk.core.Vector3D
    """
    return adsk.core.Vector3D.create(v[0], v[1], v[2])










# ============================== Selection Input for Vertexes ==============================# 


class VertexSelectionInput:
    handlers = []
    mesh_points = None
    mesh_tris = None
    selected_points = set()


    def __init__(self, args):
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

        self.siMesh = inputs.addSelectionInput('siViMesh', 'Mesh', 'Select Mesh')
        self.siMesh.addSelectionFilter('MeshBodies')
        self.siMesh.setSelectionLimits(0, 1)

        global pars
        self.fsRadius = inputs.addFloatSpinnerCommandInput('fsViRadius', 'Selection Radius', '', 0, 10000, 5, pers['fsViRadius'])

        self.tbSelected = inputs.addTextBoxCommandInput("tbViSelected", "", "0 Selected", 1, True)
        self.tbSelected.isFullWidth = False


    class VertexSelectionClickEventHandler(adsk.core.MouseEventHandler):
        def __init__(self, parent):
            super().__init__()
            self.parent = parent

        def notify(self, args):
            try:    
                print("click")
                if self.parent.mesh_points is not None:
                    # Gets the click & camera position in model space 
                    clickPos3D = np.array(args.viewport.viewToModelSpace(args.viewportPosition).asArray())
                    cameraPos = np.array(args.viewport.camera.eye.asArray())

                    # Checks if camera is in perspective mode
                    if args.viewport.camera.cameraType:
                        # Gets the distance of points to camera click line                      
                        d = distPtToLine(self.parent.mesh_points, clickPos3D, cameraPos)
                    
                    else:
                        centerPos = np.array(args.viewport.viewToModelSpace(adsk.core.Point2D.create(args.viewport.width/2,args.viewport.height/2)).asArray())

                        # Gets the distance of points to click line parallel to view direction                      
                        d = distPtToLine(self.parent.mesh_points, clickPos3D, clickPos3D + (cameraPos-centerPos))

                    # Adds the indices of points closer than the selection radius to the selection set
                    for i, j in enumerate(d):
                        if j < self.parent.fsRadius.value/10:
                            # Adds selection points if shift is not held
                            if(not args.keyboardModifiers == 33554432):
                                self.parent.selected_points.add(i)
                            # Removes selection points when shift is held
                            else:
                                if(i in self.parent.selected_points):
                                    self.parent.selected_points.remove(i)

                    # Updates number of selected points (this also triggers the inputChangedEventHandler)
                    self.parent.tbSelected.text = "{} Selected".format(len(self.parent.selected_points))
            except:
                print(traceback.format_exc())

        
    class VertexSelectionInputChangedEventHandler(adsk.core.InputChangedEventHandler):
        def __init__(self, parent):
            super().__init__()
            self.parent = parent

        def notify(self, args):
            try:
                if args.input.id == "fsViRadius":
                    global pers
                    pers["fsViRadius"] = args.input.value
                # Resonsible for translating mesh when one is selected
                if args.input.id == "siViMesh":
                    if args.input.selectionCount == 1:

                        # Takes focus away from the selection input so clicking vertices does not deselect the mesh
                        args.input.hasFocus = False

                        # Gets node (vertx) coordinates from the display mesh.
                        # The display mesh should be the same as the actual mesh, but the actual mesh has a bug resulting in incorect coordinates
                        nc = args.input.selection(0).entity.displayMesh.nodeCoordinates

                        # Gets the transformation matrix to to transform the local space coordinates to world space
                        mat = GetRootMatrix(args.input.selection(0).entity.parentComponent)

                        # Transforms the coordinates to world space
                        for i in nc:
                            i.transformBy(mat)

                        # Converts coordinates to np array 
                        self.parent.mesh_points = np.array([[i.x, i.y, i.z] for i in nc])

                        # Gets the triangles associated with the mesh
                        self.parent.mesh_tris = self.parent.mesh_points[np.array(args.input.selection(0).entity.displayMesh.nodeIndices)].reshape(-1,3,3)

                    # Clears mesh data when mesh is deselected 
                    else:
                        self.parent.mesh_points = None
                        self.parent.mesh_tris = None
                        self.parent.selected_points = set()
                        self.parent.tbSelected.text = "0 Selected"

            except:
                print(traceback.format_exc())


    class VertexSelectionInputExecutePreviewHandler(adsk.core.CommandEventHandler):
        def __init__(self, parent):
            super().__init__()
            self.parent = parent

        def notify(self, args):
            try:
                # Highlights selected mesh points
                if self.parent.mesh_points is not None and len(self.parent.selected_points) > 0:
                    
                    # Gets the selected points
                    pts = self.parent.mesh_points[list(self.parent.selected_points)]

                    # Creates a new CustomGraphicsGroup
                    cgg = adsk.core.Application.get().activeProduct.rootComponent.customGraphicsGroups.add()

                    # Generates CustomGraphicsCoordinates from selected points
                    coords = adsk.fusion.CustomGraphicsCoordinates.create(np.asarray(pts.reshape(-1), dtype='d'))
                    
                    # Adds the CustomGraphicsCoordinates to the CustomGraphicsGroup
                    cgg.addPointSet(coords, range(len(pts)), 0, 'Resources/point.png')
            except:
                print(traceback.format_exc())


  