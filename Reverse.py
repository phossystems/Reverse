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

                avgCrds = np.average(crds, axis=0)

                # Fits a plane to the set of coordinates
                # result contains metadata res.x is actual result
                res = fitCylinderToPonts(crds, seed = np.array([avgCrds[0], avgCrds[1], avgCrds[2], 1, 1, 1, 2.5]))

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

        self.floatSpinner = inputs.addFloatSpinnerCommandInput('vertexSelectionRadius', 'Selection Radius', '', 0, 10000, 5, 0.5)

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


  