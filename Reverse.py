#Author-Nico Schlüter
#Description-An Addin for reconstructing surfaces from meshes

import adsk.core, adsk.fusion, adsk.cam, traceback
import time
import inspect
import os
import sys


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

shapes = []










# ============================== Addin Start & Stop  ==============================
# Responsible for createing and cleaning up commands & UI stuff


def run(context):
    try:
        app = adsk.core.Application.get()
        ui = app.userInterface
        
        commandDefinitions = ui.commandDefinitions

        tabReverse = ui.allToolbarTabs.itemById("tabReverse")
        if tabReverse:
            tabReverse.deleteMe()

        tabReverse = ui.workspaces.itemById("FusionSolidEnvironment").toolbarTabs.add("tabReverse", "Reverse Engineer")

        panelSetup = ui.allToolbarPanels.itemById("panelReverseSetup")
        if panelSetup:
            panelSetup.deleteMe()

        panelSetup = tabReverse.toolbarPanels.add("panelReverseSetup", "Setup")

        

        cmdDef = commandDefinitions.itemById("commandReversePlace")
        if cmdDef:
            cmdDef.deleteMe()

        cmdDef = commandDefinitions.addButtonDefinition("commandReversePlace", "Place",
                                                        "Places Mesh on XY Plane", '')
        

        onCommandCreated = CommandPlaceCreatedHandler()
        cmdDef.commandCreated.add(onCommandCreated)
        _handlers.append(onCommandCreated)

        panelSetup.controls.addCommand(cmdDef)

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

                crds = self.vsi.mesh_points[ list(self.vsi.selected_points) ]

                res = fitPlaneToPoints( crds , seed=np.concatenate((crds[0], np.cross(crds[0]-crds[-1], crds[1]-crds[-1]) )))

                print(res)

                app = adsk.core.Application.get()
                des = app.activeProduct
                root = des.rootComponent
                bodies = root.bRepBodies

                des.designType = 0

                
                '''
                tbm = adsk.fusion.TemporaryBRepManager.get()

                #Array to keep track of TempBRepBodies
                tempBRepBodies = []

                cylinder = tbm.createCylinderOrCone(adsk.core.Point3D.create(res.x[0], res.x[1], res.x[2]),
                                                    5,
                                                    adsk.core.Point3D.create(res.x[0]+res.x[3]/100, res.x[1]+res.x[4]/100, res.x[2]+res.x[5]/100),
                                                    5)
                tempBRepBodies.append(cylinder)

                for b in tempBRepBodies:
                    bodies.add(b)
                '''
                
                #print(root.allOccurrencesByComponent(self.vsi.selectionInput.selection(0).entity.parentComponent).count)
                #print(self.vsi.selectionInput.selection(0).entity)

                bodies = adsk.core.ObjectCollection.create()
                bodies.add(self.vsi.selectionInput.selection(0).entity)

                # Create a transform to do move
                vector = adsk.core.Vector3D.create(0.0, 10.0, 0.0)
                transform = adsk.core.Matrix3D.create()

                transform.setToRotation(3.14/4, adsk.core.Vector3D.create(0,0,1), adsk.core.Point3D.create(0,0,0))

                #adsk.core.Vector3D.create(res.x[3], res.x[4], res.x[5])

                #transform.setToRotateTo(adsk.core.Vector3D.create(0,1,0),
                #                        adsk.core.Vector3D.create(0,0,1))

                #transform.setToRotateTo(adsk.core.Vector3D.create(res.x[3], res.x[4], res.x[5]),
                #                        adsk.core.Vector3D.create(0,0,1))


                # Create a move feature
                moveFeats = root.features.moveFeatures
                moveFeatureInput = moveFeats.createInput(bodies, transform)
                moveFeats.add(moveFeatureInput)
            
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
def fitLineToPoints(pts):
    return scipy.optimize.minimize(lambda x: np.sum(distPtToLine(pts, np.array([x[0], x[1], x[2]]), np.array([x[0] + x[3], x[1] + x[4], x[2]+ x[5]]))**2), np.ones(6) , method = 'Powell').x
    

def fitPlaneToPoints(pts, seed=np.array([1,1,1,1,1,1])):
    return scipy.optimize.minimize(lambda x: np.sum(

        distPtToPlane(pts, np.array([x[0], x[1], x[2]]), np.array([x[3], x[4], x[5]]))**2
        
        ), seed , method = 'Powell')

#Returns point, vector and radius [px, py, pz, vx, vy, vz, r]
def fitCylinderToPonts(pts):
    return scipy.optimize.minimize(lambda x: np.sum((distPtToLine(pts, np.array([x[0], x[1], x[2]]), np.array([x[0] + x[3], x[1] + x[4], x[2]+ x[5]]))-x[6])**2) , np.ones(7) , method = "Powell").x
  

#Returns point, radius [px, py, pz, r]
def fitSphereToPoints(pts):
    return scipy.optimize.minimize(lambda x: np.sum((np.linalg.norm(pts-np.array([x[0], x[1], x[2]]), axis=1)-x[3])**2) , np.ones(4) , method = "Powell")
  

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


def generateInfoText(avgD=None, maxD=None, px=None, py=None, pz=None, vx=None, vy=None, vz=None, radius=None, length=None, dim=None):
    text = []
    if(avgD):
        text.append("Average Deviation  . . . :  {}\n".format(avgD))
    if(maxD):
        text.append("Maximum Deviation . . :  {}\n".format(maxD))
    if(px):
        text.append("Position X  . . . . . . . . . :  {}\n".format(px))
    if(py):
        text.append("Position Y  . . . . . . . . . :  {}\n".format(py))
    if(pz):
        text.append("Position Z  . . . . . . . . . :  {}\n".format(pz))
    if(vx):
        text.append("Orientation X . . . . . . . :  {}\n".format(vx))
    if(vy):
        text.append("Orientation Y . . . . . . . :  {}\n".format(vy))
    if(vz):
        text.append("Orientation Z . . . . . . . :  {}\n".format(vz))
    if(radius):
        text.append("Radius . . . . . . . . . . . . :  {}\n".format(radius))
    if(length):
        text.append("Length . . . . . . . . . . . . :  {}\n".format(length))
    if(dim):
        text.append("Dimensions . . . . . . . . :  {}\n".format(dim))
    return ''.join(text)


def updateTable(table, shapes):
    selected = table.selectedRow

    table.clear()

    for i, s in enumerate(shapes):
        dd = table.commandInputs.addDropDownCommandInput('dropDownTable{}'.format(i), '', 0)
        dd.listItems.add("Flat", s[2]==0, '')
        dd.listItems.add("Cylinder", s[2]==1, '')
        dd.listItems.add("Sphere", s[2]==2, '')
        dd.listItems.add("Plane", s[2]==3, '')
        dd.listItems.add("Line", s[2]==4, '')
        dd.listItems.add("Point", s[2]==5, '')
        table.addCommandInput(dd, i, 0)

        bv = table.commandInputs.addBoolValueInput('boolValueTable{}'.format(i), 'Auto', True)
        bv.value = s[3]
        table.addCommandInput(bv, i, 1)

        sv = table.commandInputs.addStringValueInput('textBoxTable{}'.format(i), '', '{} ({})'.format(len(s[0])+len(s[1]), len(s[0])))
        sv.isReadOnly = True
        table.addCommandInput(sv,i, 2)

        fs = table.commandInputs.addFloatSpinnerCommandInput('floatSpinnerTable{}'.format(i), '', '', 0, 100000, 5, s[4]) 
        table.addCommandInput(fs, i, 3)

    table.selectedRow = selected


def clearCustomGraphicsGroup(cgg):
    for i in range(cgg.count):
        cgg.item(i).deleteMe()


def visualizePoints(cgg, pts):
    coords = adsk.fusion.CustomGraphicsCoordinates.create(np.asarray(pts.reshape(-1), dtype='d'))
    cgg.addPointSet(coords, range(len(pts)), 
                   adsk.fusion.CustomGraphicsPointTypes.UserDefinedCustomGraphicsPointType,
                   'TestPoint.png')

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

                    coords = adsk.fusion.CustomGraphicsCoordinates.create(np.asarray(pts.reshape(-1), dtype='d'))
                    
                    cgg.addPointSet(coords, range(len(pts)), 0, 'TestPoint.png')
            except:
                print(traceback.format_exc())


  