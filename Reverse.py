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

#print('\n'.join(sys.path))

try:
    import numpy as np
    import scipy
finally:
    del sys.path[-1]

_handlers = []

COMMAND_ID = "commandNscReverse"
COMMAND_NAME = "Reverse Engineer"
COMMAND_TOOLTIP = "Creates Surfaces from Mesh Points"

command_ref = None
mesh_points = None
mesh_tris = None
graphics = None
shapes = []
   
TOOLBAR_PANELS = ["SurfaceCreatePanel"]


# Fires when the CommandDefinition gets executed.
# Responsible for adding commandInputs to the command &
# registering the other command handlers.
class CommandCreatedHandler(adsk.core.CommandCreatedEventHandler):
    def __init__(self):
        super().__init__()
    def notify(self, args):
        try:
            # Get the command that was created.
            cmd = adsk.core.Command.cast(args.command)

            global command_ref
            command_ref = cmd

            global graphics
            graphics = adsk.core.Application.get().activeProduct.rootComponent.customGraphicsGroups.add()

            # Some common EventHandlers
            # For more Handlers and Info go to:
            # http://help.autodesk.com/view/fusion360/ENU/?guid=GUID-3922697A-7BF1-4799-9A5B-C8539DF57051

            # Registers the CommandDestryHandler
            onExecute = CommandExecuteHandler()
            cmd.execute.add(onExecute)
            _handlers.append(onExecute)  
            
            # Registers the CommandExecutePreviewHandler
            onExecutePreview = CommandExecutePreviewHandler()
            cmd.executePreview.add(onExecutePreview)
            _handlers.append(onExecutePreview)
            
            # Registers the CommandInputChangedHandler          
            onInputChanged = CommandInputChangedHandler()
            cmd.inputChanged.add(onInputChanged)
            _handlers.append(onInputChanged)            
            
            # Registers the CommandDestryHandler
            onDestroy = CommandDestroyHandler()
            cmd.destroy.add(onDestroy)
            _handlers.append(onDestroy)

            # Registers the CommandMouseHandler
            onMouse = CommandMouseClickEventHandler()
            cmd.mouseClick.add(onMouse)
            _handlers.append(onMouse)

                
            # Get the CommandInputs collection associated with the command.
            inputs = cmd.commandInputs
            
            # Implements the three UI sections
            group1 = inputs.addGroupCommandInput('group1', "Selection Settings")
            group1.isExpanded = True

            group2 = inputs.addGroupCommandInput('group2', "Segments")
            group2.isExpanded = True

            group3 = inputs.addGroupCommandInput('group3', "Info")
            group3.isExpanded = False

            #Fist section
            selectionInput = group1.children.addSelectionInput('selection', 'Mesh', 'Select Mesh')
            #selectionInput.addSelectionFilter('MeshBody')
            selectionInput.setSelectionLimits(1, 1)

            floatSpinner1 = group1.children.addFloatSpinnerCommandInput('floatSpinner1', 'Selection Radius', '', 0, 10000, 5, 15)

            boolValue1 = group1.children.addBoolValueInput('boolVale1', 'Select Through', True)

            #Second section

            table1 = group2.children.addTableCommandInput('table1', '', 4, "3:1:2:3")
            table1.minimumVisibleRows = 4
            table1.maximumVisibleRows = 8

            updateTable(table1, [[[1,2,3], [1], 2, True, 105], [[1,2,3], [1], 2, True, 105], [[1,2,3], [1], 2, True, 105]])

            boolValue2 = table1.commandInputs.addBoolValueInput('boolValue2', 'New', False, '', True)
            table1.addToolbarCommandInput(boolValue2)

            boolValue3 = table1.commandInputs.addBoolValueInput('boolValue3', 'Delete', False, '', True)
            table1.addToolbarCommandInput(boolValue3)

            #Third Section
            textBox1 = group3.children.addTextBoxCommandInput('textBox1', '', generateInfoText(radius=69, px=420), 13, True)
            textBox1.isFullWidth = True
            
            #Average deviation, Maximum deviation, Position, Orientation, Radius, Length, DImensions


           
        except:
            print(traceback.format_exc())




#Fires when the User executes the Command
#Responsible for doing the changes to the document
class CommandExecuteHandler(adsk.core.CommandEventHandler):
    def __init__(self):
        super().__init__()
    def notify(self, args):
        try:
            print("execute")
            # TODO: Add Command Execution Stuf Here
            pass                
            
        except:
            print(traceback.format_exc())




# Fires when the Command is being created or when Inputs are being changed
# Responsible for generating a preview of the output.
# Changes done here are temporary and will be cleaned up automatically.
class CommandExecutePreviewHandler(adsk.core.CommandEventHandler):
    def __init__(self):
        super().__init__()
    def notify(self, args):
        try:
            eventArgs = adsk.core.CommandEventArgs.cast(args)
            
            # TODO: Add Command Execution Preview Stuff Here
            
            # If set to True Fusion will use the last preview instead of calling
            # the ExecuteHandler when the user executes the Command.
            # If the preview is identical to the actual executing this saves recomputation
            eventArgs.isValidResult = True                
            
        except:
            print(traceback.format_exc())



# Fires wwhen a MouseEvent occurs
class CommandMouseClickEventHandler(adsk.core.MouseEventHandler):
    def __init__(self):
        super().__init__()
    def notify(self, args):
        try:
            #global command_ref
            global mesh_points
            global mesh_tris


            isPerspective = args.viewport.camera.cameraType == 1

            clickPos3D = np.array(args.viewport.viewToModelSpace(args.viewportPosition).asArray())
            cameraPos = np.array(args.viewport.camera.eye.asArray())

            app = adsk.core.Application.get()
            design = app.activeProduct
            rootComp = design.rootComponent

            sketches = rootComp.sketches
            xyPlane = rootComp.xYConstructionPlane
            sketch = sketches.add(xyPlane)

            sketchPoints = sketch.sketchPoints
            
            startTime = time.time()

            d = distPtToLine(mesh_points, clickPos3D, cameraPos)

            print("Time:")
            print(time.time()-startTime)
            print()

            print(mesh_tris)
            v = isPointInvisiblePerspecive(mesh_points,cameraPos, mesh_tris)

            for i, j in enumerate(mesh_points):
                if not v[i]:
                    sketchPoints.add(adsk.core.Point3D.create(j[0], j[1], j[2]))

            visible = doesLineIntersectTriangle

            
            
        except:
            print(traceback.format_exc())



# Fires when CommandInputs are changed
# Responsible for dynamically updating other Command Inputs
class CommandInputChangedHandler(adsk.core.InputChangedEventHandler):
    def __init__(self):
        super().__init__()
    def notify(self, args):
        try:
            inputChanged = args.input

            global mesh_points
            global mesh_tris

            #Removes focus from selectionInput so clicking wont de-select it
            if inputChanged.id == "selection":
                if inputChanged.selectionCount == 1:
                    inputChanged.hasFocus = False

                    #TO-DO use nodeCoordinatesAsDouble (returns 1D list that needs to be seperated but is probably faster)
                    nc = inputChanged.selection(0).entity.displayMesh.nodeCoordinates

                    mat = GetRootMatrix(inputChanged.selection(0).entity.parentComponent)

                    for i in nc:
                        i.transformBy(mat)
                    mesh_points = np.array( [ [i.x, i.y, i.z] for i in nc] )
                    
                    print("Node Indecies:")
                    print(mesh_points)

                    mesh_tris = mesh_points[np.array(inputChanged.selection(0).entity.displayMesh.nodeIndices)].reshape(-1,3,3)

                    
                    
                    
                        

        except:
            print(traceback.format_exc())
                
                
                
# Fires when the Command gets Destroyed regardless of success
# Responsible for cleaning up                 
class CommandDestroyHandler(adsk.core.CommandEventHandler):
    def __init__(self):
        super().__init__()
    def notify(self, args):
        try:
            # TODO: Add Destroy stuff
            pass
        except:
            print(traceback.format_exc())




def run(context):
    try:
        #import sys
        #print('\n'.join(sys.path))

        

        

        app = adsk.core.Application.get()
        ui = app.userInterface
        
        commandDefinitions = ui.commandDefinitions
        #check the command exists or not
        cmdDef = commandDefinitions.itemById(COMMAND_ID)
        if not cmdDef:
            cmdDef = commandDefinitions.addButtonDefinition(COMMAND_ID, COMMAND_NAME,
                                                            COMMAND_TOOLTIP, '')
        #Adds the commandDefinition to the toolbar
        for panel in TOOLBAR_PANELS:
            ui.allToolbarPanels.itemById(panel).controls.addCommand(cmdDef)
        
        onCommandCreated = CommandCreatedHandler()
        cmdDef.commandCreated.add(onCommandCreated)
        _handlers.append(onCommandCreated)

    except:
        print(traceback.format_exc())




def stop(context):
    try:
        app = adsk.core.Application.get()
        ui = app.userInterface
        
        #Removes the commandDefinition from the toolbar
        for panel in TOOLBAR_PANELS:
            p = ui.allToolbarPanels.itemById(panel).controls.itemById(COMMAND_ID)
            if p:
                p.deleteMe()
        
        #Deletes the commandDefinition
        ui.commandDefinitions.itemById(COMMAND_ID).deleteMe()

    except:
        print(traceback.format_exc())

# a = Point b,c = Line
def distPtToLine(a, b, c):
    return np.linalg.norm( np.cross(c-b, b-a), axis=1) / np.linalg.norm(c-b)


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
    return scipy.optimize.minimize(lambda x: np.sum(distPtToLine(pts, np.array([x[0], x[1], x[2]]), np.array([x[0] + x[3], x[1] + x[4], x[2]+ x[5]]))**2), np.ones(6) , method = 'Nelder-Mead').x
    

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

        tb = table.commandInputs.addTextBoxCommandInput('textBoxTable{}'.format(i), '', '{} ({})'.format(len(s[0])+len(s[1]), len(s[0])), 1, True)
        table.addCommandInput(tb, i, 2)

        fs = table.commandInputs.addFloatSpinnerCommandInput('floatSpinnerTable{}'.format(i), '', '', 0, 100000, 5, s[4]) 
        table.addCommandInput(fs, i, 3)

    table.selectedRow = selected



#shape [ [  [selected], [auto-selected], shape, auto, over  ], ... ]