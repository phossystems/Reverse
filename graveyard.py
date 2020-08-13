# A bunch of code deleted from the main file that may come in handy later on


def clearCustomGraphicsGroup(cgg):
    for i in range(cgg.count):
        cgg.item(i).deleteMe()


def visualizePoints(cgg, pts):
    coords = adsk.fusion.CustomGraphicsCoordinates.create(np.asarray(pts.reshape(-1), dtype='d'))
    cgg.addPointSet(coords, range(len(pts)), 
                   adsk.fusion.CustomGraphicsPointTypes.UserDefinedCustomGraphicsPointType,
                   'TestPoint.png')

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










def run(context):
    try:
        app = adsk.core.Application.get()
        ui = app.userInterface
        
        commandDefinitions = ui.commandDefinitions

        cmdDefCylinder = commandDefinitions.itemById("commandReverseCylinder")
        if not cmdDefCylinder:
            cmdDefCylinder = commandDefinitions.addButtonDefinition("commandReverseCylinder", "Cylinder", "Reconstructs a cylindrical face", 'Resources/Cylinder')

        onCommandCylinderCreated = CommandCylinderCreatedHandler()
        cmdDefCylinder.commandCreated.add(onCommandCylinderCreated)
        _handlers.append(onCommandCylinderCreated)

        ui.allToolbarPanels.itemById("SurfaceCreatePanel").controls.addCommand(cmdDefCylinder)


    except:
        print(traceback.format_exc())


def stop(context):
    try:
        app = adsk.core.Application.get()
        ui = app.userInterface

        commandDefinitions = ui.commandDefinitions

        p = ui.allToolbarPanels.itemById("SurfaceCreatePanel").controls.itemById("ReverseCylinder")
        if p:
            p.deleteMe()

        ui.commandDefinitions.itemById("commandReverseCylinder").deleteMe()


    except:
        print(traceback.format_exc())


def getBoundingBoxVolume(bb):
    return np.prod(np.array(bb.maxPoint.asArray()) - np.array(bb.minPoint.asArray()))


#Returns point, radius [px, py, pz, r]
def fitSphereToPoints(pts, seed=np.array([1,1,1,1])):
    return scipy.optimize.minimize(lambda x: np.sum((np.linalg.norm(pts-np.array([x[0], x[1], x[2]]), axis=1)-x[3])**2) , seed , method = "Powell")
  

#Returns point and vector [px, py, pz, vx, vy, vz]
def fitLineToPoints(pts, seed=np.array([1,1,1,1,1,1])):
    return scipy.optimize.minimize(lambda x: np.sum(distPtToLine(pts, np.array([x[0], x[1], x[2]]), np.array([x[0] + x[3], x[1] + x[4], x[2]+ x[5]]))**2), seed , method = 'Powell')
    

def isPointInvisiblePerspecive(points, cameraPos, tris):
    return [np.any( doesLineIntersectTriangle(np.repeat(np.array([[p, cameraPos]]), len(tris), axis=0) , tris) ) for p in points]


#Takes Array of lines (-1,2,3) and array of triangles and checks for intersection line by line
def doesLineIntersectTriangle(line, triangle):
    
    a = np.sign( spv(line[:,0], line[:,1], triangle[:,0], triangle[:,1]) )
    b = np.sign( spv(line[:,0], line[:,1], triangle[:,1], triangle[:,2]) )
    c = np.sign( spv(line[:,0], line[:,1], triangle[:,2], triangle[:,0]) )

    return np.logical_and(np.not_equal(np.sign(spv(line[:,0], triangle[:,0], triangle[:,1], triangle[:,2]) ), np.sign(spv(line[:,1], triangle[:,0], triangle[:,1], triangle[:,2]) )), np.logical_and(np.equal(a, b), np.equal(b, c)))
            

#Signed volume of a Parallelepiped, equal to 6 times the signed volume of a tetrahedron
def spv(a, b, c, d):
    #Einsum seems to be used for row-wise dot products
    return np.einsum('ij,ij->i', d-a, np.cross(b-a, c-a))


