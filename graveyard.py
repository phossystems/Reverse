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