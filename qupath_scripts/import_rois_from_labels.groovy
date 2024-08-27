import qupath.lib.objects.PathObjects
import qupath.lib.objects.classes.PathClass
import qupath.lib.regions.ImagePlane
import ij.IJ
import ij.process.ImageProcessor
import ij.measure.Calibration
import qupath.imagej.processing.RoiLabeling
import qupath.imagej.tools.IJTools
import qupath.lib.roi.RoiTools

import static qupath.lib.gui.scripting.QPEx.*

def importLabels(imageData, imp, cal, plane, downsample, threshold, className) {
    // Thresholding fibres
    def ip = imp.getProcessor()
    ip.setThreshold (threshold, 255, ImageProcessor.NO_LUT_UPDATE)
    ImageProcessor ipFibreMask = ip.createMask() // binary image processor
    def roisIJ = RoiLabeling.labelsToConnectedROIs(ipFibreMask, 255)
    
    // Convert ImageJ ROIs to QuPath ROIs
    def rois = roisIJ.collect {
        if (it == null)
            return
        return IJTools.convertToROI(it, cal, downsample, plane);
    }
    
    // Remove all null values from list
    rois = rois.findAll{null != it}
    
    // Convert QuPath ROIs to objects
    def pathObjects = rois.collect {
        return PathObjects.createAnnotationObject(it, getPathClass(className))
    }
    addObjects(pathObjects)
    
    
    // Get the annotations by class and split the multi-part Rois
    def hierarchy = imageData.getHierarchy()
    def selectedObjects = hierarchy.getAnnotationObjects().findAll{it.getPathClass() == getPathClass(className)}
    selectedObjects.each{ it ->
        splitRois = RoiTools.splitROI(it.getROI())
    }
    
    // remove the muti-part Roi and add the split Rois
    removeObjects(selectedObjects, true) 
    def newObjs=[]
    splitRois.each{
        newObjs << PathObjects.createAnnotationObject(it,getPathClass(className))
    }
    addObjects(newObjs)
}

// Delete all existing objects
clearAllObjects()

/// Get the current image
def imageData = getCurrentImageData()
def server = imageData.getServer()

// Get the mask file
def uri = server.getURIs()[0]
def fileImage = new File(uri)
def fileMask = new File (fileImage.getParentFile(), 'mask.png')
print(fileMask)

def path = fileMask.getPath()
def imp = IJ.openImage(path)

double downsample = 1
ImagePlane plane = ImagePlane.getDefaultPlane()
Calibration cal = new Calibration (imp)

importLabels(imageData, imp, cal, plane, downsample, 128, "Outer")
importLabels(imageData, imp, cal, plane, downsample, 255, "Inner")
