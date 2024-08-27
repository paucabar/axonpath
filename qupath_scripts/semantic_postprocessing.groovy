import qupath.imagej.tools.IJTools
import qupath.lib.roi.RoiTools
import qupath.lib.roi.GeometryTools
import qupath.lib.roi.interfaces.ROI
import qupath.lib.regions.ImagePlane
import qupath.lib.objects.PathObject

import static qupath.lib.gui.scripting.QPEx.*
import org.locationtech.jts.geom.Geometry
import ij.ImagePlus
import ij.process.ImageProcessor
import qupath.imagej.processing.SimpleThresholding


def getIntersectedObjects (fibreObjects, semanticRoi, className) {
    Geometry g1 = semanticRoi.getGeometry()
    List<ROI> intersectionList = []
    ImagePlane plane = ImagePlane.getDefaultPlane()
    fibreObjects.each { fibre ->
        Geometry gFib = fibre.getROI().getGeometry()
        Geometry intersection = g1.intersection(gFib)
        intersectionList << new GeometryTools().geometryToROI(intersection, plane)
    }
    def pathObjects = intersectionList.collect { roi ->
        return PathObjects.createAnnotationObject(roi, getPathClass(className))
    }
    addObjects(pathObjects)
    return
}

// Get the current image
def imageData = getCurrentImageData()
def server = imageData.getServer()


// Create a region request for the entire image & request the pixels from the server
// (This assumes the image is 2D)
double downsample = 1
def request = RegionRequest.createInstance(server, downsample)
ImagePlus imp = IJTools.convertToImagePlus(server, request).getImage()


// Get fibre objects
// Currently, it requires to manually import them from a GeoJASON file
def fibre_objects = getAnnotationObjects().findAll{(it.getPathClass() == getPathClass("Fibre")) }
clearAllObjects()


// Create semantic ROIs from thresholds
ImageProcessor ip = imp.getProcessor()
ip.setThreshold(1, 3, ImageProcessor.NO_LUT_UPDATE)
def multipartFibreRoi = SimpleThresholding.thresholdToROI(ip, request) // generates a multi-part ROI including all non-background pixels
ip.setThreshold(2, 3, ImageProcessor.NO_LUT_UPDATE)
def multipartInnerRoi = SimpleThresholding.thresholdToROI(ip, request) // generates a multi-part ROI including inner tongue and axon pixels
ip.setThreshold(3, 3, ImageProcessor.NO_LUT_UPDATE)
def multipartAxonRoi = SimpleThresholding.thresholdToROI(ip, request) // generates a multi-part ROI including axon pixels only


// Create the three AimSeg object sets
getIntersectedObjects (fibre_objects, multipartFibreRoi, "Outer")
getIntersectedObjects (fibre_objects, multipartInnerRoi, "Inner")
getIntersectedObjects (fibre_objects, multipartAxonRoi, "Axon")

// Resolve hierarchy to make it easier to match objects
resolveHierarchy()