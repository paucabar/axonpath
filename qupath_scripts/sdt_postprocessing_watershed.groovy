import qupath.imagej.processing.Watershed
import qupath.imagej.tools.IJTools
import qupath.lib.common.ColorTools
import qupath.lib.common.GeneralTools
import qupath.lib.images.servers.LabeledImageServer
import qupath.lib.objects.PathObjects
import qupath.lib.regions.RegionRequest

import static qupath.lib.gui.scripting.QPEx.*
import ij.ImagePlus
import ij.process.ImageProcessor
import qupath.imagej.processing.SimpleThresholding
import qupath.lib.roi.RoiTools
import qupath.imagej.processing.RoiLabeling
import ij.measure.Calibration
import qupath.lib.regions.ImagePlane



// Get the current image
def imageData = getCurrentImageData()
def server = imageData.getServer()


// Create output path (relative to project)
def outputDir = buildFilePath(PROJECT_BASE_DIR, 'export')
mkdirs(outputDir)

// Create output subfolders
def labelDir = buildFilePath(outputDir, 'labels')
mkdirs(labelDir)

// Set a threshold and get the seeds as a QuPath ROI List
// Define as double to avoid Groovy automatically choosing the type of the number
double min_threshold = 0.7
double max_threshold = 1
double downsample = 1

// Create a region request for the entire image & request the pixels from the server
// (This assumes the image is 2D)
def request = RegionRequest.createInstance(server, downsample)
ImagePlus imp = IJTools.convertToImagePlus(server, request).getImage()

// Create ROIs from thresholds
// Don't need 'new SimpleThresholding()' etc. because the methods are static
ImageProcessor ip = imp.getProcessor()
ip.setThreshold(min_threshold, max_threshold, ImageProcessor.NO_LUT_UPDATE)
def multipartRoi = SimpleThresholding.thresholdToROI(ip, request) // generates a multi-part ROI including all the thresholded regions
def roiList = RoiTools.splitROI(multipartRoi) // split the multi-part ROI into separate ROIs

// Convert QuPath ROIs to objects
def className = "Seed"
def pathObjects = roiList.collect { roi ->
    return PathObjects.createAnnotationObject(roi, getPathClass(className))
}
addObjects(pathObjects)

// Create an ImageServer for seed instances
def minSize = 100
def seedServer = new LabeledImageServer.Builder(imageData)
        .backgroundLabel(0, ColorTools.BLACK) // Specify background label (usually 0 or 255)
        .downsample(downsample)    // Choose server resolution; this should match the resolution at which tiles are exported
        .useAnnotations()
        .useInstanceLabels()
        .useFilter(p -> p.isAnnotation() && p.getPathClass() == getPathClass('Seed') && p.getROI().getArea() > minSize)
        .multichannelOutput(false) // If true, each label refers to the channel of a multichannel binary image (required for multiclass probability)
        .build()

// Uncomment if you want to export the label image
def name = GeneralTools.stripExtension(imageData.getServer().getMetadata().getName()) // get image name to export annotations
//def pathLabel = buildFilePath(labelDir, name + ".tif") // Define instance output file paths
//writeImage(seedServer, pathLabel) // write the image

// Open the seed labels with ImageJ
ImagePlus impLabels = IJTools.convertToImagePlus(seedServer, request).getImage()
ImageProcessor ipLabels = impLabels.getProcessor()

// Delete all existing objects
clearAllObjects()

// Apply a 2D watershed transform, constraining region growing using an intensity threshold.
// Parameters:
// ip - image containing intensity information
// ipLabels - image containing starting labels; these will be modified
// minIntensity - minimum threshold; labels will not expand into pixels with values below the threshold
// conn8 - true if 8-connectivity should be used; alternative is 4-connectivity

// Use QuPath's ImageJ-friendly Watershed class (not the general Watershed class for SimpleImage inputs)
double minIntensity = 0
boolean conn8 = true
Watershed.doWatershed(ip, ipLabels, minIntensity, conn8)

// Create annotation objects from label image
def roiFibreList = RoiLabeling.labelsToFilledRoiList(ipLabels, conn8)

// Convert ImageJ ROIs to QuPath ROIs
ImagePlane plane = ImagePlane.getDefaultPlane()
Calibration cal = imp.getCalibration()

// Convert ImageJ ROIs to QuPath annotations
def classFibreName = "Fibre"
def pathFibreObjects = roiFibreList.collect { roiIJ ->
    def roi = IJTools.convertToROI(roiIJ, cal, downsample, plane);
    def annotation = PathObjects.createAnnotationObject(roi, getPathClass(classFibreName))
    return annotation
}
addObjects(pathFibreObjects)

// export annotations to GeoJASON
def pathGeoJSON = buildFilePath(labelDir, name + ".geojson") // Define instance output file paths
def annotations = getAnnotationObjects()
exportObjectsToGeoJson(annotations, pathGeoJSON, "FEATURE_COLLECTION")