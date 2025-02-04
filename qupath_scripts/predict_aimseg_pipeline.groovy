/**
 * This script demonstrates how to run an AimSeg model in QuPath,
 * including the basic functions for model inference and post-processing in QuPath.
 * The output is stored as three distinct classes of segmented objects: axon, inner tongue, and fibre.
 * 
 * The script establishes meaningful hierarchies, recognising that a fibre can contain multiple
 * inner tongue objects, and an inner tongue object may contain multiple axon objects.
 *
 * Prior to running this script, ensure that the DJL extension is installed in QuPath 
 * and PyTorch has been downloaded – see https://qupath.readthedocs.io/en/stable/docs/deep/djl.html
 */
 
 /*
  * TODO: currently the script works with annotation objects, but it should use detection objects
  */ 


/**
 * Some imports
 */
 
import ij.ImagePlus
import ij.process.ImageStatistics
import qupath.lib.images.servers.PixelType
import qupath.lib.regions.Padding
import qupath.lib.regions.RegionRequest
import qupath.opencv.ops.ImageOps
import qupath.opencv.tools.OpenCVTools

import ij.process.ImageProcessor
import qupath.imagej.processing.SimpleThresholding
import qupath.lib.roi.RoiTools
import qupath.imagej.processing.RoiLabeling
import ij.measure.Calibration
import qupath.lib.regions.ImagePlane

import qupath.imagej.processing.Watershed
import qupath.imagej.tools.IJTools
import qupath.lib.common.ColorTools
import qupath.lib.common.GeneralTools
import qupath.lib.images.servers.LabeledImageServer
import qupath.lib.objects.PathObjects

import java.nio.file.Paths

import ij.IJ
import ij.plugin.filter.ParticleAnalyzer
import ij.measure.ResultsTable
import ij.plugin.frame.RoiManager
import ij.measure.Measurements

import org.locationtech.jts.geom.Geometry
import qupath.lib.objects.hierarchy.PathObjectHierarchy
import qupath.lib.objects.PathObject

import static qupath.lib.gui.scripting.QPEx.*
import qupath.ext.djl.DjlTools


/**
 * Define some methods
 */

/**
 * Function to calculate the downsample factor based on target pixel size
 */
double calculateDownsampleFactor(imageData, double targetPixelSizeMicrons, boolean allowUpscaling = false) {
    // Get the current pixel size from image metadata
    def pixelSizeMicrons = imageData.getServer().getPixelCalibration().getAveragedPixelSizeMicrons()
    
    if (pixelSizeMicrons == null) {
        throw new IllegalArgumentException("Pixel size could not be determined from the image metadata.")
    }
    
    // Calculate the downsample factor
    def downsampleFactor =  targetPixelSizeMicrons / pixelSizeMicrons
    
    // Handle upscaling based on the user's preference
    if (!allowUpscaling && downsampleFactor < 1) {
        throw new IllegalArgumentException("Target pixel size is smaller than the current pixel size. Upscaling is disabled.")
    }

    // Ensure the downsample factor is at least 1 if upscaling is not allowed
    if (!allowUpscaling) {
        downsampleFactor = Math.max(downsampleFactor, 1)
    }

    return downsampleFactor.round(1)
}

/**
 * This function runs an AimSeg model using DJL in QuPath
 */
ImagePlus modelInference (uri, layout, inputWidth, inputHeight, padding, inputShape, imageData, server, request) {
    ImagePlus imp = IJTools.convertToImagePlus(server, request).getImage()
    
    // Get the statistics of the image to get the minimum and maximum pixel values
    ImageStatistics stats = imp.getStatistics()
    double min = stats.min
    double max = stats.max
    double difference = max - min
    
    // Apply prediction
    try (def dnn = DjlTools.createDnnModel(uri, layout, inputShape as int[])) {
        def op = ImageOps.buildImageDataOp()
            .appendOps(
                    ImageOps.Core.ensureType(PixelType.FLOAT32),
                    //ImageOps.Normalize.percentile(0.1, 99.9),
                    ImageOps.Core.subtract(min),
                    ImageOps.Core.divide(difference),
                    ImageOps.ML.dnn(dnn, inputWidth, inputHeight, padding)
            )
    
        // Run the prediction, getting an OpenCV Mat as output
        def mat = op.apply(imageData, request)
    
        // Convert to an ImageJ ImagePlus
        impOutput = OpenCVTools.matToImagePlus("Prediction", mat)
        mat.close()
    }
    
    //impOutput.show()
    
    return impOutput
}

/**
 * Implements ImageJ's Particle Analyzer
 * The method will always return an ImagePlus
 * options is defined as an integer using ParticleAnalyzer fields
 * options is defined as an integer using Interface Measurements fields
 * results table is not given as an argument because the method is never used to measure
 */
ImagePlus analyzeParticles (ImagePlus imp, int options, int measurements, double minSize, double maxSize, double minCirc, double maxCirc) {
    def rt = new ResultsTable()
    def pa = new ParticleAnalyzer(options, measurements, rt, minSize, maxSize, minCirc, maxCirc)
    ImageProcessor ip = imp.getProcessor()
    ip.setBinaryThreshold()
    pa.setHideOutputImage(true)
    pa.analyze(imp, ip)
    ImagePlus impOutput = pa.getOutputImage()
    if (impOutput.isInvertedLut()) {
        IJ.run(impOutput, "Grays", "") // get the non-inverted LUT
    }
    return impOutput
}

/**
 * Method to get an SDT channel from an image plus and return an instance segmentation in the
 * form of QuPath objects.
 */
void processSDT(ImagePlus imp, String className, int channel, double min_threshold, double max_threshold, double downsample, imageData, request, double translateX = 0, double translateY = 0) {
    // Create ROIs from thresholds
    imp.setC(channel) // Set the channel index (1-based)
    ImageProcessor ip = imp.getProcessor() // Get the ImageProcessor of the specified channel
    ip.setThreshold(min_threshold, max_threshold, ImageProcessor.NO_LUT_UPDATE)
    def multipartRoi = SimpleThresholding.thresholdToROI(ip, request) // generates a multi-part ROI including all the thresholded regions
    def roiList = RoiTools.splitROI(multipartRoi) // split the multi-part ROI into separate ROIs
    
    // Convert QuPath ROIs to objects
    def pathObjects = roiList.collect { roi ->
        return PathObjects.createAnnotationObject(roi, getPathClass("Seed"))
    }
    addObjects(pathObjects)
    
    // Create an ImageServer for seed instances
    def minSizePixels = 700
    
    def seedServer = new LabeledImageServer.Builder(imageData)
            .backgroundLabel(0, ColorTools.BLACK) // Specify background label (usually 0 or 255)
            .downsample(downsample)    // Choose server resolution; this should match the resolution at which tiles are exported
            .useAnnotations()
            .useInstanceLabels()
            .useFilter(p -> p.isAnnotation() && p.getPathClass() == getPathClass('Seed') && p.getROI().getArea() > minSizePixels)
            .multichannelOutput(false) // If true, each label refers to the channel of a multichannel binary image (required for multiclass probability)
            .build()
    
    // Uncomment if you want to export the label image
    //def name = GeneralTools.stripExtension(imageData.getServer().getMetadata().getName()) // get image name to export annotations
    //def pathLabel = buildFilePath(labelDir, name + ".tif") // Define instance output file paths
    //writeImage(seedServer, pathLabel) // write the image
    
    // Open the seed labels with ImageJ
    ImagePlus impLabels = IJTools.convertToImagePlus(seedServer, request).getImage()
    ImageProcessor ipLabels = impLabels.getProcessor()
    
    // Delete all existing objects
    removeObjects(getCurrentImageData().getHierarchy().getAnnotationObjects().findAll { it.getPathClass() == getPathClass("Seed") }, true)
    
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
    def roiDetected = RoiLabeling.labelsToFilledRoiList(ipLabels, conn8)
    
    // Convert ImageJ ROIs to QuPath ROIs
    ImagePlane plane = ImagePlane.getDefaultPlane()
    Calibration cal = imp.getCalibration()
    
    // Convert ImageJ ROIs to QuPath annotations
    def pathDetectedObjects = roiDetected.collect { roiIJ ->
        def roi = IJTools.convertToROI(roiIJ, cal, downsample, plane);
        def annotation = PathObjects.createAnnotationObject(roi.translate(translateX, translateY), getPathClass(className))
        return annotation
    }
    addObjects(pathDetectedObjects)
}

/**
 * Method to get a semantic channel from an image plus and return an instance segmentation in the
 * form of QuPath objects.
 */
void processSemantic(ImagePlus imp, String className, int channel, int label, double downsample, imageData, request, double translateX, double translateY) {
    // Create ROIs from thresholds
    imp.setC(channel) // Set the channel index (1-based)
    ImageProcessor ip = imp.getProcessor() // Get the ImageProcessor of the specified channel
    ip.setThreshold(label, label, ImageProcessor.NO_LUT_UPDATE)
    ImageProcessor ip_mask = ip.createMask() // image processor
    ImagePlus imp_mask = new ImagePlus("Binary Mask", ip_mask) // image processor to image plus

    int options_add_manager = ParticleAnalyzer.SHOW_MASKS + ParticleAnalyzer.ADD_TO_MANAGER + ParticleAnalyzer.COMPOSITE_ROIS
    int measurements_area = Measurements.AREA
    ImagePlus binaryMask = analyzeParticles(imp_mask, options_add_manager, measurements_area, 0, Double.POSITIVE_INFINITY, 0, 1)
    RoiManager rm = RoiManager.getInstance()
    rm.setVisible(false)
    def roiList = rm.getRoisAsArray()
    rm.close()

    // Convert ImageJ ROIs to QuPath annotations
    ImagePlane plane = ImagePlane.getDefaultPlane()
    Calibration cal = imp.getCalibration()
    
    def pathDetectedObjects = roiList.collect { roi ->
        def roiIJ = IJTools.convertToROI(roi, cal, downsample, plane)
        def annotation = PathObjects.createAnnotationObject(roiIJ.translate(translateX, translateY), getPathClass(className))
        return annotation
    }
    addObjects(pathDetectedObjects)
}

// Method to compute the intersection over Object2 area (IoO2A) between two object classes 
// and create hierarchical relationships
// If IoO2A == 1, Object2 is added below Object1 immediately
// If 0.9 < IoO2A < 1, Object2 is replaced by the intersection before being added below Object1

def establishHierarchyBasedOnIoO2(objectsPrimary, objectsSecondary) {
    // Initialise the PathObjectHierarchy
    def pathHierarchy = new PathObjectHierarchy()

    objectsPrimary.eachWithIndex { parentObject, indexPrimary ->
        Geometry parentGeometry = parentObject.getROI().getGeometry()
        
        objectsSecondary.eachWithIndex { secondaryObject, indexSecondary ->
            Geometry secondaryGeometry = secondaryObject.getROI().getGeometry()
            float intersectionArea = parentGeometry.intersection(secondaryGeometry).getArea()
            
            if (intersectionArea > 0) {
                def parentArea = parentGeometry.getArea()
                def secondaryArea = secondaryGeometry.getArea()
                float ioo2 = intersectionArea / secondaryArea
                
                // Establish hierarchy directly if IoO2 == 1
                if (ioo2 == 1) {
                    pathHierarchy.addObjectBelowParent(parentObject, secondaryObject, true)
                }
                // Replace secondary object with intersection and establish hierarchy if 0.9 < IoO2 < 1
                else if (ioo2 > 0.9) {
                    def intersectionGeometry = parentGeometry.intersection(secondaryGeometry)
                    def intersectionROI = GeometryTools.geometryToROI(intersectionGeometry, secondaryObject.getROI().getImagePlane())
                    
                    // Create the intersection object as an annotation with the same PathClass
                    def intersectionObject = PathObjects.createAnnotationObject(intersectionROI, secondaryObject.getPathClass())
                    
                    // Replace the secondary object with the intersection object
                    pathHierarchy.removeObject(secondaryObject, true) // Remove the original object
                    pathHierarchy.addObject(intersectionObject, false) // Add the intersection object
                    
                    // Establish the hierarchy
                    pathHierarchy.addObjectBelowParent(parentObject, intersectionObject, true)
                }
            }
        }
    }
}

// Method to identify all the objects with no parent object

def findParentless (objects) {
    println "Checking ${objects.size()} objects"
    def parentless = objects.findAll { it.getLevel() == 1 } // level 1 because image is the 'root'
    println "Found ${parentless.size()} parentless objects"
    return parentless
}




/**
 * Segmentation pipeline
 */

//Some parameters

// Model file
def modelPath = "D:/pcarrillo/Git_Repos/AimSeg-Monai_3Targets/weights/weights_tem.pt" // the path to your model here
def uri = Paths.get(modelPath).toUri()

// Image data
def imageData = getCurrentImageData()

// Model parameters
int inputWidth = 512
int inputHeight = inputWidth
int nChannels = 1
def padding = Padding.symmetric(32)
def layout = "NCHW"
def inputShape = [1, nChannels, inputHeight, inputWidth]

// Image parameters
double targetPixelSizeMicrons = 0.008 // optimised pixel size for electron microscopy, 0.07 microns for brightfield images
double downsample = calculateDownsampleFactor(imageData, targetPixelSizeMicrons, true)

// Post-processing parameters
double min_threshold = 0.7
double max_threshold = 1

// Get an ImageJ representation of the output
ImagePlus impOutput

// Use a selected annotation if we have one, otherwise request pixels for the full image
double translateX = 0.0
double translateY = 0.0

def selectedObject = getSelectedObject()
RegionRequest request
def server = imageData.getServer()
if (selectedObject != null && selectedObject.isAnnotation()) {
    def roi = selectedObject.getROI()
    translateX = roi.getBoundsX()
    translateY = roi.getBoundsY()
    request = RegionRequest.createInstance(server.getPath(), downsample, roi)
} else {
    request = RegionRequest.createInstance(server, downsample)
}

// Run model on the specified image region
impOutput = modelInference (uri, layout, inputWidth, inputHeight, padding, inputShape, imageData, server, request)

// Instance segmentation on model prediction
processSDT(impOutput, "Fibre", 2, min_threshold, max_threshold, downsample, imageData, request, translateX, translateY)
processSDT(impOutput, "Axon", 3, min_threshold, max_threshold, downsample, imageData, request, translateX, translateY)
processSemantic(impOutput, "Inner Tongue", 1, 2, downsample, imageData, request, translateX, translateY)

// Establish hierarchy
def fibre_objects = getAnnotationObjects().findAll{(it.getPathClass() == getPathClass("Fibre")) }
def axon_objects = getAnnotationObjects().findAll{(it.getPathClass() == getPathClass("Axon")) }
def inner_tongue_objects = getAnnotationObjects().findAll {it.getPathClass() == getPathClass("Inner Tongue")}

println "Comparing ${fibre_objects.size()} fibre objects vs ${inner_tongue_objects.size()} inner tongue objects"
establishHierarchyBasedOnIoO2 (fibre_objects, inner_tongue_objects)

println "Comparing ${inner_tongue_objects.size()} inner tongue objects vs ${axon_objects.size()} axon objects"
establishHierarchyBasedOnIoO2 (inner_tongue_objects, axon_objects)

// Remove parentless pbjects
def combined_objects = axon_objects + inner_tongue_objects
parentless = findParentless (combined_objects)
removeObjects (parentless, false) // true to keep children objects

return