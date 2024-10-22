/**
 * This script demonstrates how to run a model using DJL in QuPath.
 *
 * You should first install the DJL extension in QuPath, and download PyTorch -
 * see https://qupath.readthedocs.io/en/stable/docs/deep/djl.html
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

import static qupath.lib.gui.scripting.QPEx.*
import qupath.ext.djl.DjlTools

def modelPath = "D:/pcarrillo/Git_Repos/AimSeg-Monai_3Targets/data_evaluation/lee_alpha_03_01_3targets_bigmodel/weights.pt"
def uri = Paths.get(modelPath).toUri()
def imageData = getCurrentImageData()

int inputWidth = 512
int inputHeight = inputWidth
int nChannels = 1
def padding = Padding.symmetric(32)
def layout = "NCHW"
def inputShape = [1, nChannels, inputHeight, inputWidth]
double downsample = 2.0
double min_threshold = 0.7
double max_threshold = 1


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

void processSDT(ImagePlus imp, String className, int channel, double min_threshold, double max_threshold, double downsample, imageData, request, double translateX, double translateY) {
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
    def minSizePixels = 2000
    // Get the pixel size in microns (assuming x and y pixel sizes are the same)
    //def pixelSizeMicrons = imageData.getServer().getPixelCalibration().getPixelWidth()
    //def minSizeMicrons = minSizePixels * Math.pow(pixelSizeMicrons, 2) // Calculate the minimum size in microns^2
    
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

// Method to compute the IoU between 2 object classes and create hierarchical relationships

def assessIoO2A (objects1, objects2) {
    // fill intersection over prediction
    def POH = new PathObjectHierarchy()
    objects1.eachWithIndex { target, index_y ->
        Geometry g1 = target.getROI().getGeometry()
        objects2.eachWithIndex { prediction, index_x ->
            Geometry g2 = prediction.getROI().getGeometry()
            float intersection = g1.intersection(g2).getArea()
            if (intersection > 0) {
                def target_area = g1.getArea()
                def prediction_area = g2.getArea()
                float union = target_area + prediction_area - intersection
                //float iou = intersection / union
                float iop = intersection / prediction_area
                
                // add object2 below object1 if the intersection over the object2 area is close to 1
                if (iop > 0.9) {
                    POH.addObjectBelowParent(target, prediction, true ) // true to fireUpdate
                }
            }
        }
    }
}

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

processSDT(impOutput, "Fibre", 2, min_threshold, max_threshold, downsample, imageData, request, translateX, translateY)
processSDT(impOutput, "Axon", 3, min_threshold, max_threshold, downsample, imageData, request, translateX, translateY)
processSemantic(impOutput, "Inner Tongue", 1, 2, downsample, imageData, request, translateX, translateY)

def fibre_objects = getAnnotationObjects().findAll{(it.getPathClass() == getPathClass("Fibre")) }
def axon_objects = getAnnotationObjects().findAll{(it.getPathClass() == getPathClass("Axon")) }
def inner_tongue_objects = getAnnotationObjects().findAll {it.getPathClass() == getPathClass("Inner Tongue")}

println "Comparing ${fibre_objects.size()} objects vs ${inner_tongue_objects.size()} objects"
assessIoO2A (fibre_objects, inner_tongue_objects)

println "Comparing ${fibre_objects.size()} objects vs ${axon_objects.size()} objects"
assessIoO2A (fibre_objects, axon_objects)