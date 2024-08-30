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

import java.nio.file.Paths

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
double downsample = 4.0

// Get an ImageJ representation of the output
ImagePlus impOutput

// Use a selected annotation if we have one, otherwise request pixels for the full image
def selectedObject = getSelectedObject()
RegionRequest request
def server = imageData.getServer()
if (selectedObject != null && selectedObject.isAnnotation())
    request = RegionRequest.createInstance(server.getPath(), downsample, selectedObject.getROI())
else
    request = RegionRequest.createInstance(server, downsample)

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

impOutput.show()