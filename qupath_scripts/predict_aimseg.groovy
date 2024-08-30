/**
 * This script demonstrates how to run a model using DJL in QuPath.
 *
 * You should first install the DJL extension in QuPath, and download PyTorch -
 * see https://qupath.readthedocs.io/en/stable/docs/deep/djl.html
 */

import ij.ImagePlus
import qupath.lib.images.servers.PixelType
import qupath.lib.regions.Padding
import qupath.lib.regions.RegionRequest
import qupath.opencv.ops.ImageOps
import qupath.opencv.tools.OpenCVTools

import java.nio.file.Paths

import static qupath.lib.gui.scripting.QPEx.*
import qupath.ext.djl.DjlTools

def uri = Paths.get("/path/to/weights.pt").toUri()
def imageData = getCurrentImageData()

int inputWidth = 512
int inputHeight = inputWidth
int nChannels = 1
def padding = Padding.symmetric(32)
def layout = "NCHW"
def inputShape = [1, nChannels, inputHeight, inputWidth]
double downsample = 1.0

// Get an ImageJ representation of the output
ImagePlus impOutput

// Apply prediction
try (def dnn = DjlTools.createDnnModel(uri, layout, inputShape as int[])) {
    def op = ImageOps.buildImageDataOp()
        .appendOps(
                ImageOps.Core.ensureType(PixelType.FLOAT32),
                ImageOps.Normalize.percentile(0.1, 99.9),
                ImageOps.ML.dnn(dnn, inputWidth, inputHeight, padding)
        )

    // Use a selected annotation if we have one, otherwise request pixels for the full image
    def selectedObject = getSelectedObject()
    RegionRequest request
    def server = imageData.getServer()
    if (selectedObject != null && selectedObject.isAnnotation())
        request = RegionRequest.createInstance(server.getPath(), downsample, selectedObject.getROI())
    else
        request = RegionRequest.createInstance(server, downsample)

    // Run the prediction, getting an OpenCV Mat as output
    def mat = op.apply(imageData, request)

    // Convert to an ImageJ ImagePlus
    impOutput = OpenCVTools.matToImagePlus("Prediction", mat)
    mat.close()
}

impOutput.show()