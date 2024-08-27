def imageData = getCurrentImageData()
def hierarchy = imageData.getHierarchy()

// Create output path (relative to project)
def outputDir = buildFilePath(PROJECT_BASE_DIR, 'export')
mkdirs(outputDir)

// Create output subfolders
def semanticDir = buildFilePath(outputDir, 'masks')
mkdirs(semanticDir)
def instanceDir = buildFilePath(outputDir, 'labels')
mkdirs(instanceDir)
def imageDir = buildFilePath(outputDir, 'images')
mkdirs(imageDir)

// Define how much to downsample during export
double downsample = 2

// Create an ImageServer where the pixels are derived from annotations
def semanticServer = new LabeledImageServer.Builder(imageData)
  .backgroundLabel(0, ColorTools.BLACK) // Specify background label (usually 0 or 255)
  .downsample(downsample)    // Choose server resolution; this should match the resolution at which tiles are exported
  .addLabel('Outer', 1)      // Choose output labels (the order matters!)
  .addLabel('Inner', 2)
  .addLabel('Axon', 3)
  .multichannelOutput(false) // If true, each label refers to the channel of a multichannel binary image (required for multiclass probability)
  .build()

// Create an ImageServer for fibre instances
def instanceServer = new LabeledImageServer.Builder(imageData)
  .backgroundLabel(0, ColorTools.BLACK) // Specify background label (usually 0 or 255)
  .downsample(downsample)    // Choose server resolution; this should match the resolution at which tiles are exported
  .useAnnotations()
  .useInstanceLabels()
  .useFilter(p -> p.isAnnotation() && p.getPathClass() == getPathClass('Outer'))
  .multichannelOutput(false) // If true, each label refers to the channel of a multichannel binary image (required for multiclass probability)
  .build()
  
// Create an ImageServer where raw images are downsampled
def server = imageData.getServer()
def region = RegionRequest.createInstance(server, downsample)


// Get all annotation objects and keep only the Tiles
def tileObjects = hierarchy.getAnnotationObjects().findAll{it.getPathClass() == getPathClass("Tile")}

print(tileObjects.size())


// Get image name to export annotations
def name = GeneralTools.getNameWithoutExtension(imageData.getServer().getMetadata().getName())

// Loop through tiles to write image regions
if(tileObjects.size()>0) {
    tileObjects.eachWithIndex{it,index->
        roi = it.getROI() // get tile roi
        // Define export paths
        def pathSemantic = buildFilePath(semanticDir, name + "_t" + index.toString() + ".tif") // Define semantic output file paths
        def pathInstance = buildFilePath(instanceDir, name + "_t" + index.toString() + ".tif") // Define instance output file paths
        def pathImage = buildFilePath(imageDir, name + "_t" + index.toString() + ".tif") // Define image output file path
        // Get tile reguin from each server
        def requestROISemantic = RegionRequest.createInstance(semanticServer.getPath(), downsample, roi)
        def requestROIInstance = RegionRequest.createInstance(instanceServer.getPath(), downsample, roi)
        def requestROIImage = RegionRequest.createInstance(server.getPath(), downsample, roi)
        // Write the images
        writeImageRegion(semanticServer, requestROISemantic, pathSemantic)
        writeImageRegion(instanceServer, requestROIInstance, pathInstance)
        writeImageRegion(server, requestROIImage, pathImage)
    }
// Write full image if there are no tiles
} else {
    // Define export paths
    def pathSemantic = buildFilePath(semanticDir, name + ".tif") // Define semantic output file paths
    def pathInstance = buildFilePath(instanceDir, name + ".tif") // Define instance output file paths
    def pathImage = buildFilePath(imageDir, name + ".tif") // Define image output file path
    // Write the images
    writeImage(semanticServer, pathSemantic)
    writeImage(instanceServer, pathInstance)
    writeImageRegion(server, region, pathImage)
}