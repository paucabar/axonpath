/*
 * This script is designed to extract morphometric features from myelinated fibre objects,
 * assigning the features directly to the parent "Fibre" objects to enable efficient
 * generation of a result table. It assumes the image has been processed with AimSeg 
 * detections, where objects are already organised into meaningful hierarchical relationships.
 * As part of the process, the script validates the detected objects, ensuring that 
 * only biologically relevant data is retained by removing invalid objects prior to quantification.
 */


/**
 * Some imports
 */

import static qupath.lib.gui.scripting.QPEx.*

import qupath.lib.roi.RoiTools
import java.lang.Math

/**
 * Define some methods
 */

/**
 * This function processes Fibre objects in QuPath by verifying and filtering their hierarchical relationships.
 * It identifies child objects up to two levels deep, ensuring that first-level children belong to the "Inner Tongue"
 * class and second-level children belong to the "Axon" class. Fibre objects that lack at least one "Inner Tongue"
 * child and one "Axon" descendant are removed from the hierarchy, along with their associated child objects.
 * 
 * The function retains only valid Fibre objects with meaningful relationships and removes invalid ones,
 * ensuring accurate and biologically relevant data organisation.
 */
 
Collection<PathObject> removeChildless() {
    // Get all Fibre objects
    def fibre_objects = getAnnotationObjects().findAll { it.getPathClass() == getPathClass("Fibre") }
    
    // Create a map to store Fibre objects and their child objects
    def validFibreToChildrenMap = [:]
    
    // Iterate through each Fibre object
    fibre_objects.each { fibre ->
        // Get the first level of child objects (filter by "Inner Tongue" class)
        def firstLevelChildren = fibre.getChildObjects().findAll { it.getPathClass() == getPathClass("Inner Tongue") }
    
        // Get the second level of child objects from first-level children (filter by "Axon" class)
        def secondLevelChildren = firstLevelChildren.collectMany { it.getChildObjects() }
                                                     .findAll { it.getPathClass() == getPathClass("Axon") }
    
        // Check if the Fibre object has at least one "Inner Tongue" child and one "Axon" child
        if (firstLevelChildren && secondLevelChildren) {
            // Combine the valid first and second-level children into a single list
            def allValidChildren = firstLevelChildren + secondLevelChildren
    
            // Store the Fibre object and its valid children in the map
            validFibreToChildrenMap[fibre] = allValidChildren
        }
    }
    
    // Output the results for debugging or further processing
    println "Valid Fibre objects and their children:"
    validFibreToChildrenMap.each { fibre, children ->
        println "Fibre: ${fibre.getName()} has ${children.size()} valid child objects"
    }
    
    // Remove invalid Fibre objects and their children
    def invalidFibreObjects = fibre_objects - validFibreToChildrenMap.keySet()
    println "Removing ${invalidFibreObjects.size()} invalid Fibre objects..."
    removeObjects(invalidFibreObjects, false) // true to remove chilfren
    
    return invalidFibreObjects
}

/**
 * This function calculates myelin metrics from QuPath objects organised in a hierarchical structure. 
 * The assumed hierarchy is as follows: Fibre > Inner Tongue > Axon. 
 * This means that each Fibre object contains one or more Inner Tongue objects, 
 * and each Inner Tongue object contains one or more Axon objects.
 * 
 * The function sums the areas of all Axon and Inner Tongue objects within a Fibre object 
 * to produce a single area measurement for each Fibre.
 * All area calculations take into account the image calibration, specifically the pixel size 
 * in square microns, ensuring accurate and meaningful results.
 */

void computeFeatures(imageData) {
    // Get calibration
    def pixelHeightMicrons = imageData.getServer().getPixelCalibration().getPixelHeightMicrons()
    def pixelWidthMicrons = imageData.getServer().getPixelCalibration().getPixelWidthMicrons()
    
    // Check if the height and width are the same
    if (pixelHeightMicrons != pixelWidthMicrons) {
        throw new IllegalArgumentException("The pixel calibration is not isotropic (Pixel Height: ${pixelHeightMicrons}, Pixel Width: ${pixelWidthMicrons}). This method requires isotropic XY pixels.")
    }
    
    // Use the pixel size (assuming isotropic calibration)
    def pixelSizeSquaredMicrons = pixelHeightMicrons * pixelWidthMicrons
    
    parentAnnotations = getAnnotationObjects().findAll{it.getPathClass() == getPathClass("Fibre")}
    def roiTools = new RoiTools()
    
    parentAnnotations.each { parent ->
    
        // define metrics
        float axon_area = 0
        float inner_region_area = 0
        float fibre_area = 0
        float axon_gratio = 0
        float myelin_gratio = 0
        int axon_objects = 0
        float fibre_circularity = 0
        float fibre_solidity = 0
    
        childAnnotations = parent.getChildObjects()
        childAnnotations.each { child ->
            inner_region_area += child.getROI().getArea() * pixelSizeSquaredMicrons
            grandchildAnnotations = child.getChildObjects()
            axon_objects = grandchildAnnotations.size()
            
            grandchildAnnotations.each { grandchild ->
                axon_area += grandchild.getROI().getArea() * pixelSizeSquaredMicrons
            }
        }
                
        // fibre metrics
        fibre_area = parent.getROI().getArea() * pixelSizeSquaredMicrons
        fibre_circularity = roiTools.getCircularity(parent.getROI())
        fibre_solidity = parent.getROI().getSolidity()
        
        // g-ratio netrics
        float fibre_diameter = 2 * Math.sqrt(fibre_area / Math.PI)
        float inreg_diameter = 2 * Math.sqrt(inner_region_area / Math.PI)
        float axon_diameter = 2 * Math.sqrt(axon_area / Math.PI)
        myelin_gratio = inreg_diameter / fibre_diameter
        axon_gratio = axon_diameter / fibre_diameter
        
        // add measurements
        parent.getMeasurementList().putMeasurement("Axon Area", axon_area)
        parent.getMeasurementList().putMeasurement("Inner Region Area", inner_region_area)
        parent.getMeasurementList().putMeasurement("Fibre Area", fibre_area)
        parent.getMeasurementList().putMeasurement("Axon g-ratio", axon_gratio)
        parent.getMeasurementList().putMeasurement("Myelin g-ratio", myelin_gratio)
        parent.getMeasurementList().putMeasurement("Axon Objects", axon_objects)
        parent.getMeasurementList().putMeasurement("Fibre Circularity", fibre_circularity)
        parent.getMeasurementList().putMeasurement("Fibre Solidity", fibre_solidity)
        
        return
    }
}




/**
 * Quantification pipeline
 */
 
 // get image data
 def imageData = getCurrentImageData()
 
 // Remove objects invalid for quantification
Collection<PathObject> invalidFibreObjects = removeChildless() // Storing invalid objects, could be useful for semi-automated annotation

// Feature extraction
computeFeatures(imageData)

return
