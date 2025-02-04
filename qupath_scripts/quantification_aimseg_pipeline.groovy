/*
 * This script is designed to extract morphometric features from myelinated fibre objects,
 * assigning the features directly to the parent "Fibre" objects to enable efficient
 * generation of a result table. It assumes the image has been processed with AimSeg 
 * detections, where objects are already organised into meaningful hierarchical relationships.
 * As part of the process, the script validates the detected objects, ensuring that 
 * only biologically relevant data is retained by removing invalid objects prior to quantification.
 */


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
 * Quantification pipeline
 */
 
 // Remove objects invalid for quantification
Collection<PathObject> invalidFibreObjects = removeChildless() // Storing invalid objects, could be useful for semi-automated annotation
