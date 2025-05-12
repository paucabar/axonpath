// Get the current image data
def imageData = getCurrentImageData()

// Get all annotations
def annotations = getAnnotationObjects()

// Loop through annotations and rename classes as needed
annotations.each { annotation ->
    def pathClass = annotation.getPathClass()
    if (pathClass != null) {
        switch (pathClass.getName()) {
            case 'Outer':
                annotation.setPathClass(getPathClass('Fibre'))
                break
            case 'Inner':
                annotation.setPathClass(getPathClass('Inner Tongue'))
                break
            case 'Tile':
                annotation.setPathClass(null) // Remove class
                break
        }
    }
}

// Fire an event to update the display
fireHierarchyUpdate()
print "Annotation class update complete."
