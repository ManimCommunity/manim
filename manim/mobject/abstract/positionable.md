# Positionable

## Notes
* How should mobject with 0 points be handled?
    * Currently: Treats behavior as undefined.
    * Advantage: Makes some calculations a lot simpler.
    * Disadvantage: Results in some breaking changes.
    * Consideration: Simplicity/Efficiency vs Guarding Exceptions.
* Should properties be dropped in favor of setter/getter methods?
    * E.g. `width`, `height` and `depth`.
    * Advantages: 
        * More in line with "manim-code-style".
        * Would allow method chaining.
        * Would allow additional optional parameters.
    * Alternative: 
        * Support both.
        * Disadvantage: 
            * Isn't really an actual value behind the scenes.
            * Other indirect attributes have setter methods.


## Hierarchy

* shift
    * move_to
        * align_on_border
            * to_corner
            * to_edge
        * align_to
        * center
        * set_coord
            * set_(x|y|z)
    * next_to (TODO: Implement using `move_to`)
* apply_array_function
    * apply_function
        * apply_complex_function
    * apply_matrix
        * rotate
            * flip
            * pose_at_angle
    * scale
        * scale_to_fit
            * scale_to_fit_(width|height|depth)
        * stretch
            * stretch_to_fit
                * stretch_to_fit_(width|height|depth)
* length_over_dim
    * get_(width|height|depth)
* get_bounding_box
    * get_critical_point (or get_corner, get_edge_center)
        * get_(center|bottom|top|left|right|nadir|zenith)
        * get_coord
            * get_(x|y|z)

# Deprecated
* width|height|depth = (set|get)_(width|height|depth)