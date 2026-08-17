# Positionable

## Notes
* How should mobject with 0 points be handled?
    * Currently: Treats behavior as undefined.
    * Advantage: Makes calculations simpler and more efficient.
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


## Progress