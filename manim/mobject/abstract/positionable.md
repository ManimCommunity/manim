# Positionable

## Changes
> TODO

## TODO
* Handling for 0 points
* Documentation
* Helpful error messages 

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


# Testing

Tries to ensure that the behavior for mobjects with at least 1 point stays the same through randomized testing.
