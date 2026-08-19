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

Tries to ensure the same behavior for mobjects with at least 1 point by randomized testing.
Run `python test.py` to run the randomized tests.

## Pseudo Code
```py
function = lambda mob, kwargs: mob.some_function(**kwargs)  # function that you want to test

for point_count in (1, 100):                                # tests different point counts
    for _ in range(100):                                    # test every point count many times
        points = random_points(point_count)                 # generate points
        mob_old = Mobject().set_points(points)              # create old implementation
        mob_new = Positionable().set_points(points)         # create new implementation
        kwargs = random_parameters()                        # randomize parameters
        result_old = function(mob_old, kwargs)              # apply old implementation
        result_new = function(mob_new, kwargs)              # apply new implementation
        validate(mob_old, mob_new, result_old, result_new)  # compare results
```