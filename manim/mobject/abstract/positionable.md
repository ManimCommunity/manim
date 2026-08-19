# Positionable

## TODO

- Handling for 0 points?
- Documentation

## Changes

|              Attribute              |                           Description                            | Speed |
| :---------------------------------: | :--------------------------------------------------------------: | :---: |
|          `align_on_border`          |       made `buff` keyword-only<br>added `frame` parameter        | 1.51x |
|             `align_to`              |                  made `direction` keyword-only                   | 0.72x |
|       `apply_array_function`        |     new<br>renamed from `apply_points_function_about_point`      |   -   |
|      `apply_complex_function`       |                                -                                 |   -   |
|          `apply_function`           |                                -                                 |   -   |
|    `apply_function_to_position`     | deprecated<br>use `move_to(function(self.get_center()))` instead |   -   |
|           `apply_matrix`            |                                -                                 |   -   |
| `apply_points_function_about_point` |         deprecated - use `apply_array_function` instead          |   -   |
|              `center`               |                                -                                 | 2.08x |
|               `depth`               |               deprecated - use `get_depth` instead               | 0.37x |
|               `depth`               |               deprecated - use `set_depth` instead               | 1.01x |
|               `flip`                |                                -                                 | 1.35x |
|            `get_bottom`             |                                -                                 |       |
|        `get_boundary_point`         |                                -                                 |       |
|         `get_bounding_box`          |                               new                                |   -   |
|            `get_center`             |                                -                                 |       |
|        `get_center_of_mass`         |                                -                                 |       |
|             `get_coord`             |                                -                                 |       |
|            `get_corner`             |          deprecated - use `get_critical_point` instead           |       |
|        `get_critical_point`         |                                -                                 |       |
|             `get_depth`             |                  new - replacement for `depth`                   |       |
|           `get_dim_size`            |                               new                                |       |
|          `get_edge_center`          |          deprecated - use `get_critical_point` instead           |       |
|      `get_extremum_along_dim`       |                            deprecated                            |       |
|            `get_height`             |                  new - replacement for `height`                  |       |
|             `get_left`              |                                -                                 |       |
|             `get_nadir`             |                                -                                 |       |
|             `get_right`             |                                -                                 |       |
|              `get_top`              |                                -                                 |       |
|             `get_width`             |                  new - replacement for `width`                   |       |
|               `get_x`               |                                -                                 |       |
|               `get_y`               |                                -                                 |       |
|               `get_z`               |                                -                                 |       |
|            `get_zenith`             |                                -                                 |       |
|              `height`               |              deprecated - use `get_height` instead               |       |
|              `height`               |              deprecated - use `set_height` instead               |       |
|           `is_off_screen`           |                                -                                 |       |
|          `length_over_dim`          |             deprecated - use `get_dim_size` instead              |       |
|            `match_coord`            |                  made `direction` keyword-only                   |       |
|            `match_depth`            |                       made kwargs explicit                       |       |
|          `match_dim_size`           |                       made kwargs explicit                       |       |
|           `match_height`            |                       made kwargs explicit                       |       |
|           `match_points`            |               removed `copy_submobjects` parameter               |       |
|            `match_width`            |                                                                  |       |
|              `match_x`              |                                                                  |       |
|              `match_y`              |                                                                  |       |
|              `match_z`              |                                                                  |       |
|              `move_to`              |                                                                  |       |
|              `next_to`              |                                                                  |       |
|           `pose_at_angle`           |                                                                  |       |
|      `reduce_across_dimension`      |                                                                  |       |
|          `rescale_to_fit`           |                                                                  |       |
|              `rotate`               |                                                                  |       |
|        `rotate_about_origin`        |                                                                  |       |
|               `scale`               |                                                                  |       |
|           `scale_to_fit`            |                              (new)                               |       |
|        `scale_to_fit_depth`         |                                                                  |       |
|        `scale_to_fit_height`        |                                                                  |       |
|        `scale_to_fit_width`         |                                                                  |       |
|             `set_coord`             |                                                                  |       |
|             `set_depth`             |                                                                  |       |
|           `set_dim_size`            |                              (new)                               |       |
|            `set_height`             |                                                                  |       |
|             `set_width`             |                                                                  |       |
|               `set_x`               |                                                                  |       |
|               `set_y`               |                                                                  |       |
|               `set_z`               |                                                                  |       |
|               `shift`               |                                                                  |       |
|         `shift_onto_screen`         |                                                                  |       |
|              `stretch`              |                                                                  |       |
|        `stretch_about_point`        |                                                                  |       |
|          `stretch_to_fit`           |                              (new)                               |       |
|       `stretch_to_fit_depth`        |                                                                  |       |
|       `stretch_to_fit_height`       |                                                                  |       |
|       `stretch_to_fit_width`        |                                                                  |       |
|             `to_corner`             |                                                                  |       |
|              `to_edge`              |                                                                  |       |
|               `width`               |                                                                  |       |

## Hierarchy

## Testing

Tries to ensure the same behavior for mobjects with at least 1 point by randomized testing.
Run`python test.py` to run the randomized tests.

### Pseudo Code

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
