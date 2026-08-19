# Positionable

## TODO

- Handling for 0 points?
- Documentation

## Changes

|              Attribute              |                                                Description                                                 |
|:-----------------------------------:|:----------------------------------------------------------------------------------------------------------:|
|          `align_on_border`          |                            made `buff` keyword-only<br>added `frame` parameter                             |
|             `align_to`              |                                       made `direction` keyword-only                                        |
|       `apply_array_function`        |                          new<br>renamed from `apply_points_function_about_point`                           |
|      `apply_complex_function`       |                                                     -                                                      |
|          `apply_function`           |                                                     -                                                      |
|    `apply_function_to_position`     |                      deprecated<br>use `move_to(function(self.get_center()))` instead                      |
|           `apply_matrix`            |                                                     -                                                      |
| `apply_points_function_about_point` |                              deprecated - use `apply_array_function` instead                               |
|              `center`               |                                                     -                                                      |
|               `depth`               |                                    deprecated - use `get_depth` instead                                    |
|               `depth`               |                                    deprecated - use `set_depth` instead                                    |
|               `flip`                |                                                     -                                                      |
|            `get_bottom`             |                                                     -                                                      |
|        `get_boundary_point`         |                                                     -                                                      |
|         `get_bounding_box`          |                                                    new                                                     |
|            `get_center`             |                                                     -                                                      |
|        `get_center_of_mass`         |                                                     -                                                      |
|             `get_coord`             |                                                     -                                                      |
|            `get_corner`             |                               deprecated - use `get_critical_point` instead                                |
|        `get_critical_point`         |                                                     -                                                      |
|             `get_depth`             |                                       new - replacement for `depth`                                        |
|           `get_dim_size`            |                                                    new                                                     |
|          `get_edge_center`          |                               deprecated - use `get_critical_point` instead                                |
|      `get_extremum_along_dim`       |                                                 deprecated                                                 |
|            `get_height`             |                                       new - replacement for `height`                                       |
|             `get_left`              |                                                     -                                                      |
|             `get_nadir`             |                                                     -                                                      |
|             `get_right`             |                                                     -                                                      |
|              `get_top`              |                                                     -                                                      |
|             `get_width`             |                                       new - replacement for `width`                                        |
|               `get_x`               |                                                     -                                                      |
|               `get_y`               |                                                     -                                                      |
|               `get_z`               |                                                     -                                                      |
|            `get_zenith`             |                                                     -                                                      |
|              `height`               |                                   deprecated - use `get_height` instead                                    |
|              `height`               |                                   deprecated - use `set_height` instead                                    |
|           `is_off_screen`           |                                                     -                                                      |
|          `length_over_dim`          |                                  deprecated - use `get_dim_size` instead                                   |
|            `match_coord`            |                                       made `direction` keyword-only                                        |
|            `match_depth`            |                                            made kwargs explicit                                            |
|          `match_dim_size`           |                                            made kwargs explicit                                            |
|           `match_height`            |                                            made kwargs explicit                                            |
|           `match_points`            |                                    removed `copy_submobjects` parameter                                    |
|            `match_width`            |                                            made kwargs explicit                                            |
|              `match_x`              |                                       made `direction` keyword-only                                        |
|              `match_y`              |                                       made `direction` keyword-only                                        |
|              `match_z`              |                                       made `direction` keyword-only                                        |
|              `move_to`              |                              made `aligned_edge` and `coor_mask` keyword-only                              |
|              `next_to`              | implementation without submobject logic<br>made `direction`,`buff`,`aligned_edge`,`coor_mask` keyword-only |
|           `pose_at_angle`           |                                     deprecated<br>made kwargs explicit                                     |
|      `reduce_across_dimension`      |                                                     -                                                      |
|          `rescale_to_fit`           |                            made `stretch` keyword-only<br>made kwargs explicit                             |
|              `rotate`               |                                      removed unused kwargs parameter                                       |
|        `rotate_about_origin`        |                                     deprecated - use `rotate` instead                                      |
|               `scale`               |                                      supports scaling by a 3D vector                                       |
|           `scale_to_fit`            |                                                    new                                                     |
|        `scale_to_fit_depth`         |                                            made kwargs explicit                                            |
|        `scale_to_fit_height`        |                                            made kwargs explicit                                            |
|        `scale_to_fit_width`         |                                            made kwargs explicit                                            |
|             `set_coord`             |                                         made `direction` explicit                                          |
|             `set_depth`             |                                       new - replacement for `depth`                                        |
|           `set_dim_size`            |                                   new - replacement for `rescale_to_fit`                                   |
|            `set_height`             |                                       new - replacement for `height`                                       |
|             `set_width`             |                                       new - replacement for `width`                                        |
|               `set_x`               |                                       made `direction` keyword-only                                        |
|               `set_y`               |                                       made `direction` keyword-only                                        |
|               `set_z`               |                                       made `direction` keyword-only                                        |
|               `shift`               |                                   changed vararg `*vectors` to `vector`                                    |
|         `shift_onto_screen`         |                                            made kwargs explicit                                            |
|              `stretch`              |                                                     -                                                      |
|        `stretch_about_point`        |                                     deprecated - use `stretch` instead                                     |
|          `stretch_to_fit`           |                                                    new                                                     |
|       `stretch_to_fit_depth`        |                                            made kwargs explicit                                            |
|       `stretch_to_fit_height`       |                                            made kwargs explicit                                            |
|       `stretch_to_fit_width`        |                                            made kwargs explicit                                            |
|             `to_corner`             |                                          made `buff` keyword-only                                          |
|              `to_edge`              |                                          made `buff` keyword-only                                          |
|               `width`               |                                    deprecated - use `get_width` instead                                    |
|               `width`               |                                    deprecated - use `set_width` instead                                    |

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
