from inspect import getmembers, ismethod
import manim as mn

custom_method_order = {
    'Mobject': {
        'Movement': [
            'align_on_border',
            'align_points',
            'align_points_with_larger',
            'align_to',
            'arrange',
            'arrange_in_grid',
            'arrange_submobjects',
            'center',
            'flip',
            'move_to',
            'next_to',
            'rotate',
            'rotate_about_origin',
            'rotate_in_place',
            'shift',
            'shift_onto_screen',
            'to_corner',
            'to_edge',
        ],
        'Position': [
            'get_coord',
            'get_corner',
            'get_critical_point',
            'get_depth',
            'get_edge_center',
            'get_end',
            'get_extremum_along_dim',
            'get_bottom',
            'get_boundary_point',
            'get_center',
            'get_center_of_mass',
            'get_width',
            'get_x',
            'get_y',
            'get_z',
            'get_z_index_reference_point',
            'set_coord',
            'set_depth',
            'set_height',
            'set_width',
            'set_x',
            'set_y',
            'set_z',
            'set_z_index',
            'set_z_index_by_z_coordinate',
            'get_height',
            'get_left',
            'get_nadir',
            'get_num_points',
            'get_pieces',
            'get_point_mobject',
            'get_points_defining_boundary',
            'get_right',
            'get_start',
            'get_start_and_end',
            'get_top',
            'get_zenith',
        ],
        'Transform': [
            'apply_complex_function',
            'apply_function',
            'apply_function_to_position',
            'apply_function_to_submobject_positions',
            'apply_matrix',
            'apply_over_attr_arrays',
            'apply_points_function_about_point',
            'apply_to_family',
            'become',
            'match_color',
            'match_coord',
            'match_depth',
            'match_dim_size',
            'match_height',
            'match_width',
            'match_x',
            'match_y',
            'match_z',
            'match_updaters',
            'scale',
            'scale_about_point',
            'scale_in_place',
            'rescale_to_fit',
            'stretch',
            'stretch_about_point',
            'stretch_in_place',
            'stretch_to_fit_depth',
            'stretch_to_fit_height',
            'stretch_to_fit_width',
            'fade',
            'fade_to',
        ],
        'Submobjects': [
            'add',
            'add_to_back',
            'remove',
            'get_family',
            'get_family_updaters',
            'add_background_rectangle_to_family_members_with_points',
            'add_background_rectangle_to_submobjects',
            'add_n_more_submobjects',
            'set_submobject_colors_by_gradient',
            'set_submobject_colors_by_radial_gradient',
            'sort_submobjects',
            'shuffle_submobjects',
            'space_out_submobjects',
            'align_submobjects',
            'family_members_with_points',
        ],
        'Style': [
            'get_color',
            'set_color',
            'set_color_by_gradient',
            'set_colors_by_radial_gradient',
            'add_background_rectangle',
            'to_original_color',
        ],
        'Updater': [
            'add_updater',
            'clear_updaters',
            'get_time_based_updaters',
            'get_updaters',
            'has_time_based_updater',
            'remove_updater',
            'resume_updating',
            'suspend_updating',
            'update',
        ],
        'Points': [
            'generate_points',
            'get_all_points',
            'has_points',
            'has_no_points',
            'reset_points',
            'reverse_points',
            'throw_error_if_no_points',
        ],
    }
}

for cls in custom_method_order:
    # add all methods not specified above to a 'Misc.' section
    obj = getattr(mn, cls)()
    all_methods = {
        m for m, _ in getmembers(obj, predicate=ismethod)
        if not m.startswith('_')
    }
    all_prev_methods = {
        m
        for section, items in custom_method_order[cls].items()
        for m in items
    }
    custom_method_order[cls]['Misc.'] = list(all_methods - all_prev_methods)

    # make sure sections do not have non-existent methods
    all_items = set(dir(getattr(mn, cls)))
    my_items = {m for section, items in custom_method_order[cls].items() for m in items}
    diff = my_items - all_items
    assert len(diff) == 0, f'Tried to include non-existent methods {diff}'

    # sort each section alphabetically
    for section in custom_method_order[cls]:
        custom_method_order[cls][section] = sorted(custom_method_order[cls][section])
