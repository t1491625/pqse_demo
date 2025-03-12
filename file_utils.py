def convert_script_params_to_filename(script_params):
    # sort keys alphabetically
    sorted_keys = sorted(script_params.keys())
    s = "".join([k + "_" + str(script_params[k]) + "_" for k in sorted_keys])
    return s[:-1]  # get rid of trailing underscore
