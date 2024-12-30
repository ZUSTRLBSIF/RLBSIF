#Optional parameters. Readers can set the path by themselves according to their needs.


import tempfile

rna_opts = {}




rna_opts["out_replace_pdb_dir"] = ""
rna_opts["out_replace_decoypdb_dir"] = ""
rna_opts["tmp_dir"] = ""

rna_opts["decoy_rna_pdbqt"] = ""
rna_opts["decoy2_rna_pdbqt"] = ""


rna_opts["temp_mid"] = ""

rna_opts["decoy_pdbqt_to_pdbs"] = ""
rna_opts["decoy2_pdbqt_to_pdbs"] = ""

rna_opts["Ply"] = ""

rna_opts["decoy2_Out_Ply"] = ""
rna_opts["decoy_Out_Ply"] = ""
rna_opts["bind_Out_Ply"] = ""

rna_opts["ply_file_template"] = rna_opts["ply_chain_dir"] + "/{}.ply"
rna_opts["in_ply_file"] = rna_opts["ply_chain_dir"] + "/{}.ply"

rna_opts["Decoy_lable_touch"] = ""


rna_opts["compute_iface"] = True

rna_opts["mesh_res"] = 1.0
rna_opts["feature_interpolation"] = True


rna_opts["radius"] = 12.0

rna_opts["site"] = {}
rna_opts["site"]["max_shape_size"] = 100
rna_opts["site"]["n_conv_layers"] = 3
rna_opts["site"]["max_distance"] =12.0 
rna_opts["site"][
    "rna_precomputation_dir"
] = ""

rna_opts["site"][
    "bind_data"
] = "/"

rna_opts["site"][
    "decoy_data"
] = ""


rna_opts["site"][
    "decoy2_data"
] = ""


rna_opts["site"]["range_val_samples"] = 0.9  
rna_opts["site"]["model_dir"] = ""
rna_opts["site"]["out_pred_dir"] = ""




rna_opts["site"]["out_surf_dir"] = ""
rna_opts["site"]["feat_mask"] = [1.0] * 5


custom_params = {}
# predict rna_sit
custom_params["model_dir"] = ""
custom_params["feat_mask"] = [1.0, 1.0, 1.0, 1.0, 1.0]
custom_params["n_conv_layers"] = 3
custom_params["out_pred_dir"] = ""
custom_params["out_surf_dir"] = ""

