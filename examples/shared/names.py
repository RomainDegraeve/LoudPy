"""
names.py — the HPNEW geometry's group names before and after the readable renaming (loudpy 0.3).

Results stored before it (the Results/Files of examples 01–09) still use the old names; the plot scripts read them
with lp.load(path, rename=OLD_NAMES), which changes nothing for files written with the new names.
"""
OLD_NAMES = {
    # mesh groups (the geometry's physical groups)
    "submeca_membranne": "cone", "submeca_dustcap": "dustcap", "submeca_surround": "surround",
    "submeca_former": "former", "submeca_coil": "coil", "submeca_glue_1": "glue_1", "submeca_glue_2": "glue_2",
    "submeca_spider": "spider", "subacou_front": "air_front", "subacou_rear": "air_rear",
    "interface_constrained_r_1": "axis", "interface_constrained_rz_1": "spider_rim",
    "interface_constrained_rz_2": "surround_rim", "interface_forced_z_1": "coil_drive",
    "interface_acou_meca_front": "wet_front", "interface_acou_meca_rear": "wet_rear",
    "PML_r+_1_front": "PML_front_r", "PML_z+_1_front": "PML_front_z", "PML_r+_2_rear": "PML_rear_r",
    "PML_z-_2_rear": "PML_rear_z",
    # item names of the examples' models (boundaries of the results)
    "membranne": "cone", "interface_constrained_rz": "rims", "interface_constrained_r": "axis",
    "interface_forced": "coil_drive", "subacou": "air",
}
