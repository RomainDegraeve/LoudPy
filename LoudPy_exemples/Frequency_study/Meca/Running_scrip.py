import numpy as np
from loudpy.Studies import Problem, FreqStudy
from loudpy import DomainSpecMecaRayleigh, DomainSpecMecaHysteretic, InterfaceSpecClamped, InterfaceSpecForced, DomainSpecAcou, DomainSpecPML, InterfaceSpecAcouMeca
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

geo_path = "LoudPy_exemples/Geometries/Loudspeaker/HPNEW-Sketch.geo"
msh_path = "LoudPy_exemples/Geometries/Loudspeaker/HPNEW-Sketch.msh"
mat_path = "src/loudpy/Materials_Bank/materials.json"
out_path = "LoudPy_exemples/Frequency_study/Meca/Results/Files/"


f_array = np.logspace(np.log10(20), np.log10(20000), 400)

force = 0.1 # N

problem = Problem(geo_path=geo_path, msh_path=msh_path,
                  mat_path=mat_path, subdomains_key="sub")


problem.add_sub_domain(
    DomainSpecMecaRayleigh("membranne",  material="Paper",           size=0.0005),
    DomainSpecMecaRayleigh("coil",       material="Copper",          size=0.0005),
    DomainSpecMecaRayleigh("surround",   material="Rubber",          size=0.0005),
    DomainSpecMecaRayleigh("spider",     material="PhenolicCloth",   size=0.0005),
    DomainSpecMecaRayleigh("former",     material="Kapton",          size=0.0005),
    DomainSpecMecaRayleigh("glue",       material="SolidGlue",       size=0.0005),
    DomainSpecMecaRayleigh("dustcap",    material="Polypropylene",   size=0.0005),
   
)


problem.add_interface(InterfaceSpecClamped("interface_constrained"))
problem.add_interface(InterfaceSpecForced("interface_forced"))
problem.add_interface(InterfaceSpecAcouMeca ("interface_acou_meca_front"))
problem.add_interface(InterfaceSpecAcouMeca ("interface_acou_meca_rear"))

problem.mesh(show_mesh_gui=False)
    # One study per frequency 
study = FreqStudy(problem)
study.assemble_meca()


for f in (f_array):
   
    study.solve_meca(freq = f, force = force)
   
study.save(out_path+f"snap_f_{int(f_array.min())}-{int(f_array.max())}Hz.h5", case="membrane_fsi_sweep")





