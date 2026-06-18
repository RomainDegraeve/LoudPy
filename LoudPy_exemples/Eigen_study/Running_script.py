from loudpy.Studies import Problem, EigenStudy
from loudpy import (
    DomainSpecMecaRayleigh, DomainSpecMeca, DomainSpecMecaHysteretic,
    InterfaceSpecClamped, InterfaceSpecForced, InterfaceSpecAcouMeca 
)

geo_path = "LoudPy_exemples/Geometries/Loudspeaker/HPNEW-Sketch.geo"
msh_path = "LoudPy_exemples/Geometries/Loudspeaker/HPNEW-Sketch.msh"
mat_path = "src/loudpy/Materials_Bank/materials.json"
out_path = "LoudPy_exemples/Eigen_study/Results/Files/eigen_study.h5"

problem = Problem(geo_path=geo_path, msh_path=msh_path, mat_path=mat_path)

problem.add_sub_domain(
    DomainSpecMecaRayleigh("membranne",  material="Paper",           size=0.0005),
    DomainSpecMecaRayleigh("coil",       material="Copper",          size=0.0005),
    DomainSpecMecaRayleigh("surround",   material="Rubber",          size=0.0005),
    DomainSpecMecaRayleigh("spider",     material="PhenolicCloth",   size=0.0005),
    DomainSpecMecaRayleigh("former",     material="Kapton",          size=0.0005),
    DomainSpecMecaRayleigh("glue",       material="SolidGlue",       size=0.0005),
    DomainSpecMecaRayleigh("dustcap",    material="Polypropylene",   size=0.0005), 
)

problem.add_interface(InterfaceSpecClamped("interface_constrained"),
                      InterfaceSpecForced("interface_forced"), 
                      InterfaceSpecAcouMeca ("interface_acou_meca_front")
                      )



problem.mesh(show_mesh_gui=True, write_mesh_file=True)

study = EigenStudy(problem)

study.solve_meca_eigen_ARPAC(n_modes=45, f_target=1)

study.save(out_path) 
