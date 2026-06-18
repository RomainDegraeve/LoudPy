import numpy as np
from loudpy.Studies import Problem, FreqStudy
from loudpy import DomainSpecMecaRayleigh, DomainSpecMecaHysteretic, InterfaceSpecClamped, InterfaceSpecForced, DomainSpecAcou, DomainSpecPML, InterfaceSpecAcouMeca
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

geo_path = "LoudPy_exemples/Geometries/Loudspeaker/HPNEW-Sketch.geo"
msh_path = "LoudPy_exemples/Geometries/Loudspeaker/HPNEW-Sketch.msh"
mat_path = "src/loudpy/Materials_Bank/materials.json"
out_path = "LoudPy_exemples/Frequency_study/FSI/Results/Files/"


f_array = np.unique(np.logspace(np.log10(20), np.log10(20000), 320).astype(int))
c  = 344 # m/s
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


problem.add_sub_domain( DomainSpecAcou        ("subacou",   "Air"),)

problem.add_interface(InterfaceSpecClamped("interface_constrained"))
problem.add_interface(InterfaceSpecForced("interface_forced"))
problem.add_interface(InterfaceSpecAcouMeca ("interface_acou_meca_front"))
problem.add_interface(InterfaceSpecAcouMeca ("interface_acou_meca_rear"))

pml = DomainSpecPML("PML", "Air")
problem.add_sub_domain(pml)


for k, f in enumerate(f_array):
   
    lam = c / f
    pml.size, pml.f_pml, pml.t = lam/8, f, lam
    problem.set_mesh_sizes({"coil": 0.0015,"subacou": min(lam/6, 0.1),})
    problem.mesh(show_mesh_gui=False)

    # One study per frequency 
    study = FreqStudy(problem)
    study.assemble_domains()
    study.solve_fsi(freq = f, force = force) # records into _results

    fpath = out_path+f"snap_{k:04d}_f{f:.2f}Hz.h5"
    study.save(fpath, case="membrane_fsi_sweep", index=k, lam=lam)

   
    print(f"[{k+1}/{len(f_array)}]  f = {f:8.2f} Hz  →  {fpath}")




