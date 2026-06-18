import gmsh
import os 


def prepare_geometry( step_path: str, brep_path: str, unit : str = "mm") -> None:
        """
        Step 1: Clean the STEP file and create the BREP for manual tagging.
        """
        if not os.path.exists(step_path):
            raise FileNotFoundError(f"The file {step_path} could not be found.")

        gmsh.initialize()
        gmsh.option.setNumber("General.Terminal", 1)
        gmsh.model.add("geometry_prep")

        gmsh.model.occ.importShapes(step_path)
        gmsh.model.occ.synchronize()
        gmsh.model.occ.removeAllDuplicates()
        
        entities = gmsh.model.occ.getEntities()
        # Scale the geometry (Conversion from mm to m)
        if unit == "m":
            pass
        elif unit == "mm":
            gmsh.model.occ.dilate(entities, 0, 0, 0, 0.001, 0.001, 0.001)
        elif unit == "inch":
            gmsh.model.occ.dilate(entities, 0, 0, 0, 0.0254, 0.0254, 0.0254)
        else:
            raise ValueError(f"Unsupported unit '{unit}'. Expected 'm', 'mm', 'inch'.")
        
        gmsh.model.occ.synchronize()

        gmsh.write(brep_path)
        gmsh.fltk.run()
        gmsh.finalize()

        print(f"Geometry cleaned and saved: {brep_path}")

