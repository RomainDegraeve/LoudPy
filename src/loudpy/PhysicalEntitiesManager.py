from loudpy.Meshing.MeshManager import MeshManager

class PhysicalEntitiesManager:
    def __init__(self, domains, interfaces, geo_path: str, msh_path: str,  mat_path : str, subdomains_key : str, directions_key : dict, orientations_key : dict):
        self.SubDomains = domains
        self.Interfaces = interfaces
        self.geo_path = geo_path
        self.msh_path = msh_path
        self.mat_path = mat_path
        self.subdomains_key = subdomains_key 
        self.directions_key = directions_key
        self.orientations_key = orientations_key
    def create_meshed_objs(self, show_gui, write_mesh_file):
        domains, boundaries, nodes  = MeshManager(self.SubDomains, self.subdomains_key).generate_mesh(self.geo_path , self.msh_path, write_mesh_file, show_gui=show_gui, extract_mesh_data=True)
    
        fem_objects = []
        for domain in domains:
            for sim_object in self.SubDomains:
                if sim_object.name in domain.name:
                    try:
                        params = sim_object.physics.get_params(sim_object.material, self.mat_path)
                    except (KeyError, FileNotFoundError):
                        params = {}
                    for key in sim_object.material_keys():
                        val = getattr(sim_object, key)
                        if val is not None:
                            params[key] = val
                    matched_keys = [k for k in self.orientations_key if k in domain.name]
                    if matched_keys:
                        key = matched_keys[0]
                        orientation = self.orientations_key[key]
                        
                        fem_objects.append(sim_object.physics(domain, orientation=orientation, **params)) #sim_object.physic_type is a class like subdomainacou ect ...
                    else :
                        fem_objects.append(sim_object.physics(domain, **params)) #sim_object.physic_type is a class like subdomainacou ect ...

        for boundary in boundaries:
            for sim_object in self.Interfaces:
                if sim_object.name in boundary.name:
                    value = None
                    for d in [self.directions_key, self.orientations_key]:
                        matched_key = next((k for k in d if k in boundary.name), None)
                        if matched_key:
                            value = d[matched_key]
                            break
                    
                    if value is not None:
                        fem_objects.append(sim_object.physics(boundary, value))
                    else:
                        fem_objects.append(sim_object.physics(boundary))

        return fem_objects,  nodes


       
       
