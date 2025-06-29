import torch

from src.smplx_support.part_volume import PartVolume

SMPLX_PART_BOUNDS = './assets/smplx/part_meshes_ply/smplx_segments_bounds.pkl'
FID_TO_PART = './assets/smplx/part_meshes_ply/fid_to_part.pkl'
PART_VID_FID = './assets/smplx/part_meshes_ply/smplx_part_vid_fid.pkl'
HD_SMPLX_MAP  = './assets/smplx/smplx_neutral_hd_sample_from_mesh_out.pkl'

class SimpleCoM:
    def __init__(self, part_vid_fid, part_bounds, faces):
        """
        A minimal CoM class to compute volume-weighted center of mass without HD mapping.
        """
        self.part_vid_fid = part_vid_fid
        self.part_bounds = part_bounds
        self.faces = faces

    def compute_per_part_volume(self, vertices):
        """
        Compute the volume of each part in the reposed mesh
        """
        part_volume = []
        for part_name, part_bounds in self.part_bounds.items():
            # get part vid and fid
            part_vid = torch.LongTensor(self.part_vid_fid[part_name]['vert_id']).to(vertices.device)
            part_fid = torch.LongTensor(self.part_vid_fid[part_name]['face_id']).to(vertices.device)
            pv = PartVolume(part_name, vertices, self.faces)
            for bound_vids in part_bounds.values():
                pv.close_mesh(bound_vids)
            # add extra vids and fids to original part ids
            new_vert_ids = torch.LongTensor(pv.new_vert_ids).to(vertices.device)
            new_face_ids = torch.LongTensor(pv.new_face_ids).to(vertices.device)
            part_vid = torch.cat((part_vid, new_vert_ids), dim=0)
            part_fid = torch.cat((part_fid, new_face_ids), dim=0)
            pv.extract_part_triangles(part_vid, part_fid)
            part_volume.append(pv.part_volume())  # shape: [B,]
        return torch.stack(part_volume, dim=1)  # shape: [B, P]

    def compute_com(self, vertices):
        """
        Computes the volume-weighted center of mass from coarse SMPL-X vertices.
        """
        B, V, _ = vertices.shape
        device = vertices.device

        # Compute per-part volume
        per_part_volume = self.compute_per_part_volume(vertices)  # [B, P]

        # Now evenly distribute part volumes across the part's original vertices
        vertex_weights = torch.zeros(B, V, device=device)
        for i, (part_name, vol_column) in enumerate(zip(self.part_bounds.keys(), per_part_volume.transpose(0, 1))):
            vids = torch.LongTensor(self.part_vid_fid[part_name]['vert_id']).to(device)
            volume_per_vertex = vol_column / len(vids)
            vertex_weights[:, vids] += volume_per_vertex[:, None]  # [B, len(vids)]

        # Compute weighted mean
        weighted_vertices = vertices * vertex_weights[:, :, None]  # [B, V, 3]
        com = weighted_vertices.sum(dim=1) / vertex_weights.sum(dim=1, keepdim=True)  # [B, 3]
        return com

class CoM():
    def __init__(self, part_vid_fid, part_bounds, hdfy_op):
        """
        Initialize the CoM class with part vertex and face ids, part bounds, and hdfy operation.
        :param part_vid_fid: Dictionary containing vertex and face ids for each part.
        :param part_bounds: Dictionary containing bounds for each part.
        :param hdfy_op: Operation to convert mesh to high-density format.
        """
        self.part_vid_fid = part_vid_fid
        self.part_bounds = part_bounds
        self.hdfy_op = hdfy_op
        self.faces = None
        
    def compute_per_part_volume(self, vertices):
        """
        Compute the volume of each part in the reposed mesh
        """
        part_volume = []
        for part_name, part_bounds in self.part_bounds.items():
            # get part vid and fid
            part_vid = torch.LongTensor(self.part_vid_fid[part_name]['vert_id']).to(vertices.device)
            part_fid = torch.LongTensor(self.part_vid_fid[part_name]['face_id']).to(vertices.device)
            pv = PartVolume(part_name, vertices, self.faces)
            for bound_name, bound_vids in part_bounds.items():
                pv.close_mesh(bound_vids)
            # add extra vids and fids to original part ids
            new_vert_ids = torch.LongTensor(pv.new_vert_ids).to(vertices.device)
            new_face_ids = torch.LongTensor(pv.new_face_ids).to(vertices.device)
            part_vid = torch.cat((part_vid, new_vert_ids), dim=0)
            part_fid = torch.cat((part_fid, new_face_ids), dim=0)
            pv.extract_part_triangles(part_vid, part_fid)
            part_volume.append(pv.part_volume())
        return torch.vstack(part_volume).permute(1,0).to(vertices.device)
   
    def simple_vertex_volume_map(per_part_volumes, part_vid_fid, num_vertices):
        """
        per_part_volumes: [1, P] tensor of each part’s volume
        part_vid_fid:    dict mapping part→{'vert_id': [...], …}
        num_vertices:    total V in the mesh
        returns: [1, V] tensor of per-vertex weights summing to sum(parts)
        """
        device = per_part_volumes.device
        v_weights = torch.zeros(1, num_vertices, device=device)
        for i, (part, vol) in enumerate(zip(part_vid_fid.keys(), per_part_volumes[0])):
            vids = torch.LongTensor(part_vid_fid[part]['vert_id']).to(device)
            v_weights[0, vids] = vol / vids.numel()
        return v_weights  # shape [1, V]

    def vertex_id_to_part_volume_mapping(self, per_part_volume, device):
        batch_size = per_part_volume.shape[0]
        self.hd_vid_in_part = self.hd_vid_in_part.to(device)
        hd_vid_in_part = self.hd_vid_in_part[None, :, :].repeat(batch_size, 1, 1)
        vid_to_vol = torch.bmm(hd_vid_in_part, per_part_volume[:, :, None])
        return vid_to_vol

    def vertex_id_to_part_mapping(self, hd_vert_on_fid, fid_to_part):
        vid_to_part = torch.mm(hd_vert_on_fid, fid_to_part)
        return vid_to_part
    
 
    def forward(self, vertices):
        # Note: the vertices should be aligned along y-axis and in world coordinates
        batch_size = vertices.shape[0]
        # calculate per part volume
        per_part_volume = self.compute_per_part_volume(vertices)
        # sample 20k vertices uniformly on the smpl mesh
        vertices_hd = self.hdfy_op.hdfy_mesh(vertices)
        # get volume per vertex id in the hd mesh
        volume_per_vert_hd = self.vertex_id_to_part_volume_mapping(per_part_volume, vertices.device)
        # calculate com using volume weighted mean
        com = torch.sum(vertices_hd * volume_per_vert_hd, dim=1) / torch.sum(volume_per_vert_hd, dim=1)
        return com
        
        