import scipy.io as spio
import numpy as np
import torch
import os

def compute_edge_index_from_faces(faces):
    """
    Compute the edge index from the faces of a 3D mesh using PyTorch tensors.
    
    Parameters:
        faces (torch.Tensor): A tensor of shape (F, 3), where F is the number of faces.
                              Each row contains the vertex indices of a triangular face.
    
    Returns:
        torch.Tensor: A tensor of shape (2, E), where E is the number of unique edges.
                      Each column represents an edge as a pair of vertex indices.
    """
    # Extract edges from faces
    edges = torch.cat([
        faces[:, [0, 1]],
        faces[:, [1, 2]],
        faces[:, [2, 0]]
    ], dim=0)
    
    # Sort each edge so that lower index comes first
    edges = torch.sort(edges, dim=1)[0]
    
    # Remove duplicate edges
    unique_edges = torch.unique(edges, dim=0)
    
    # Transpose to shape (2, E)
    edge_index = unique_edges.t()
    
    return edge_index

def face2edge(faces):
    edge_dict = {}
    
    for i in range(faces.shape[0]):
        a,b,c = faces[i,0],faces[i,1],faces[i,2]

        if a not in edge_dict: edge_dict[a] = set()
        if b not in edge_dict: edge_dict[b] = set()
        if c not in edge_dict: edge_dict[c] = set()

        edge_dict[a].add(b)
        edge_dict[a].add(c)

        edge_dict[b].add(a)
        edge_dict[b].add(c)

        edge_dict[c].add(a)
        edge_dict[c].add(b)
    return edge_dict

def generate_ico_orders(ico_mesh_files,output_ico_order_files,output_base_v_files):
    for ico_mesh_file,ico_order_file,base_v_file in zip(ico_mesh_files,
                                                         output_ico_order_files,
                                                         output_base_v_files):
        verts = spio.loadmat(ico_mesh_file)['vertices']
        faces = spio.loadmat(ico_mesh_file)['faces']
        base_vs,ico_orders = compute_ico_orders(verts,faces)
        spio.savemat(base_v_file,{'base_v':base_vs})
        spio.savemat(ico_order_file,{'adj_mat_order':ico_orders+1})
        
def compute_ico_orders(ic_verts,ic_faces):
    # assuming the verts of the icosahedron are zero centered ranging from -1 to 1
    # the base direction for each vetices in its tangent plane is the cross product of
    # the z-axis and the vertices coordinate vector
    # the order is determined to be clockwise w.r.t the base direction
    # Input: 
    #      ic_verts: numpy array of size N by 3. N is the number vertices in the icosahedron used
    #      ic_faces: numpy array of size M b y 3. M is the number triangular faces in the icosahedron used.
    #                ith row of ic_faces contains the index of 3 three vertices that make up ith triangle
    z_axis_array = np.zeros(ic_verts.shape) + np.asarray([0,0,1]).T

    # compute the reference vector for each vertex
    base_directions = np.cross(z_axis_array,ic_verts)
    base_directions /= (np.zeros(base_directions.shape).T + np.linalg.norm(base_directions,axis=1)).T
    base_directions[np.isnan(base_directions[:,0])==1,:] = np.asarray([[1,0,0],[1,0,0]])
    
    
    edge_dict = face2edge(ic_faces)
    ico_neighbor_orders = np.zeros((ic_verts.shape[0],7))
    for i in range(ic_verts.shape[0]):
        normal = ic_verts[i,:]/np.linalg.norm(ic_verts[i,:])
        x = -base_directions[i,:]
        y = np.cross(normal,x)
        
        neighbors_idx = np.asarray(list(edge_dict[i]))
        neighbors = ic_verts[neighbors_idx,:]-ic_verts[i,:]
        neighbor_x = np.dot(neighbors,x)
        neighbor_y = np.dot(neighbors,y)

        angles = -np.arctan2(neighbor_y,neighbor_x)
    
    
        ico_neighbor_orders[i,:len(neighbors)] = neighbors_idx[np.argsort(angles)]
        if neighbors.shape[0] == 5:
            ico_neighbor_orders[i,-2] = i 
        ico_neighbor_orders[i,-1] = i
    return base_directions,ico_neighbor_orders.astype(np.int32)


def get_upconv_index(adj_mat_order):  
    adj_mat_order = adj_mat_order -1
    nodes = len(adj_mat_order)
    next_nodes = int((len(adj_mat_order)+6)/4)
    upconv_top_index = np.zeros(next_nodes).astype(np.int64) - 1
    for i in range(next_nodes):
        upconv_top_index[i] = i * 7 + 6
    upconv_down_index = np.zeros((nodes-next_nodes) * 2).astype(np.int64) - 1
    for i in range(next_nodes, nodes):
        raw_neigh_order = adj_mat_order[i]
        parent_nodes = raw_neigh_order[raw_neigh_order < next_nodes]
        assert(len(parent_nodes) == 2)
        for j in range(2):
            parent_neigh = adj_mat_order[parent_nodes[j]]
            index = np.where(parent_neigh == i)[0][0]
            upconv_down_index[(i-next_nodes)*2 + j] = parent_nodes[j] * 7 + index
    
    return upconv_top_index, upconv_down_index

# 0,1,2,3,4,5,6,7 orders
level_vertices_nums= [12,42,162,642,2562,10242,40962,163842]

aux_data_dir = os.environ.get("ANATOMY_CORTEXDIFFUSION_AUX_DATA_DIR", os.path.dirname(__file__))
level_data_dict = {}

for i in range(len(level_vertices_nums)):
    level_data_dict[i] = {}
    mat_path = os.path.join(aux_data_dir, f"ic{i}.mat")
    if not os.path.exists(mat_path):
        raise FileNotFoundError(
            f"Missing aux mesh file: {mat_path}. "
            "Set `ANATOMY_CORTEXDIFFUSION_AUX_DATA_DIR` to a directory containing ic*.mat."
        )
    mat = spio.loadmat(mat_path)
    verts,faces = mat['vertices'],mat['faces']
    level_data_dict[i]['verts'] = torch.from_numpy(mat['vertices']).float()
    level_data_dict[i]['faces'] = torch.from_numpy(mat['faces'].astype(np.int32)).long()
    level_data_dict[i]['ei'] = compute_edge_index_from_faces(level_data_dict[i]['faces'])
    level_data_dict[i]['verts_num'] = level_vertices_nums[i]
    level_data_dict[i]['ico_order'] = compute_ico_orders(verts,
                                                         faces)[1]
    
    
    # print (i)
    # print (level_data_dict[i]['verts'].shape)
    # print (level_data_dict[i]['ei'].shape)
