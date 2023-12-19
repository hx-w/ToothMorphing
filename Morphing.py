"""
A deformation method based on markers.
"""

from collections import defaultdict
from typing import Tuple, Dict, Set, Optional

import numpy as np
import tqdm
import scipy.sparse as sparse
import scipy.sparse.linalg as sparse_linalg
import trimesh

import meshlib

def compute_adjacent_by_edges(mesh: meshlib.Mesh):
    """Computes the adjacent triangles by using the edges"""
    candidates = defaultdict(set)  # Edge -> Faces
    for n, f in enumerate(mesh.faces):
        f0, f1, f2 = sorted(f)
        candidates[(f0, f1)].add(n)
        candidates[(f0, f2)].add(n)
        candidates[(f1, f2)].add(n)

    faces_adjacent: Dict[int, Set[int]] = defaultdict(set)  # Face -> Faces
    for faces in candidates.values():
        for f in faces:
            faces_adjacent[f].update(faces)

    faces_sorted = sorted([(f, [a for a in adj if a != f]) for f, adj in faces_adjacent.items()], key=lambda e: e[0])
    return [adj for f, adj in faces_sorted]


def get_aec(columns, rows):
    return sparse.identity(columns, dtype=float, format="csc")[:rows]


def get_bec(closest_points: np.array, verts: np.array):
    return verts[closest_points]


#########################################################
# Matrix builder for T Transformation entries

class TransformMatrix:
    __row_partial_baked = np.array([0, 1, 2] * 4)

    @classmethod
    def expand(cls, f: np.ndarray, inv: np.ndarray, size: int):
        i0, i1, i2, i3 = f
        col = np.array([i0, i0, i0, i1, i1, i1, i2, i2, i2, i3, i3, i3])
        data = np.concatenate([-inv.sum(axis=0), *inv])
        return sparse.coo_matrix((data, (cls.__row_partial_baked, col)), shape=(3, size), dtype=float)

    @classmethod
    def construct(cls, faces: np.ndarray, invVs: np.ndarray, size: int, desc="Building Transformation Matrix"):
        assert len(faces) == len(invVs)
        return sparse.vstack([
            cls.expand(f, inv, size) for f, inv in tqdm.tqdm(zip(faces, invVs), total=len(faces), desc=desc)
        ], dtype=float)


def apply_markers(A: sparse.spmatrix, b: np.ndarray, markers: dict) \
        -> Tuple[sparse.spmatrix, np.ndarray]:
    basemarker = np.array(list(markers.keys()))
    invmarker = np.setdiff1d(np.arange(A.shape[1]), basemarker)
    zb = b - A[:, basemarker] * np.array(list(markers.values()))
    return A[:, invmarker].tocsc(), zb


def revert_markers(A: sparse.spmatrix, x: np.ndarray, markers: dict,
                   *, out: Optional[np.ndarray] = None):
    if out is None:
        out = np.zeros((A.shape[1] + len(markers), 3))
    else:
        assert out.shape == (A.shape[1] + len(markers), 3)
    invmarker = np.setdiff1d(np.arange(len(out)), list(markers.keys()))
    out[invmarker] = x
    out[list(markers.keys())] = np.array(list(markers.values()))
    return out

#########################################################
# Identity Cost - of transformations


def construct_identity_cost(subject, invVs) -> Tuple[sparse.spmatrix, np.ndarray]:
    """ Construct the terms for the identity cost """
    AEi = TransformMatrix.construct(
        subject.faces, invVs, len(subject.vertices),
        desc="Building Identity Cost"
    ).tocsr()
    AEi.eliminate_zeros()

    Bi = np.tile(np.identity(3, dtype=float), (len(subject.faces), 1))
    assert AEi.shape[0] == Bi.shape[0]
    return AEi.tocsr(), Bi


#########################################################
# Smoothness Cost - of differences to adjacent transformations


def construct_smoothness_cost(subject, invVs, adjacent) -> Tuple[sparse.spmatrix, np.ndarray]:
    """ Construct the terms for the Smoothness cost"""
    count_adjacent = sum(len(a) for a in adjacent)
    size = len(subject.vertices)

    # Prebuild TransformMatrix for each face to reduce memory allocations
    transforms = [
        TransformMatrix.expand(f, inv, size).tocsr() for (f, inv) in
        tqdm.tqdm(zip(subject.faces, invVs), total=len(subject.faces), desc="Building TransformMatrices")
    ]

    def construct(index):
        a = transforms[index]
        for adj in adjacent[index]:
            yield a, transforms[adj]

    lhs, rhs = zip(*(adjacents for index in
                        tqdm.trange(len(subject.faces), desc="Building Smoothness Cost")
                        for adjacents in construct(index)))
    # Use compressed row format for subtraction
    AEs = (sparse.vstack(lhs).tocsr() - sparse.vstack(rhs).tocsr()).tocsc()

    # Cleanup & store in cache
    AEs.eliminate_zeros()

    Bs = np.zeros((count_adjacent * 3, 3))
    assert AEs.shape[0] == Bs.shape[0]
    return AEs, Bs


def compute_morphing(source_org: meshlib.Mesh, markers: np.ndarray) -> meshlib.Mesh:
    Ws = 1.0
    Wi = 0.001

    source = source_org.to_fourth_dimension()
    
    print("Precalculate adjacent list")

    # adjacent = compute_adjacent_by_vertices(source_org)
    adjacent = compute_adjacent_by_edges(source_org)

    #########################################################
    print("Inverse Triangle Spans")
    invVs = np.linalg.inv(source.span)
    assert len(source.faces) == len(invVs)

    #########################################################
    # Preparing the transformation matrices
    print("Preparing Transforms")
    # transforms = [TransformEntry(f, invV) for f, invV in zip(source.faces, invVs)]

    AEi, Bi = apply_markers(*construct_identity_cost(source, invVs), markers)

    AEs, Bs = apply_markers(*construct_smoothness_cost(source, invVs, adjacent), markers)

    #########################################################
    print("Building KDTree for closest points")
    # KDTree for closest points in E_c
    vertices: np.ndarray = np.copy(source.vertices)

    #########################################################
    # Start of loop

    iterations = 10
    total_steps = 2  # Steps per iteration

    # Progress bar
    pBar = tqdm.tqdm(total=iterations * total_steps)

    for iteration in range(iterations):

        def pbar_next(msg: str):
            pBar.set_description(f"[{iteration + 1}/{iterations}] {msg}")
            pBar.update()

        Astack = [AEi * Wi, AEs * Ws]
        Bstack = [Bi * Wi, Bs * Ws]

        #########################################################
        pbar_next("Combining Costs")

        A: sparse.spmatrix = sparse.vstack(Astack, format="csc")
        A.eliminate_zeros()
        b = np.concatenate(Bstack)

        #########################################################
        pbar_next("Solving")
        A = A.tocsc()

        # Calculate inverse markers for source
        assert A.shape[1] == len(vertices) - len(markers)
        assert A.shape[0] == b.shape[0]

        LU = sparse_linalg.splu((A.T @ A).tocsc())
        x = LU.solve(A.T @ b)

        # Reconstruct vertices x
        revert_markers(A, x, markers, out=vertices)

        riposta = meshlib.Mesh(
            vertices=vertices[:len(source_org.vertices)],
            faces=source_org.faces
        )
        
        return riposta


def parse_markers(src_file: str, markfile: str) -> dict: # { int: np.array }  source index -> target vertex
    template = trimesh.load(src_file)

    temp_pnts = []
    raw_pnts = []
    with open(markfile, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            pnts = list(map(lambda x : float(x), line.strip().split(',')))
            temp_pnts.append(np.array(pnts[:3]))
            raw_pnts.append(np.array(pnts[3:]))

    markers = dict(zip(template.nearest.vertex(temp_pnts)[1], raw_pnts))

    return markers

if __name__ == "__main__":
    '''
    Template file => src_path
    Markers file  => marker_path
    '''
    src_path = 'samples/template.obj'
    marker_path = 'samples/markers.csv'
    
    markers = parse_markers(src_path, marker_path)

    source_mesh = meshlib.Mesh.load(src_path)

    riposta_mesh = compute_morphing(source_mesh, markers)

    meshlib.Mesh.save_obj(riposta_mesh, "samples/result.obj")
