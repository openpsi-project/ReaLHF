from reallm.profiler.device_mesh import *


def find_all_sub_device_meshes(device_mesh: DeviceMesh):
    assert device_mesh.n_gpus_per_node == 8
    # if device_mesh.n_nodes == 1:
    #     return [
    #         DeviceMesh(device_mesh_name=f"{device_mesh.node_names[0]}:"
    #                    f"{gpu_id},{gpu_id+1},{gpu_id+2},{gpu_id+3}",
    #                    n_nodes=1,
    #                    n_gpus=4,
    #                    node_names=device_mesh.node_names,
    #                    gpu_ids=[gpu_id + i for i in range(4)]) for gpu_id in [0, 4]
    #     ] + [device_mesh]
    if device_mesh.n_gpus == 1:
        return [device_mesh]
    elif device_mesh.n_gpus == 2:
        return [
            DeviceMesh(
                device_mesh_name=f"{device_mesh.node_names[0]}:{gpu_id}",
                n_nodes=1,
                n_gpus=1,
                node_names=device_mesh.node_names,
                gpu_ids=[gpu_id],
            ) for gpu_id in device_mesh.gpu_ids
        ] + [device_mesh]
    elif device_mesh.n_gpus == 4:
        return ([
            DeviceMesh(
                device_mesh_name=f"{device_mesh.node_names[0]}:{gpu_id},{gpu_id+1}",
                n_nodes=1,
                n_gpus=2,
                node_names=device_mesh.node_names,
                gpu_ids=[gpu_id, gpu_id + 1],
            ) for gpu_id in device_mesh.gpu_ids[:-1]
        ] + [
            DeviceMesh(
                device_mesh_name=f"{device_mesh.node_names[0]}:{gpu_id}",
                n_nodes=1,
                n_gpus=1,
                node_names=device_mesh.node_names,
                gpu_ids=[gpu_id],
            ) for gpu_id in device_mesh.gpu_ids
        ] + [device_mesh])
    elif device_mesh.n_gpus == 8:
        return ([
            DeviceMesh(
                device_mesh_name=f"{device_mesh.node_names[0]}:"
                f"{gpu_id},{gpu_id+1},{gpu_id+2},{gpu_id+3}",
                n_nodes=1,
                n_gpus=4,
                node_names=device_mesh.node_names,
                gpu_ids=[gpu_id + i for i in range(4)],
            ) for gpu_id in [0, 4]
        ] + [
            DeviceMesh(
                device_mesh_name=f"{device_mesh.node_names[0]}:{gpu_id},{gpu_id+1}",
                n_nodes=1,
                n_gpus=2,
                node_names=device_mesh.node_names,
                gpu_ids=[gpu_id, gpu_id + 1],
            ) for gpu_id in [0, 2, 4, 6]
        ] + [
            DeviceMesh(
                device_mesh_name=f"{device_mesh.node_names[0]}:{gpu_id}",
                n_nodes=1,
                n_gpus=1,
                node_names=device_mesh.node_names,
                gpu_ids=[gpu_id],
            ) for gpu_id in device_mesh.gpu_ids
        ] + [device_mesh])

    # single node meshes
    res = []
    for node in device_mesh.node_names:
        res += find_sub_device_meshes(
            DeviceMesh(
                device_mesh_name=f"{node}",
                n_nodes=1,
                n_gpus=8,
                node_names=[node],
                gpu_ids=list(range(8)),
            ))

    # multi-node meshes
    node_ids = sorted([parse_node_id(node) for node in device_mesh.node_names])
    for i in range(2, device_mesh.n_nodes):
        for j in range(device_mesh.n_nodes - i + 1):
            sub_mesh_node_ids = node_ids[j:j + i]
            node_names = [f"QH-com{node_id:02d}" for node_id in sub_mesh_node_ids]
            res.append(
                DeviceMesh(
                    device_mesh_name=slurm_nodelist_from_nodes(node_names),
                    n_nodes=i,
                    n_gpus=8 * i,
                    node_names=node_names,
                    gpu_ids=list(range(8)),
                ))

    res += [device_mesh]
    return res


def find_three_integer_decomposition(n):
    decomps = []
    for x in range(1, n + 1):
        for y in range(x, n + 1):
            z = n / (x * y)
            if n % (x * y) == 0:
                decomps.append((x, y, int(z)))
    return decomps


if __name__ == "__main__":
    total = 0
    dm = make_device_mesh_from_name("QH-com[01-08]")
    device_meshes = find_all_sub_device_meshes(dm)
    for device_mesh in device_meshes:
        decomps = find_three_integer_decomposition(device_mesh.n_gpus)
        print(device_mesh.device_mesh_name, len(decomps))
        total += len(decomps)
    print(total)
