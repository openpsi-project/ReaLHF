import zmq

import base.name_resolve
import base.names
import base.network


class PipeScheduleController:
    """A controller for distributed pipeline module schedule. """

    def __init__(self, experiment_name, trial_name, model_name):
        """Specifies the name of the worker that WorkerControlPanel can used to find and manage.
        Args:
            worker_name: Typically "<worker_type>/<worker_index>".
        """
        self.__experiment_name = experiment_name
        self.__trial_name = trial_name
        self.__model_name = model_name

        host_ip = base.network.gethostip()
        port = base.network.find_free_port()

        self.__context = zmq.Context()
        self.__socket = self.__context.socket(zmq.PUB)
        self.__socket.bind(f"tcp://*:{port}")

        controller_name = base.names.model_controller(
            experiment_name,
            trial_name,
            model_name,
        )
        base.name_resolve.add(controller_name, f"{host_ip}:{port}", keepalive_ttl=30)
