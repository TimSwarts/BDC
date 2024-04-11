#!/usr/bin/env python3

"""
Assignment 2: Big Data Computing
Made in python 3.10.8
Calculate average PHRED scores over the network.
Usage:
    server mode:
        assignment2.py -s rnaseqfile.fastq --host <a workstation> --port <a port> --chunks <some integer>
    client mode:
        assignment2.py -c --host <server host> --port <server port> --n <number of cpus in client pc>
"""

__author__ = "Tim Swarts"
__version__ = "1.0"

import multiprocessing as mp
from multiprocessing.managers import BaseManager
import time
import queue
from pathlib import Path
import argparse
import sys

ASSIGNMENT1_PATH = str(Path(__file__).parent.parent.joinpath("Assignment1"))
sys.path.append(ASSIGNMENT1_PATH)
from assignment1 import get_chunks, post_processing, phred_sum_parser

AUTHKEY = b"brawlhallaspielen"
POISONPILL = "KO"
ERROR = "You fucked up"


def argument_parser() -> argparse.Namespace:
    """
    Argument parser for the command line arguments.
    :return argparse.parse_args(): This is an object with the command line arguments.
    Use .[argument] to get the arguments back by name.
    """

    description = "Script for Assignment 2 of Big Data Computing; Calculate PHRED scores over the network."
    argparser = argparse.ArgumentParser(description=description)

    mode = argparser.add_mutually_exclusive_group(required=True)

    server_help = "Run the program in Server mode; see extra options needed below"
    mode.add_argument("-s", action="store_true", help=server_help)

    client_help = "Run the program in Client mode; see extra options needed below"
    mode.add_argument("-c", action="store_true", help=client_help)

    # Arguments when run in server mode
    server_args = argparser.add_argument_group(title="Server mode arguments")

    csvfile_help = (
        "CSV file to save the output in. Default is output to terminal STDOUT"
    )
    server_args.add_argument(
        "-o",
        action="store",
        dest="csvfile",
        type=Path,
        required=False,
        help=csvfile_help,
    )

    fastq_files_help = "At least 1 Illumina Fastq Format file to process"
    server_args.add_argument(
        "fastq_files",
        action="store",
        type=Path,
        nargs="*",
        help=fastq_files_help,
    )

    chunks_help = "Number of chunks to split the files into. Default is 1"
    server_args.add_argument(
        "--chunks",
        action="store",
        required=False,
        type=int,
        help=chunks_help,
        default=1,
    )

    # Arguments when run in client mode
    client_args = argparser.add_argument_group(title="Client mode arguments")

    cores_help = "Number of cores to use per host."
    client_args.add_argument(
        "-n",
        action="store",
        dest="client_cores",
        required=False,
        type=int,
        help=cores_help,
    )

    # Global arguments
    hostname_help = "The hostname where the Server is listening"
    argparser.add_argument("--host", action="store", type=str, help=hostname_help)

    port_help = "The port on which the Server is listening"
    argparser.add_argument("--port", action="store", type=int, help=port_help)

    return argparser.parse_args()


class Client(mp.Process):
    """
    This is the client process, it connects to the server and starts the workers.
    :param host: The host name of the server.
    :param port: The port of the server.
    :param cores: The number of cores to use on this client.
    """

    def __init__(self, *, host: str, port: str, cores: int):
        """
        Initialize the client process.
        :param host: The host name of the server.
        :param port: The port of the server.
        :param cores: The number of cores to use.
        """
        super().__init__()
        self.host: str = host
        self.port: str = port
        self.cores: int = cores

    def run(self) -> None:
        """
        Start the client process, connects to the manager server and starts the workers.
        This method is called when the process is started with the start() method,
        therefore, calling it directly is not necessary.
        """
        # Connect to the manager server and fetch the queues
        manager = self.__make_client_manager()
        job_queue = manager.get_job_q()
        result_queue = manager.get_result_q()

        # Start the workers and pass the queues along
        self.__run_peons(job_queue, result_queue)

    def __run_peons(self, job_q: 'queue.Queue', result_q: 'queue.Queue') -> None:
        """
        This runs the peons, which are the workers that do the actual work,
        they are run in parallel as mp.Process objects.
        :param job_q: The queue that contains the jobs to be done.
        :param result_q: The queue that contains the results of the jobs.
        """
        # Intialize the list of processes
        processes = []
        # Create the processes for every core available and start them.
        for _ in range(self.cores):
            worker = Peon(job_q=job_q, result_q=result_q)
            processes.append(worker)
            # Start the worker
            worker.start()
        # Tell the user that the workers are running.
        print(f"{self.cores} peon workers are now running on this client!")

        # Wait for all the workers to finish.
        for process in processes:
            process.join()

    def __make_client_manager(self) -> BaseManager:
        """
        Create a manager for a client. This manager connects to a server on the
        given address and exposes the get_job_q and get_result_q methods for
        accessing the shared queues from the server.
        """

        # Create a custom manager class to register the queues with.
        class ClientQueueManager(BaseManager):
            """Custom manager class for connecting to the server."""

        ClientQueueManager.register("get_job_q")
        ClientQueueManager.register("get_result_q")

        manager = ClientQueueManager(address=(self.host, self.port), authkey=AUTHKEY)
        manager.connect()

        print(f"Client connected to {self.host}:{self.port}")
        return manager


class Server(mp.Process):
    """
    This is the server process, it listens for clients and distributes jobs to them.
    At the end it collects the results and writes them to a file or the terminal.
    :param target_function: The function that the workers should run.
    :param data: The data that should be processed.
    :param host: The host name of the server.
    :param port: The port of the server.
    :param ouputfile: The file to write the output to.
    """

    def __init__(
        self,
        *,
        target_function,
        data: str,
        host: str,
        port: str,
        output_file: Path | None = None,
        origin_files: list[Path],
    ) -> None:
        """
        Initialize the server process.
        :param target_function: The function that the workers should run.
        :param data: The data that should be processed.
        :param host: The host name of the server.
        :param port: The port of the server.
        :param ouputfile: The file to write the output to.
        :param origin_files: The files that were used to generate the data.
        """
        super().__init__()
        self.target_function = target_function
        self.data = data
        self.host = host
        self.port = port
        self.output_file = output_file
        self.origin_files = origin_files

    def run(self) -> None:
        """
        Start the server process, which creates a manager that distributes jobs to clients
        and collects the results.

        This method is called when the process is started with the start() method,
        therefore, calling it directly is not necessary.
        """
        # Start a shared manager server and access its queues
        with self.__make_server_manager() as manager:
            shared_job_q = manager.get_job_q()
            shared_result_q = manager.get_result_q()
            # Put the data in the job queue
            print(f"Sending data! {len(self.data)} chunks of it!")
            for data_chunk in self.data:
                shared_job_q.put({"target": self.target_function, "data": data_chunk})
            time.sleep(2)
            results = []
            result_counter = 1
            while True:
                try:
                    result = shared_result_q.get_nowait()
                    results.append(result)
                    print(f"Got a result! {result_counter}/{len(self.data)}")
                    result_counter += 1
                    if len(results) == len(self.data):
                        print("Got all results!")
                        break
                except queue.Empty:
                    time.sleep(1)
                    continue
            # Tell the client process no more data will be forthcoming
            print(
                "All data processed, time to seperate the workers from the means of production."
            )
            shared_job_q.put(POISONPILL)
            # Sleep a bit before shutting down the server - to give clients time to
            # realize the job queue is empty and exit in an orderly way.
            time.sleep(5)
            print("Server is done, only post processing left.")
        # Execute final post processing steps
        results_to_post_process = [job_dict["result"] for job_dict in results]
        post_processing(results_to_post_process, self.origin_files, self.output_file)

    def __make_server_manager(self) -> BaseManager:
        """Create a manager for the server, listening on the given port.
        Return a manager object with get_job_q and get_result_q methods.
        """
        # Create the job and result queues
        job_q = queue.Queue()
        result_q = queue.Queue()

        class QueueManager(BaseManager):
            """Custom manager class for hosting job en result"""

        QueueManager.register("get_job_q", callable=lambda: job_q)
        QueueManager.register("get_result_q", callable=lambda: result_q)

        manager = QueueManager(address=(self.host, self.port), authkey=AUTHKEY)
        manager.start()
        print(f"Server started at host: {self.host}:{self.port}")
        return manager


class Peon(mp.Process):
    """
    This is the worker process, it runs on a client and processes jobs from the job queue.
    :param job_q: The queue from which the worker gets jobs.
    :param result_q: The queue to which the worker puts results.
    """
    def __init__(self, job_q: 'queue.Queue', result_q: 'queue.Queue'):
        """
        Initialize the worker process.
        :param job_q: The queue from which the worker gets jobs.
        :param result_q: The queue to which the worker puts results.
        """
        super().__init__()
        self.job_q = job_q
        self.result_q = result_q

    def run(self) -> None:
        while True:
            try:
                # Get a job from the queue
                job = self.job_q.get_nowait()
                # If the job is the poison pill, put it back in the queue and exit the loop
                if job == POISONPILL:
                    self.job_q.put(POISONPILL)
                    print("Aaaaaaargh", self.name)
                    return
                # If the job is not the poison pill, do the job
                try:
                    # Fetch the target function and the data from the job
                    target_function = job["target"]
                    data_chunk = job["data"]
                    # Do the job
                    print(f"{self.name} working on {data_chunk}")
                    result = target_function(data_chunk)
                    # Put the result in the result queue
                    self.result_q.put({"job": job, "result": result})
                except NameError:
                    # If the target function is not found, put an error in the result queue
                    print("Peon name not found!")
                    self.result_q.put({"job": job, "result": ERROR})
            except queue.Empty:
                # If the queue is empty, sleep for a bit
                print(f"sleepytime for: {self.name}, goodnight!")
                time.sleep(1)


def main() -> None:
    """
    Main function of assignment 2 for Big Data Computing.
    """
    args = argument_parser()

    # server
    if args.s:
        # Read the fastq files
        data_chunks = get_chunks(args.fastq_files, args.chunks)

        # Start the server
        server = Server(
            target_function=phred_sum_parser,
            data=data_chunks,
            host=args.host,
            port=args.port,
            output_file=args.csvfile,
            origin_files=args.fastq_files,
        )
        server.start()
        time.sleep(1)
    # client
    if args.c:
        client = Client(host=args.host, port=args.port, cores=args.client_cores)
        client.start()
        time.sleep(1)
        client.join()
    return 0


if __name__ == "__main__":
    main()
