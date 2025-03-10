import os
import sys
from abc import ABC, abstractmethod
from typing import Optional, Callable, Any

 

class Task(ABC):

    def __init__(self, 
                 task : Callable[[],Any], 
                 check_execution : Callable[[], bool] = lambda : False,
                 name : Optional[str] = None , 
                 cleanup : Optional[Callable[[],None]] = lambda : None) -> None:
        """
        task: a callable
        check_execution: a callable that checks if the task has already been executed
        name: string
        cleanup : a callable that removes artefact of task completion such that check_execution returns False
        """
        self.task = task
        self.check_execution = check_execution
        self.__cleanup = cleanup
        if name:
            self.name = name
        else:
            self.name = abs(hash(task))

    @abstractmethod
    def pre_task(self) -> None:
        """what to do before the task starts"""
        self.graceful_exit()
        raise NotImplementedError

    @abstractmethod
    def post_task(self) -> None:
        """What to do after the task successfully completed"""
        raise NotImplementedError
    
    @abstractmethod
    def handle_error(self) -> None:
        """Cleanup in case of error in task completion"""
        raise NotImplementedError
    
    @abstractmethod
    def graceful_exit(self) -> None:
        """Implements a graceful exit"""
        raise NotImplementedError

    def cleanup(self) -> None:
        self.__cleanup()

    def execute(self) -> Any:
        if self.check_execution():
            print(f"Task {self.name} already executed")
        else:
            try:
                print(f"Executing task {self.name}")
                self.pre_task()
                result = self.task()
                self.post_task()
                return result
            except KeyboardInterrupt:
                print(f"Task {self.name} interrupted", "cleaning up")
                self.handle_error()
                sys.exit(0)
            except Exception as e:
                print(f"Error running task {self.name}", "cleaning up")
                self.handle_error()
                print(e)


class SlurmTask(Task):
    """
    This subclasses uses lockfiles to ensure that no same task is running on different nodes.

    If your script has multiple tasks to run you can just launch the script multiple times on different nodes
    and each node will take care of a different task

    Lockfiles are stored in ./tmp
    """
    def __init__(self, 
                 task : Callable[[], Any], 
                 check_execution : Callable[[], bool] = lambda : False,
                 name : Optional[str] = None , 
                 cleanup : Optional[Callable[[], None]] = None
                 ) -> None:
        
        self.lockfile_path = f"./tmp/{name.replace("/", "")}"
        check = lambda : os.path.isfile(self.lockfile_path) or check_execution()
        super().__init__(task, check, name, cleanup)

    def pre_task(self) -> None:
        self.graceful_exit()
        #create the lockfile
        with open(self.lockfile_path, "w") as f:
            pass

    def post_task(self) -> None:
        if os.path.exists(self.lockfile_path):
            # Delete the lockfile
            os.remove(self.lockfile_path)

    def handle_error(self) -> None:
        self.post_task()

    def graceful_exit(self) -> None:
        """Handles slurm session timeout, the current implementation is not working"""

        # def handle_sigterm(signum, frame):
        #     print("Job termination detected (SIGTERM received). Cleaning tmp files")
        #     self.post_task()
        #     sys.exit(0)
        # signal.signal(signal.SIGTERM, handle_sigterm)
        pass # currently disabled as this conflicts with the benchmark process for some reason
