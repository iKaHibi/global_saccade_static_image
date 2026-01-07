import os
import ctypes
import numpy as np
import matplotlib.pyplot as plt
import tqdm

# Get the directory of the current Python file
current_file_dir = os.path.dirname(os.path.abspath(__file__))

class ElectricFilter:
    def __init__(self):
        parent_dir = os.path.dirname(current_file_dir)
        shared_lib_path = os.path.join(parent_dir, "libs", "photon2electric_all_omm.so")
        self.cuda_lib_path = shared_lib_path
        pass

    def filter_multi_omm_cuda(self, photon_data, photon_num_range=255):
        """
        using cuda shared lib to do the photon to electric transduction
        photon_data: numpy array, [time_steps][omm_num]
                the time_steps should have dt=1ms
                should be between 0-1
        """
        if np.max(photon_data) > 1.01 or np.min(photon_data) < -0.01:
            raise BaseException("The elements of photon_data should be in [0, 1]")
        photon_data = np.clip(photon_data, 0.0, 1.0)

        photon_input = np.round(photon_data.T * photon_num_range)
        omm_num, time_steps = photon_input.shape
        omm_num = int(omm_num)
        time_steps = int(time_steps)
        photon_input_resize = photon_input.reshape(int(omm_num * time_steps))
        photon_input_resize = np.array(photon_input_resize, dtype=np.float32)
        electric_output = np.zeros(int(omm_num * time_steps), dtype=np.float32)
        # Load the shared object file
        # print(f"using {self.cuda_lib_path}")
        lib = ctypes.CDLL(self.cuda_lib_path)  # Replace with the path to your .so file
        # Define the argument and return types for the function
        lib.get_all_omm.argtypes = [
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_int,
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_float),
            ]
        lib.get_all_omm.restype = None
        # Convert the Python lists or numpy arrays to C-compatible pointers
        photon_input_ptr = np.ctypeslib.as_ctypes(photon_input_resize)
        electric_output_ptr = np.ctypeslib.as_ctypes(electric_output)

        # Call the CUDA function
        # print("calling C extern funciton")
        lib.get_all_omm(photon_input_ptr, omm_num, time_steps, electric_output_ptr)

        # photon_input = np.ctypeslib.as_array(photon_input_ptr, shape=len(photon_input))
        electric_output = np.ctypeslib.as_array(electric_output_ptr, shape=len(electric_output))

        electric_output = electric_output.reshape([omm_num, time_steps])
        # print("processed data:", np.mean(electric_output))
        # start_t = 5
        # plt.plot(electric_output[0, :])
        # plt.show()

        return electric_output.T