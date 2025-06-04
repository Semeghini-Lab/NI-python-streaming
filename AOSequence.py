import inspect
from functools import partial
import Commands
from Commands import *
from Instruction import *
import numpy as np

class Sequence:
    '''
    Base class for tracking instructions for a single channel. The instruction methods are
    programatically generated from Commands.py. During compilation, we check for conflicting instructions
    and fill any gaps (samples with no instructions) with the const instruction of the last value.
    '''

    # parse Commands.py once, the first time an object of this class is created
    _commands_loaded = False

    @classmethod
    def _load_commands(cls):
        if cls._commands_loaded:
            return
        # Find all commands for this category
        for name,obj in inspect.getmembers(Commands, inspect.isfunction):
            if hasattr(obj, '_category') and obj._category == cls._command_category:
                # Extract the function parameters
                params = list(filter(lambda x: x != 't', inspect.signature(obj).parameters.keys()))

                print(name, params)

                # Create a method for the command
                cls._create_class_method(name, obj, params)
        cls._commands_loaded = True

    @classmethod
    def _create_class_method(cls, func_name, command_func, param_names):
        '''
        func_name (str): name of the function
        command_func: reference to the function object in commands.py
        param_names: names of the kwargs for the function
        '''

        def method(self, t, duration, **kwargs):
            # do not allow instructions to be added if the sequence has been compiled
            if self.is_compiled:
                raise RuntimeError(f"Cannot add {func_name} instruction: sequence is already compiled. Create a new sequence or clear existing instructions.")
            
            # validating parameters
            if t < 0:
                raise ValueError(f"Start time must be >= 0, got {t} for {func_name}")
            if duration <= 1.0/self.sample_rate:
                raise ValueError(f"Duration must be > 1/sample_rate ({1.0/self.sample_rate} s), got {duration} for {func_name}")
            for param in param_names:
                if param not in kwargs:
                    raise ValueError(f"Missing required parameter '{param}' for {func_name}")
            
            # convert from real time to sample time
            start_sample = round(t * self.sample_rate)
            end_sample = start_sample + round(duration * self.sample_rate) # end sample is exclusive
            
            # create and append instruction
            instruction = Instruction(func=partial(command_func, **kwargs), start_sample=start_sample, end_sample=end_sample)
            self.instructions.append(instruction)
        
        # add method to this class
        setattr(cls, func_name, method)

    def __init__(self, channel_id, sample_rate, default_value=0.0, channel_name=""):
        """
        Initialize a sequence.
        
        Args:
            channel_id (str): Hardware identifier for the channel (e.g., "PXI1Slot3/ao0")
            sample_rate (int): Sample rate in Hz
            default_value: Default output value
            channel_name (str, optional): Name of the channel for operational use (e.g., "ao0", "do1")
        """
        self.sample_rate = sample_rate
        self.sample_rate = int(sample_rate)
        self.default_value = default_value
        self.instructions = []
        self.is_compiled = False
        self.channel_id = channel_id
        self.channel_name = channel_name
        
        self.__class__._load_commands()

    def compile(self, stopsamp):
        """Compile the sequence with the given stop sample.
        
        1. Sorts instructions by start sample
        2. Checks for overlaps between adjacent instructions
        3. Fills gaps with the last value
        
        Args:
            stopsamp: The stop sample for compilation
        """
        # if there are no instructions, fill the entire sequence with the default value
        if not self.instructions:
            default_instruction = Instruction(
                func=partial(const, value=self.default_value),
                start_sample=0,
                end_sample=int(stopsamp)
            )
            self.instructions = [default_instruction]
            self.is_compiled = True
            return
            
        # sort instructions by start sample
        self.instructions.sort(key=lambda x: x.start_sample)
        
        # check that adjacent samples do not overlap
        for i in range(len(self.instructions) - 1):
            current = self.instructions[i]
            next = self.instructions[i + 1]
            
            if current.end_sample > next.start_sample:
                current_func = current.func.func.__name__
                next_func = next.func.func.__name__
                
                current_start_time = current.start_sample / self.sample_rate
                current_end_time = current.end_sample / self.sample_rate
                next_start_time = next.start_sample / self.sample_rate
                next_end_time = next.end_sample / self.sample_rate
                raise ValueError(f"Instruction {current_func} ({current_start_time}s-{current_end_time}s) "
                               f"overlaps with {next_func} ({next_start_time}s-{next_end_time}s)")
        
        # fill gaps between instructions
        filled_instructions = []
        current_sample = 0
        last_value = self.default_value
        
        for instruction in self.instructions:
            # check if there is a gap
            if current_sample < instruction.start_sample:
                gap_instruction = Instruction(
                    func=partial(const, value=last_value),
                    start_sample=current_sample,
                    end_sample=instruction.start_sample
                )
                filled_instructions.append(gap_instruction)
            
            # copy over current instruction
            filled_instructions.append(instruction)
            current_sample = instruction.end_sample
            
            # compute the end value of the current instruction
            duration = (instruction.end_sample - instruction.start_sample) / self.sample_rate
            end_time_array = np.array([duration]) 
            func = instruction.func.func
            params = instruction.func.keywords
            last_value = func(end_time_array, **params)[0] 
        
        # check if there is a gap between last instruction and stop sample
        if current_sample < stopsamp:
            final_gap_instruction = Instruction(
                func=partial(const, value=last_value),
                start_sample=current_sample,
                end_sample=int(stopsamp)
            )
            filled_instructions.append(final_gap_instruction)
        
        self.instructions = filled_instructions
        self.is_compiled = True

    def clear(self):
        """Clear all instructions and reset compilation status."""
        self.instructions = []
        self.is_compiled = False


class AOSequence(Sequence):
    '''
    Tracks the instructions for a single analog channel.
    '''
    _command_category = 'analog_output'

    def __init__(self, channel_id, sample_rate, default_value=0.0, min_value=-10.0, max_value=10.0, channel_name=""):
        """
        Initialize an analog output sequence.
        
        Args:
            channel_id (str): Hardware identifier for the channel (e.g., "PXI1Slot3/ao0")
            sample_rate (int): Sample rate in Hz
            default_value (float): Default output value
            min_value (float): Minimum allowed output value
            max_value (float): Maximum allowed output value
            channel_name (str, optional): Name of the channel for operational use (e.g., "ao0")
        """
        super().__init__(channel_id, sample_rate, default_value, channel_name)
        self.min_value = min_value
        self.max_value = max_value


class DOSequence(Sequence):
    '''
    Tracks the instructions for a single digital channel.
    '''
    _command_category = 'digital_output'

    def __init__(self, channel_id, sample_rate, default_value=0, channel_name=""):
        """
        Initialize a digital output sequence.
        
        Args:
            channel_id (str): Hardware identifier for the channel (e.g., "PXI1Slot3/do0")
            sample_rate (int): Sample rate in Hz
            default_value (int): Default output value (0 or 1)
            channel_name (str, optional): Name of the channel for operational use (e.g., "do0")
        """
        super().__init__(channel_id, sample_rate, default_value, channel_name)


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    seq = AOSequence(channel_id="ao0", sample_rate=1e6)
    
    seq.const(1.0, 1.0, value=5.0) 
    seq.linramp(3.0, 1.0, start=0, end=10) 
    seq.sine(6.0, 0.75, freq=1, amp=2, phase=0)
    
    print(f"Before compilation - {len(seq.instructions)} instructions:")
    for i, instruction in enumerate(seq.instructions):
        start_time = instruction.start_sample / seq.sample_rate
        end_time = instruction.end_sample / seq.sample_rate
        func_name = instruction.func.func.__name__
        print(f"  {i}: {func_name} from {start_time}s to {end_time}s")
    
    seq.compile(stopsamp=8e6)  # 8 seconds at 1MHz
    
    print(f"\nAfter compilation - {len(seq.instructions)} instructions (gaps filled with const):")
    for i, instruction in enumerate(seq.instructions):
        start_time = instruction.start_sample / seq.sample_rate
        end_time = instruction.end_sample / seq.sample_rate
        func_name = instruction.func.func.__name__
        params = instruction.func.keywords
        print(f"  {i}: {func_name} from {start_time}s to {end_time}s, params: {params}")
    
    # Test overlap detection with improved error message
    print("\nTesting overlap detection:")
    seq_overlap = AOSequence(channel_id="ao0", sample_rate=1e6)
    seq_overlap.const(1.0, 2.0, value=5.0)  # 1s-3s
    seq_overlap.sine(2.5, 1.0, freq=1, amp=2, phase=0)  # 2.5s-3.5s (overlaps!)
    
    try:
        seq_overlap.compile(stopsamp=5e6)
        print("✗ ERROR: Overlapping instructions should have been detected!")
    except ValueError as e:
        print(f"✓ Overlap correctly detected: {e}")
    
    # Plot the compiled sequence
    import matplotlib.pyplot as plt
    
    # Create time array for the entire sequence
    total_samples = int(8e6)  # 8 seconds at 1MHz
    t_full = np.linspace(0, 8, total_samples)
    y_full = np.zeros(total_samples)
    
    # Fill in values from each instruction
    for instruction in seq.instructions:
        start_idx = instruction.start_sample
        end_idx = instruction.end_sample
        
        # Create time array for this instruction (relative to instruction start)
        n_samples = end_idx - start_idx
        t_instruction = np.linspace(0, (end_idx - start_idx) / seq.sample_rate, n_samples)
        
        # Evaluate the instruction function
        func = instruction.func.func
        params = instruction.func.keywords
        y_instruction = func(t_instruction, **params)
        
        # Fill the corresponding section of the full array
        y_full[start_idx:end_idx] = y_instruction
    
    # Plot the result
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(t_full, y_full, 'b-', linewidth=1)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Amplitude')
    ax.set_title('AOSequence - Compiled with Gap Filling')
    ax.grid(True, alpha=0.3)
    
    # Add vertical lines to show instruction boundaries
    for instruction in seq.instructions:
        start_time = instruction.start_sample / seq.sample_rate
        end_time = instruction.end_sample / seq.sample_rate
        func_name = instruction.func.func.__name__
        
        ax.axvline(start_time, color='red', linestyle='--', alpha=0.5)
        if func_name != 'const':  # Don't label every const boundary
            ax.text(start_time + 0.1, ax.get_ylim()[1] * 0.9, func_name, 
                   rotation=90, fontsize=8, alpha=0.7)
    
    plt.tight_layout()
    plt.show()
