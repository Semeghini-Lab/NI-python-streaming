import inspect
from functools import partial
import nistreamer.Commands as Commands
from nistreamer.Commands import *
from nistreamer.Instruction import *
import numpy as np
import bisect

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
                # If there is no _propagate_start_value attribute, make it False
                if not hasattr(obj, '_propagate_start_value'):
                    obj._propagate_start_value = False

                # If there is no _propagate_duration attribute, make it False
                if not hasattr(obj, '_propagate_duration'):
                    obj._propagate_duration = False

                # If there is no _inplace attribute, make it False
                if not hasattr(obj, '_inplace'):
                    obj._inplace = False

                # If there is no _instantaneous attribute, make it False
                if not hasattr(obj, '_instantaneous'):
                    obj._instantaneous = False
                else:
                    # Make sure that all instantaneous commands are digital
                    if obj._instantaneous and obj._category != 'digital_output':
                        raise ValueError(f"Instantaneous command {name} is not a digital output.")
                
                # Extract the function parameters and their default values
                if obj._inplace:
                    params = list(filter(lambda x: x[0] != 't' and x[0] != 'buf', inspect.signature(obj).parameters.items()))
                else:
                    params = list(filter(lambda x: x[0] != 't', inspect.signature(obj).parameters.items()))

                # Create a method for the command
                cls._create_class_method(name, obj, params)
        cls._commands_loaded = True

    @classmethod
    def _create_class_method(cls, func_name, command_func, params):
        '''
        func_name (str): name of the function
        command_func: reference to the function object in commands.py
        params: list of tuples (param_name, param_default_value)
        '''

        param_names = [param[0] for param in params]
        param_defaults = [param[1].default for param in params]

        def method(self, t, duration=None, **kwargs):
            # Do not allow instructions to be added if the sequence has been compiled
            if self.is_compiled:
                raise RuntimeError(f"Cannot add {func_name} instruction: sequence is already compiled. Create a new sequence or clear existing instructions.")
            
            # Check if the start time is non-negative
            if t < 0:
                raise ValueError(f"Start time must be >= 0, got {t} for {func_name}")
            
            # If duration is None, make sure that the command is instantaneous
            if duration is None:
                if not command_func._instantaneous:
                    raise ValueError(f"Command {func_name} is not instantaneous, but duration is None.")
                duration = 1.0/self.sample_rate
            
            # Check if the duration is more than a single sample
            if duration < 1.0/self.sample_rate:
                raise ValueError(f"Duration must be > 1/sample_rate ({1.0/self.sample_rate} s), got {duration} for {func_name}")
            
            # Check if the proper parameters are provided
            for p_id, p_name in enumerate(param_names):
                if p_name not in kwargs:
                    if param_defaults[p_id] is not inspect._empty:
                        kwargs[p_name] = param_defaults[p_id]
                    else:
                        raise ValueError(f"Missing required parameter '{p_name}' for {func_name}")
            # Convert from real time to sample time
            start_sample = round(t * self.sample_rate)
            end_sample = start_sample + round(duration * self.sample_rate) # end sample is exclusive

            # Ensure that instantaneous commands are exactly 1 sample long
            if command_func._instantaneous and end_sample - start_sample != 1:
                raise ValueError(f"Instantaneous command {func_name} must be exactly 1 sample long, got {end_sample - start_sample} samples.")
            
            # Create and append instruction
            instruction = Instruction(func=partial(command_func, **kwargs), start_sample=start_sample, end_sample=end_sample, inplace=command_func._inplace)
            self.instructions.append(instruction)
        
        # Add method to this class
        setattr(cls, func_name, method)

    def __init__(self, channel_id: str, sample_rate: int, default_value: float = 0.0, channel_name: str = ""):
        """
        Initialize a sequence.
        
        Args:
            channel_id (str): Hardware identifier for the channel (e.g., "PXI1Slot3/ao0")
            sample_rate (int): Sample rate in Hz
            default_value: Default output value
            channel_name (str, optional): Name of the channel for operational use (e.g., "ao0", "do1")
        """
        self.default_value = default_value
        self.instructions = []
        self.is_compiled = False
        self.channel_id = channel_id
        self.channel_name = channel_name

        # Make sure the sample rate is an integer
        if not isinstance(sample_rate, int):
            print(f"Sample rate must be an integer, got {sample_rate} in {self}. Rounding to the nearest integer {round(sample_rate)}.")
            sample_rate = round(sample_rate)
        
        self.sample_rate = sample_rate
        
        self.__class__._load_commands()

    def _compile_command(self, inst, last_value):
        # Calculate the total duration of the instruction
        duration = (inst.end_sample - inst.start_sample) / self.sample_rate

        # Check if the instruction needs to propagate the start value
        if inst.func.func._propagate_start_value:
            # Look for the 'start' keyword in the parameters and overlap it with the end value
            if 'start' not in inst.func.keywords:
                raise ValueError(f"Instruction {inst} needs to propagate the start value, but 'start' is not in the parameters.")
            
            # Update the start keyword to the last value
            inst.func.keywords['start'] = last_value

        # Check if the instruction needs to propagate the duration
        if inst.func.func._propagate_duration:
            # Look for the 'duration' keyword in the parameters and overlap it with the end value
            if 'cmd_duration' not in inst.func.keywords:
                raise ValueError(f"Instruction {inst} needs to propagate the duration, but 'cmd_duration' is not in the parameters.")
            
            # Update the duration keyword to the duration of the instruction
            inst.func.keywords['cmd_duration'] = duration

        # Evaluate the instruction at the end of the current instruction
        t = np.atleast_1d(duration)
        if inst.inplace:
            buf = np.zeros_like(t, dtype=self._buffer_dtype())
            inst.func.func(t, **inst.func.keywords, buf=buf)
            last_value = buf[0]
        else:
            last_value = inst.func.func(t, **inst.func.keywords)[0]
        return inst, last_value

    def _buffer_dtype(self):
        return np.float64 if isinstance(self, AOSequence) else bool

    def compile(self, chunk_size, num_chunks):
        """
        Compile the sequence with the given sample rate, chunk size, and stop sample.
        
        Compilation algorithm:
        1. Sorts instructions by start sample
        2. Checks for overlaps between adjacent instructions
        3. Fills gaps with the previous value
        
        Args:
            chunk_size: The chunk size for compilation
            num_chunks: The number of chunks for compilation (need to ensure same number of waveforms for all channels)
        """
        # Make sure the arguments are integers
        if not isinstance(chunk_size, int):
            raise ValueError(f"Chunk size must be an integer, got {chunk_size}")
        if not isinstance(num_chunks, int):
            raise ValueError(f"Number of chunks must be an integer, got {num_chunks}")

        # If there are no instructions, fill the entire sequence with the default value
        if not self.instructions:
            default_instruction = Instruction(
                func=partial(const, value=self.default_value),
                start_sample=0,
                end_sample=self.sample_rate * chunk_size * num_chunks
            )
            self.instructions = [default_instruction]
            self.final_sample = 0
            self.is_compiled = True
            return
            
        # Sort instructions by start sample
        self.instructions.sort(key=lambda x: x.start_sample)

        # Calculate the final sample index
        stop_sample = chunk_size * num_chunks
        
        # Check that adjacent samples do not overlap, and fill gaps between instructions
        compiled_instructions = []

        # If the first instruction does not start at sample 0, fill the gap with the default value
        if self.instructions[0].start_sample > 0:
            compiled_instructions.append(
                Instruction(
                    func=partial(const, value=self.default_value),
                    start_sample=0,
                    end_sample=self.instructions[0].start_sample,
                    inplace=const._inplace
                )
            )

        last_value = self.default_value
        for i in range(len(self.instructions) - 1):
            current = self.instructions[i]
            next = self.instructions[i + 1]
            
            if current.end_sample > next.start_sample:
                raise ValueError(f"Instruction {current} overlaps with {next}")
            else:
                # Compile the current command into an instruction
                inst, last_value = self._compile_command(current, last_value)

                # Add the current instruction to the compiled instructions
                compiled_instructions.append(inst)

                # If there is a gap to the next instruction, fill it with the last value
                if current.end_sample < next.start_sample:
                    compiled_instructions.append(
                        Instruction(
                            func=partial(const, value=last_value),
                            start_sample=current.end_sample,
                            end_sample=next.start_sample,
                            inplace=const._inplace
                        )
                    )

        # Compile the last command into an instruction
        inst, last_value = self._compile_command(self.instructions[-1], last_value)
        compiled_instructions.append(inst)

        # If the last instruction does not end at the stop sample, fill the gap with the last value
        last_inst_sample = self.instructions[-1].end_sample
        if last_inst_sample < stop_sample:
            compiled_instructions.append(
                Instruction(
                    func=partial(const, value=last_value),
                    start_sample=last_inst_sample,
                    end_sample=stop_sample,
                    inplace=const._inplace
                )
            )
        else:
            raise ValueError(f"Sequence {self} ({last_inst_sample/self.sample_rate}s) is longer than the external sequence length of {stop_sample/self.sample_rate}s.")

        self.instructions = compiled_instructions
        self.final_sample = stop_sample
        self.is_compiled = True

    def __str__(self):
        return f"{self.channel_name}({self.channel_id})" if self.channel_name else f"{self.channel_id}"
    
    def __repr__(self):
        return str(self)

    def clear(self):
        """Clear all instructions and reset compilation status."""
        self.instructions = []
        self.is_compiled = False

    def get_num_samples(self):
        if self.is_compiled:
            return self.final_sample
        else:
            return None

    def __call__(self, t):
        """
        Make the sequence object callable. This could be used to evaluate
        the sequence at a specific time point.
        
        Args:
            t: Time value
            
        Returns:
            Computed channel value at time t
        """
        if not self.is_compiled:
            raise RuntimeError("Sequence must be compiled before evaluating.")
        
        # Convert time to sample index and round to nearest integer
        sample_idx = round(t * self.sample_rate)

        # If the time is outside the sequence, raise an error
        if sample_idx < 0 or sample_idx >= self.final_sample:
            raise ValueError(f"Sequence {self} encountered an error while evaluating at time {t}.")
        
        # Find which instruction contains this sample index
        ins_idx = bisect.bisect_right(self.instructions, sample_idx, key=lambda x: x.start_sample)-1
        if ins_idx < 0:
            raise ValueError(f"Sequence {self} encountered an error while evaluating at time {t}.")
        
        # Calculate relative time within this instruction
        t_ins = (sample_idx - self.instructions[ins_idx].start_sample) / self.sample_rate

        # Evaluate the instruction function at this time
        result = self.instructions[ins_idx].func(t_ins)
        
        # Convert numpy array to scalar if needed
        if hasattr(result, 'item'):
            return result.item()
        else:
            return result

class AOSequence(Sequence):
    '''
    Tracks the instructions for a single analog channel.
    '''
    _command_category = 'analog_output'

    def __init__(self, channel_id: str, sample_rate: int, default_value: float = 0.0, min_value: float = -10.0, max_value: float = 10.0, channel_name: str = ""):
        """
        Initialize an analog output sequence.
        
        Args:
            channel_id (str): Hardware identifier for the channel (e.g., "ao0")
            sample_rate (int): Sample rate in Hz
            default_value (float): Default output value
            min_value (float): Minimum allowed output value
            max_value (float): Maximum allowed output value
            channel_name (str, optional): Name of the channel for operational use (e.g., "AOD Power")
        """
        super().__init__(channel_id, sample_rate, default_value, channel_name)
        self.min_value = min_value
        self.max_value = max_value


class DOSequence(Sequence):
    '''
    Tracks the instructions for a single digital channel.
    '''
    _command_category = 'digital_output'

    def __init__(self, channel_id: str, sample_rate: int, default_value: int = 0, channel_name: str = "", on_state: bool = True):
        """
        Initialize a digital output sequence.
        
        Args:
            channel_id (str): Hardware identifier for the channel (e.g., "do0")
            sample_rate (int): Sample rate in Hz
            default_value (int): Default output value (0 or 1)
            channel_name (str, optional): Name of the channel for operational use (e.g., "AOD TTL")
            on_state (bool): If True, on() maps to high(), if False, on() maps to low()
        """
        super().__init__(channel_id, sample_rate, default_value, channel_name)
        self.on_state = on_state

        if sample_rate != int(10e6):
            raise ValueError(f"Digital channel {self} needs a 10 MHz sample rate, got {sample_rate/1e6} MHz.")

    def on(self, t, duration=None):
        """Sets the digital output to the configured on state (high or low)."""
        if self.on_state:
            return self.high(t, duration)
        else:
            return self.low(t, duration)

    def off(self, t, duration=None):
        """Sets the digital output to the opposite of the configured on state."""
        if self.on_state:
            return self.low(t, duration)
        else:
            return self.high(t, duration)


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    seq = AOSequence(channel_id="ao0", sample_rate=1e6)
    
    seq.const(1.0, 1.0, value=5.0) 
    seq.linramp(3.0, 0.5, start=0, end=10) 
    seq.sine(6.0, 0.75, freq=1, amp=2, phase=0)
    seq.rampto(7.0, 0.5, value=-8)
    
    print(f"Before compilation - {len(seq.instructions)} instructions:")
    for i, instruction in enumerate(seq.instructions):
        start_time = instruction.start_sample / seq.sample_rate
        end_time = instruction.end_sample / seq.sample_rate
        func_name = instruction.func.func.__name__
        print(f"  {i}: {func_name} from {start_time}s to {end_time}s")
    
    seq.compile(chunk_size=1000, num_chunks=8000)  # 8 seconds at 1MHz
    
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
        seq_overlap.compile(chunk_size=1000, num_chunks=5000)
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
