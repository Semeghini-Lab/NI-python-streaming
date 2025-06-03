import inspect
from functools import partial
from Commands import *
from Instruction import *
import numpy as np

class AOSequence:

    _commands_loaded = False

    @classmethod
    def _load_commands(cls):
        if cls._commands_loaded:
            return
        try:
            import Commands            
            for name in dir(Commands):
                if not name.startswith('_'):
                    obj = getattr(Commands, name)
                    if callable(obj) and hasattr(obj, '__code__'):
                        sig = inspect.signature(obj)
                        params = list(sig.parameters.keys())
                        params = [p for p in params if p != 't']
                        cls._create_class_method(name, obj, params)
            cls._commands_loaded = True
                    
        except Exception as e:
            print(f"Warning: Could not load Commands: {e}")
            cls._commands_loaded = True

    @classmethod
    def _create_class_method(cls, func_name, command_func, param_names):

        def method(self, t, duration, **kwargs):
            if self.is_compiled:
                raise RuntimeError(f"Cannot add {func_name} instruction: sequence is already compiled. Create a new sequence or clear existing instructions.")
                
            if t < 0:
                raise ValueError(f"Start time must be >= 0, got {t} for {func_name}")
            if duration <= 1.0/self.samp_rate:
                raise ValueError(f"Duration must be > 1/sample_rate ({1.0/self.samp_rate} s), got {duration} for {func_name}")
            
            # validate that all required parameters are provided
            for param in param_names:
                if param not in kwargs:
                    raise ValueError(f"Missing required parameter '{param}' for {func_name}")
            
            start_sample = round(t * self.samp_rate)
            end_sample = start_sample + round(duration * self.samp_rate) # end sample is exclusive
            
            # Simply append instruction - no sorting or overlap checking during insertion
            instruction = Instruction(func=partial(command_func, **kwargs), start_samp=start_sample, end_samp=end_sample)
            self.instructions.append(instruction)
        
        # print(f'Created method {func_name}')
        setattr(cls, func_name, method)

    def __init__(self, samp_rate=1e6, default_value=0.0, min_value=-10.0, max_value=10.0):
        """Initialize an analog output sequence.
        
        Args:
            samp_rate (int): Sample rate in Hz
            default_value (float): Default output value used to fill gaps
            min_value (float): Minimum allowed output value
            max_value (float): Maximum allowed output value
        """
        self.samp_rate = int(samp_rate)
        self.default_value = default_value
        self.min_value = min_value
        self.max_value = max_value
        self.instructions = []
        self.is_compiled = False
        
        self.__class__._load_commands()

    def compile(self, stopsamp):
        """Compile the sequence with the given stop time.
        
        This method:
        1. Sorts instructions by start time
        2. Checks for overlaps between adjacent instructions
        3. Fills gaps with appropriate values
        4. Validates the sequence
        
        Args:
            stopsamp: The stop sample for compilation
            
        Raises:
            ValueError: If instructions overlap
        """
        if not self.instructions:
            # No instructions - fill entire range with default value using const
            default_instruction = Instruction(
                func=partial(const, value=self.default_value),
                start_samp=0,
                end_samp=int(stopsamp)
            )
            self.instructions = [default_instruction]
            self.is_compiled = True
            return
            
        # Sort instructions by start sample - O(m log m) where m is number of instructions
        self.instructions.sort(key=lambda x: x.start_samp)
        
        # Check for overlaps between adjacent instructions - O(m)
        for i in range(len(self.instructions) - 1):
            current = self.instructions[i]
            next_inst = self.instructions[i + 1]
            
            if current.end_samp > next_inst.start_samp:
                current_func = current.func.func.__name__
                next_func = next_inst.func.func.__name__
                raise ValueError(f"Instruction {current_func} (samples {current.start_samp}-{current.end_samp}) "
                               f"overlaps with {next_func} (samples {next_inst.start_samp}-{next_inst.end_samp})")
        
        # Fill gaps between instructions
        filled_instructions = []
        current_sample = 0
        last_value = self.default_value
        
        for instruction in self.instructions:
            # Fill gap before this instruction if needed
            if current_sample < instruction.start_samp:
                # Create const instruction directly to fill gap
                gap_instruction = Instruction(
                    func=partial(const, value=last_value),
                    start_samp=current_sample,
                    end_samp=instruction.start_samp
                )
                filled_instructions.append(gap_instruction)
            
            # Add the actual instruction
            filled_instructions.append(instruction)
            current_sample = instruction.end_samp
            
            # Evaluate the instruction at its end time to get the last value
            duration = (instruction.end_samp - instruction.start_samp) / self.samp_rate
            # Create a single-point time array at the end time (relative to instruction start)
            end_time_array = np.array([duration])  # Time relative to instruction start
            
            # Call the original function directly with its parameters
            original_func = instruction.func.func
            params = instruction.func.keywords
            last_value = original_func(end_time_array, **params)[0]  # Get the single value
        
        # Fill gap after last instruction to stopsamp if needed
        if current_sample < stopsamp:
            # Create const instruction directly to fill final gap
            final_gap_instruction = Instruction(
                func=partial(const, value=last_value),
                start_samp=current_sample,
                end_samp=int(stopsamp)
            )
            filled_instructions.append(final_gap_instruction)
        
        self.instructions = filled_instructions
        self.is_compiled = True

    def clear(self):
        """Clear all instructions and reset compilation status.
        
        This allows the sequence to be reused for building a new set of instructions.
        """
        self.instructions = []
        self.is_compiled = False

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    seq = AOSequence()
    
    seq.const(1.0, 1.0, value=5.0) 
    seq.linramp(3.0, 1.0, start=0, end=10) 
    seq.sine(6.0, 0.75, freq=1, amp=2, phase=0)
    
    print(f"Before compilation - {len(seq.instructions)} instructions:")
    for i, instruction in enumerate(seq.instructions):
        start_time = instruction.start_samp / seq.samp_rate
        end_time = instruction.end_samp / seq.samp_rate
        func_name = instruction.func.func.__name__
        print(f"  {i}: {func_name} from {start_time}s to {end_time}s")
    
    seq.compile(stopsamp=8e6)  # 8 seconds at 1MHz
    
    print(f"\nAfter compilation - {len(seq.instructions)} instructions (gaps filled with const):")
    for i, instruction in enumerate(seq.instructions):
        start_time = instruction.start_samp / seq.samp_rate
        end_time = instruction.end_samp / seq.samp_rate
        func_name = instruction.func.func.__name__
        params = instruction.func.keywords
        print(f"  {i}: {func_name} from {start_time}s to {end_time}s, params: {params}")
    
    # Plot the compiled sequence
    import matplotlib.pyplot as plt
    
    # Create time array for the entire sequence
    total_samples = int(8e6)  # 8 seconds at 1MHz
    t_full = np.linspace(0, 8, total_samples)
    y_full = np.zeros(total_samples)
    
    # Fill in values from each instruction
    for instruction in seq.instructions:
        start_idx = instruction.start_samp
        end_idx = instruction.end_samp
        
        # Create time array for this instruction (relative to instruction start)
        n_samples = end_idx - start_idx
        t_instruction = np.linspace(0, (end_idx - start_idx) / seq.samp_rate, n_samples)
        
        # Evaluate the instruction function
        original_func = instruction.func.func
        params = instruction.func.keywords
        y_instruction = original_func(t_instruction, **params)
        
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
        start_time = instruction.start_samp / seq.samp_rate
        end_time = instruction.end_samp / seq.samp_rate
        func_name = instruction.func.func.__name__
        
        ax.axvline(start_time, color='red', linestyle='--', alpha=0.5)
        if func_name != 'const':  # Don't label every const boundary
            ax.text(start_time + 0.1, ax.get_ylim()[1] * 0.9, func_name, 
                   rotation=90, fontsize=8, alpha=0.7)
    
    plt.tight_layout()
    plt.show()
