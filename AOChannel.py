import numpy as np

class AOChannel:
    """Class for managing analog output channels.
    
    This class manages a sequence of analog output instructions.
    Instructions are specified in real time (seconds) but stored and processed 
    in sample time for efficiency.
    
    Instructions are stored as (start_sample, end_sample, type, params) tuples where:
    - start_sample: Start sample index (inclusive)
    - end_sample: End sample index (exclusive)
    - type: Integer index corresponding to instruction type
    - params: Parameters specific to the instruction type:
        CONST: {'val': float}  # Constant value
        LINRAMP: {'A': float, 'B': float}  # Linear ramp y = A*T + B
        SINE: {'omega': float, 'A': float, 'offset': float, 'phase': float}  # y = A*sin(omega*T + phase) + offset
    """
    
    # Instruction type indices
    CONST = 0    # Constant value
    LINRAMP = 1  # Linear ramp
    SINE = 2     # Sine wave

    def __init__(self, samp_rate=10e6, default_val=0.0):
        """Initialize an analog output channel.
        
        Args:
            samp_rate (int): Sample rate in Hz
            default_val (float): Default output value used to fill gaps
        """
        self.samp_rate = int(samp_rate)
        self.default_val = default_val
        self.instructions = []  # List of (start_sample, end_sample, type, params) tuples where start/end_sample are in sample time
        self.is_compiled = False

    def const(self, t, duration, value):
        """Add a constant value instruction.
        
        Args:
            t (float): Start time in seconds (must be >= 0)
            duration (float): Duration in seconds (must be > 1/sample_rate)
            value (float): Constant value to output
            
        Raises:
            ValueError: If t < 0 or duration <= 1/sample_rate
        """
        if t < 0:
            raise ValueError(f"Start time must be >= 0, got {t}")
        if duration <= 1.0/self.samp_rate:
            raise ValueError(f"Duration must be > 1/sample_rate ({1.0/self.samp_rate} s), got {duration}")
            
        start_sample = int(t * self.samp_rate)
        end_sample = start_sample + int(duration * self.samp_rate)
        self.instructions.append((start_sample, end_sample, self.CONST, {'val': value}))
        self.is_compiled = False

    def linramp(self, t, duration, start_val, end_val):
        """Add a linear ramp instruction.
        
        The ramp is implemented as y = A*T + B where:
        - A is (end_val - start_val) / n_samples
        - B is start_val
        - T is sample count from start of instruction
        
        Args:
            t (float): Start time in seconds (must be >= 0)
            duration (float): Duration in seconds (must be > 1/sample_rate)
            start_val (float): Start value
            end_val (float): End value
            
        Raises:
            ValueError: If t < 0 or duration <= 1/sample_rate
        """
        if t < 0:
            raise ValueError(f"Start time must be >= 0, got {t}")
        if duration <= 1.0/self.samp_rate:
            raise ValueError(f"Duration must be > 1/sample_rate ({1.0/self.samp_rate} s), got {duration}")
            
        start_sample = int(t * self.samp_rate)
        end_sample = start_sample + int(duration * self.samp_rate)
        n_samples = end_sample - start_sample
        A = (end_val - start_val) / n_samples
        self.instructions.append((start_sample, end_sample, self.LINRAMP, {'A': A, 'B': start_val}))
        self.is_compiled = False

    def sine(self, t, duration, freq, amplitude, offset=0.0, phase=0.0):
        """Add a sine wave instruction.
        
        The sine wave is implemented as y = A*sin(omega*T + phase) + offset where:
        - A is the amplitude (half peak-to-peak)
        - omega is 2π * freq / sample_rate (in radians per sample)
        - T is sample count from start of instruction
        - phase is initial phase in radians
        - offset is DC offset
        
        Args:
            t (float): Start time in seconds (must be >= 0)
            duration (float): Duration in seconds (must be > 1/sample_rate)
            freq (float): Frequency in Hz (must be > 0)
            amplitude (float): Peak-to-peak amplitude / 2
            offset (float, optional): DC offset. Defaults to 0.0.
            phase (float, optional): Initial phase in radians. Defaults to 0.0.
            
        Raises:
            ValueError: If t < 0 or duration <= 1/sample_rate or freq <= 0
        """
        if t < 0:
            raise ValueError(f"Start time must be >= 0, got {t}")
        if duration <= 1.0/self.samp_rate:
            raise ValueError(f"Duration must be > 1/sample_rate ({1.0/self.samp_rate} s), got {duration}")
        if freq <= 0:
            raise ValueError(f"Frequency must be > 0, got {freq}")
            
        start_sample = int(t * self.samp_rate)
        end_sample = start_sample + int(duration * self.samp_rate)
        
        # Store frequency in samples for easier computation later
        omega = 2 * np.pi * freq / self.samp_rate
        
        self.instructions.append((
            start_sample, 
            end_sample, 
            self.SINE, 
            {'omega': omega, 'A': amplitude, 'offset': offset, 'phase': phase}
        ))
        self.is_compiled = False

    def compile(self):
        """Check for conflicts between instructions.
        
        This method:
        1. Sorts instructions by sample index
        2. Checks for conflicts (overlapping instructions)
        
        Raises:
            ValueError: If instructions overlap
        """
        if not self.instructions:
            self.is_compiled = True
            return
            
        # Sort instructions by sample index
        self.instructions.sort(key=lambda x: x[0])

        # Check for conflicts - no instruction should overlap with the next one
        instructions = self.instructions  # Local reference for faster access
        for i in range(len(instructions) - 1):
            curr_instr = instructions[i]
            next_instr = instructions[i + 1]
            
            if curr_instr[1] > next_instr[0]:
                raise ValueError(f"Instruction at sample {curr_instr[0]} overlaps with next instruction at sample {next_instr[0]}")
                
        self.is_compiled = True