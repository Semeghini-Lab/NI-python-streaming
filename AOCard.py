from typing import Dict
from AOChannel import AOChannel

class AOCard:
    """Class for managing multiple analog output channels.
    
    This class acts as a container for multiple AOChannel instances, allowing
    centralized management of all analog outputs on a card/device.
    """
    
    def __init__(self, samp_rate=10e6):
        """Initialize an analog output card.
        
        Args:
            samp_rate (int): Sample rate in Hz for all channels on the card
        """
        self.samp_rate = int(samp_rate)
        self.channels: Dict[int, AOChannel] = {}  # Maps channel number to AOChannel instance
        
    def add_channel(self, channel_num: int, default_val: float = 0.0) -> AOChannel:
        """Add a new analog output channel to the card.
        
        Args:
            channel_num (int): Channel number/identifier
            default_val (float): Default output value for the channel
            
        Returns:
            AOChannel: The newly created channel instance
            
        Raises:
            ValueError: If channel_num already exists
        """
        if channel_num in self.channels:
            raise ValueError(f"Channel {channel_num} already exists")
            
        channel = AOChannel(samp_rate=self.samp_rate, default_val=default_val)
        self.channels[channel_num] = channel
        return channel
        
    def compile(self):
        """Compile all channels.
        
        This ensures all instructions across all channels are valid
        by calling compile() on each channel.
        
        Raises:
            ValueError: If any channel has overlapping instructions
        """
        for channel in self.channels.values():
            channel.compile()