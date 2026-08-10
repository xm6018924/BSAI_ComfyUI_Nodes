import torch
import numpy as np

class BSAI_AnyToList:
    """
    Convert any input to a list format.
    Useful for converting single values to lists for batch processing.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input": ("*",),
            },
            "optional": {
                "repeat": ("INT", {"default": 1, "min": 1, "max": 1024, "step": 1}),
            }
        }
    
    RETURN_TYPES = ("*",)
    RETURN_NAMES = ("list_output",)
    OUTPUT_IS_LIST = (True,)
    
    FUNCTION = "convert_to_list"
    CATEGORY = "BSAI/Utils"
    
    def convert_to_list(self, input, repeat=1):
        """
        Convert input to list format.
        
        Args:
            input: Any input value
            repeat: Number of times to repeat the input in the list (default: 1)
        
        Returns:
            A list containing the input value repeated 'repeat' times
        """
        result = []
        
        # Handle different input types
        if isinstance(input, list):
            # If already a list, repeat each element
            for item in input:
                result.extend([item] * repeat)
        elif isinstance(input, torch.Tensor):
            # Handle tensor input
            if input.dim() == 0:
                # Scalar tensor
                result = [input] * repeat
            else:
                # Convert tensor to list
                result = input.tolist() * repeat
        elif isinstance(input, np.ndarray):
            # Handle numpy array
            result = input.tolist() * repeat
        else:
            # For all other types, wrap in list and repeat
            result = [input] * repeat
        
        return (result,)

NODE_CLASS_MAPPINGS = {
    "BSAI_AnyToList": BSAI_AnyToList,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "BSAI_AnyToList": "BSAI Any To List",
}
