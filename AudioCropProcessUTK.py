# AudioCropProcessUTK: Crop/trim audio to specific start and end times
import torch
from typing import Tuple, Dict, Any


class AudioCropProcessUTK:
    """
    A ComfyUI node for cropping (trimming) audio to a specific start and end time.
    """

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "crop_audio"
    CATEGORY = "BSAI"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO", {"description": "Input audio tensor"}),
                "start_time": ("STRING", {
                    "default": "0:00",
                    "description": "Start time (MM:SS or seconds)"
                }),
                "end_time": ("STRING", {
                    "default": "1:00",
                    "description": "End time (MM:SS or seconds)"
                }),
            }
        }

    def crop_audio(
        self,
        audio: Dict[str, Any],
        start_time: str = "0:00",
        end_time: str = "1:00"
    ) -> Tuple[Dict[str, Any]]:
        """
        Crop audio to specific start and end times

        Args:
            audio: Input audio tensor dict with 'waveform' and 'sample_rate'
            start_time: Start time as "MM:SS" or seconds
            end_time: End time as "MM:SS" or seconds

        Returns:
            Tuple containing cropped audio dict
        """
        try:
            waveform = audio['waveform']
            sample_rate = audio['sample_rate']

            # If no ":" in input, assume user is specifying seconds
            if ":" not in start_time:
                start_time = f"00:{start_time}"
            if ":" not in end_time:
                end_time = f"00:{end_time}"

            # Parse start time
            start_seconds = 60 * int(start_time.split(":")[0]) + int(start_time.split(":")[1])
            start_frame = start_seconds * sample_rate

            # Parse end time
            end_seconds = 60 * int(end_time.split(":")[0]) + int(end_time.split(":")[1])
            end_frame = end_seconds * sample_rate

            # Clamp to valid range
            total_frames = waveform.shape[-1]
            start_frame = max(0, min(start_frame, total_frames - 1))
            end_frame = max(0, min(end_frame, total_frames - 1))

            if start_frame >= end_frame:
                raise ValueError(
                    f"AudioCropProcessUTK: Start time ({start_time}) must be less than end time ({end_time}) "
                    f"and be within the audio length."
                )

            # Crop waveform
            cropped_waveform = waveform[..., start_frame:end_frame]

            cropped_audio = {
                'waveform': cropped_waveform,
                'sample_rate': sample_rate
            }

            return (cropped_audio,)

        except Exception as e:
            print(f"[AudioCropProcessUTK] ERROR: {str(e)}")
            return (audio,)

NODE_CLASS_MAPPINGS = {
    "AudioCropProcessUTK": AudioCropProcessUTK,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AudioCropProcessUTK": "AudioCropProcessUTK",
}
