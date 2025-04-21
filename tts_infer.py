#!/usr/bin/env python
# coding: utf-8

import os
import soundfile as sf
import argparse
from pathlib import Path
from omegaconf import OmegaConf

from f5_tts.infer.utils_infer import (
    load_model,
    load_vocoder,
    preprocess_ref_audio_text,
    infer_process,
    remove_silence_for_generated_wav
)

# =============================================================================
# MAIN PARAMETERS TO EDIT
# =============================================================================
# Model settings
MODEL = "F5TTS_Base"
VOCAB_FILE = "model/vocab.txt"
CKPT_FILE = "model/model_500000.pt"
VOCODER_NAME = "vocos"

# Audio settings
REF_AUDIO = "voice_samples/huong_giang_4.wav"
REF_TEXT = "Bây giờ anh phải cho em một cái hẹn với chị chứ còn biết làm sao nữa à Em bảo rồi rõ ràng là quyền lợi của nhà mình này Đi cà phê thì em mời nước Mà anh chị cứ sợ như kiểu là em gặp anh chị để em ăn thịt Anh chị không bằng ý."
GEN_TEXT = "Không biết căn hộ nhà mình ở Home City của anh chị có nhu cầu bán hoặc cho thuê không ạ?"
SPEED = 1.0

# Output settings
OUTPUT_DIR = "outputs"
OUTPUT_FILE = "synthesized_speech_huong_giang_4.wav"

# =============================================================================
# ADVANCED PARAMETERS (leave as default unless you know what you're doing)
# =============================================================================
DEFAULT_NFE_STEP = 32
DEFAULT_CFG_STRENGTH = 2.0
DEFAULT_TARGET_RMS = 0.1
DEFAULT_CROSS_FADE_DURATION = 0.15
DEFAULT_SWAY_SAMPLING_COEF = -1.0

def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description="F5-TTS Inference Script")
    
    # Main parameters group
    main_group = parser.add_argument_group('Main Parameters')
    main_group.add_argument("--model", type=str, default=MODEL, help="The model name")
    main_group.add_argument("--ref_audio", type=str, default=REF_AUDIO, help="The reference audio file")
    main_group.add_argument("--ref_text", type=str, default=REF_TEXT, help="The transcript for the reference audio")
    main_group.add_argument("--gen_text", type=str, default=GEN_TEXT, help="The text to synthesize")
    main_group.add_argument("--speed", type=float, default=SPEED, help="The speed of the generated audio")
    main_group.add_argument("--vocoder_name", type=str, choices=["vocos", "bigvgan"], default=VOCODER_NAME, help="Vocoder to use")
    main_group.add_argument("--vocab_file", type=str, default=VOCAB_FILE, help="Path to vocab file")
    main_group.add_argument("--ckpt_file", type=str, default=CKPT_FILE, help="Path to model checkpoint")
    main_group.add_argument("--output_file", type=str, default=OUTPUT_FILE, help="Output file name")
    main_group.add_argument("--output_dir", type=str, default=OUTPUT_DIR, help="Output directory")
    
    # Advanced parameters group
    adv_group = parser.add_argument_group('Advanced Parameters')
    adv_group.add_argument("--nfe_step", type=int, default=DEFAULT_NFE_STEP, help="Number of function evaluation steps")
    adv_group.add_argument("--cfg_strength", type=float, default=DEFAULT_CFG_STRENGTH, help="Classifier-free guidance strength")
    adv_group.add_argument("--target_rms", type=float, default=DEFAULT_TARGET_RMS, help="Target RMS value for audio normalization")
    adv_group.add_argument("--cross_fade_duration", type=float, default=DEFAULT_CROSS_FADE_DURATION, help="Cross-fade duration in seconds")
    adv_group.add_argument("--sway_sampling_coef", type=float, default=DEFAULT_SWAY_SAMPLING_COEF, help="Sway sampling coefficient")
    adv_group.add_argument("--remove_silence", action="store_true", help="Remove silence from output")
    
    args = parser.parse_args()

    # Create output directory if it doesn't exist
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    wave_path = output_dir / args.output_file

    # Load vocoder
    print(f"Loading vocoder: {args.vocoder_name}")
    vocoder = load_vocoder(vocoder_name=args.vocoder_name)

    # Get model configuration
    model_cfg_path = None
    try:
        from importlib.resources import files
        model_cfg_path = str(files("f5_tts").joinpath(f"configs/{args.model}.yaml"))
    except (ImportError, FileNotFoundError):
        # Try to find the config in the current directory
        if os.path.exists(f"configs/{args.model}.yaml"):
            model_cfg_path = f"configs/{args.model}.yaml"
        else:
            raise FileNotFoundError(f"Could not find model config for {args.model}")

    model_cfg = OmegaConf.load(model_cfg_path).model
    
    # Dynamically load model class
    if model_cfg.backbone == "DiT":
        from f5_tts.model import DiT
        model_cls = DiT
    elif model_cfg.backbone == "UNetT":
        from f5_tts.model import UNetT
        model_cls = UNetT
    else:
        raise ValueError(f"Unknown model backbone: {model_cfg.backbone}")

    # Load TTS model
    print(f"Loading model from {args.ckpt_file}")
    ema_model = load_model(
        model_cls, 
        model_cfg.arch, 
        args.ckpt_file, 
        mel_spec_type=args.vocoder_name,
        vocab_file=args.vocab_file
    )

    # Preprocess reference audio and text
    print("Preprocessing reference audio and text")
    ref_audio, ref_text = preprocess_ref_audio_text(args.ref_audio, args.ref_text)

    # Perform inference
    print(f"Generating speech for: {args.gen_text}")
    audio_segment, final_sample_rate, spectrogram = infer_process(
        ref_audio,
        ref_text,
        args.gen_text,
        ema_model,
        vocoder,
        mel_spec_type=args.vocoder_name,
        target_rms=args.target_rms,
        cross_fade_duration=args.cross_fade_duration,
        nfe_step=args.nfe_step,
        cfg_strength=args.cfg_strength,
        sway_sampling_coef=args.sway_sampling_coef,
        speed=args.speed,
        fix_duration=None,
    )

    # Save the audio
    print(f"Saving audio to {wave_path}")
    sf.write(wave_path, audio_segment, final_sample_rate)
    
    # Remove silence if requested
    if args.remove_silence:
        print("Removing silence from generated audio")
        remove_silence_for_generated_wav(wave_path)
    
    print(f"Done! Output saved to {wave_path}")

if __name__ == "__main__":
    main() 