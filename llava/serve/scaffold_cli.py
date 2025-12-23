#!/usr/bin/env python3
"""
llava/serve/scaffold_cli.py

Final fixed version with dtype consistency and proper error handling
"""

import torch
import argparse
import numpy as np
import json
import time
import warnings
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

# ShapeLLM imports - based on working cli.py
from transformers import TextStreamer
from llava.utils import disable_torch_init
from llava.model.builder import load_pretrained_model
from llava.conversation import conv_templates, SeparatorStyle
from llava.constants import POINT_TOKEN_INDEX, DEFAULT_POINT_TOKEN, DEFAULT_PT_START_TOKEN, DEFAULT_PT_END_TOKEN
from llava.mm_utils import load_pts, process_pts, rotation, tokenizer_point_token, get_model_name_from_path, KeywordsStoppingCriteria

# ScaffoldPointLoRA imports
try:
    from ScaffoldPointLoRA import ScaffoldPointLoRA
    from integrate_scaffold_pointlora import ScaffoldDataProcessor
    SCAFFOLD_LORA_AVAILABLE = True
except ImportError:
    print("⚠️ ScaffoldPointLoRA not available, using base ShapeLLM")
    SCAFFOLD_LORA_AVAILABLE = False


class ScaffoldSafetyCLI:
    """
    Final fixed ScaffoldPointLoRA integrated ShapeLLM CLI
    """
    
    def __init__(self, args):
        self.args = args
        self.model = None
        self.tokenizer = None
        self.context_len = None
        self.conv = None
        self.scaffold_lora = None
        self.data_processor = ScaffoldDataProcessor() if SCAFFOLD_LORA_AVAILABLE else None
        
        # English safety prompts
        self.safety_prompts = {
            'comprehensive': "Check this scaffold's safety please.",
            'structural': "Examine the structural integrity of this scaffold. Check framework stability and support connections.",
            'platform': "Inspect the working platform safety. Check platform surface, safety rails, and access routes.",
            'height': "Assess height safety measures. Check vertical/horizontal spacing and access methods."
        }
        
        print(f"✅ ScaffoldSafetyCLI initialized")
    
    def load_model(self):
        """Load ShapeLLM model with fixed ScaffoldPointLoRA"""
        print("📦 Loading ShapeLLM model...")
        
        # Model loading - exactly like cli.py
        disable_torch_init()
        
        model_name = get_model_name_from_path(self.args.model_path)
        self.tokenizer, self.model, self.context_len = load_pretrained_model(
            self.args.model_path, 
            self.args.model_base, 
            model_name, 
            self.args.load_8bit,
            self.args.load_4bit, 
            device=self.args.device
        )
        
        print(f"✅ Model loaded on device: {next(self.model.parameters()).device}")
        
        # Conversation setup
        conv_mode = "scaffold_safety"
        if conv_mode not in conv_templates:
            conv_mode = "llava_sw"
            print(f"⚠️ scaffold_safety template not found, using {conv_mode}")
        
        if self.args.conv_mode is not None and conv_mode != self.args.conv_mode:
            print(f'[WARNING] auto inferred conv mode is {conv_mode}, using {self.args.conv_mode}')
            conv_mode = self.args.conv_mode
        
        self.conv = conv_templates[conv_mode].copy()
        print(f"✅ Conversation mode: {conv_mode}")
        
        # Apply fixed ScaffoldPointLoRA
        if self.args.use_scaffold_lora and SCAFFOLD_LORA_AVAILABLE:
            self._apply_scaffold_lora()
        
        print("✅ Model loading complete")
    
    def _apply_scaffold_lora(self):
        """Apply fixed ScaffoldPointLoRA with proper dtype handling"""
        try:
            print("🔧 Applying ScaffoldPointLoRA...")
            
            # Get vision tower
            if hasattr(self.model, 'get_vision_tower'):
                vision_tower = self.model.get_vision_tower()
                if vision_tower is not None:
                    # Get hidden size
                    hidden_size = getattr(vision_tower, 'hidden_size', 1024)
                    
                    # Get model's dtype and device
                    model_device = next(self.model.parameters()).device
                    model_dtype = next(self.model.parameters()).dtype
                    
                    # Initialize ScaffoldPointLoRA with proper dtype
                    self.scaffold_lora = ScaffoldPointLoRA(
                        hidden_size=hidden_size,
                        lora_rank=self.args.scaffold_lora_rank,
                        lora_alpha=self.args.scaffold_lora_alpha,
                        num_selected_tokens=40
                    )
                    
                    # Move to model device with proper dtype
                    self.scaffold_lora = self.scaffold_lora.to(device=model_device, dtype=model_dtype)
                    
                    print(f"📍 ScaffoldPointLoRA device: {model_device}, dtype: {model_dtype}")
                    
                    # Wrap vision tower with fixed dtype handling
                    self._wrap_vision_tower_with_lora(vision_tower)
                    
                    # Freeze parameters if needed
                    if self.args.training_stage == 'lora_only':
                        self._freeze_non_lora_parameters()
                    
                    # Print statistics
                    self._print_parameter_stats()
                    
                    print("✅ ScaffoldPointLoRA applied successfully")
                else:
                    print("⚠️ Vision tower not found, skipping LoRA application")
            else:
                print("⚠️ get_vision_tower method not found, skipping LoRA application")
                
        except Exception as e:
            print(f"❌ ScaffoldPointLoRA application failed: {e}")
            print("Continuing with base ShapeLLM...")
    
    def _wrap_vision_tower_with_lora(self, vision_tower):
        """Wrap vision tower with fixed dtype handling"""
        original_forward = vision_tower.forward
        
        def fixed_lora_forward(pts):
            # Execute original forward
            original_output = original_forward(pts)
            
            # Apply ScaffoldPointLoRA with proper dtype handling
            if self.scaffold_lora is not None:
                try:
                    # Get consistent dtype and device from model
                    model_device = next(self.model.parameters()).device
                    model_dtype = next(self.model.parameters()).dtype
                    
                    # Extract coordinates with proper dtype conversion
                    coords = None
                    if isinstance(pts, list) and len(pts) > 0:
                        coords = pts[0][:, :3].unsqueeze(0)
                    elif isinstance(pts, torch.Tensor):
                        coords = pts[:, :, :3]
                    
                    if coords is not None:
                        # Ensure coords has correct dtype and device
                        coords = coords.to(device=model_device, dtype=model_dtype)
                        
                        # Generate dummy features with consistent dtype
                        batch_size, num_points = coords.shape[:2]
                        dummy_features = torch.randn(
                            batch_size, num_points, self.scaffold_lora.hidden_size,
                            device=model_device, dtype=model_dtype
                        )
                        
                        # Ensure scaffold_lora has consistent dtype
                        if next(self.scaffold_lora.parameters()).dtype != model_dtype:
                            self.scaffold_lora = self.scaffold_lora.to(device=model_device, dtype=model_dtype)
                        
                        # Multi-scale token selection with fixed dtype
                        selection_result = self.scaffold_lora(
                            dummy_features, coords, mode='token_selection'
                        )
                        
                        # Store selection info
                        self._last_selection_info = selection_result['selection_info']
                        
                        print(f"🎯 ScaffoldPointLoRA token selection: {selection_result['selected_tokens'].shape[1]} tokens")
                        
                except Exception as e:
                    print(f"⚠️ LoRA forward error (using original output): {e}")
                    # Additional debugging info
                    if hasattr(self, 'scaffold_lora') and self.scaffold_lora is not None:
                        print(f"   scaffold_lora device: {next(self.scaffold_lora.parameters()).device}")
                        print(f"   scaffold_lora dtype: {next(self.scaffold_lora.parameters()).dtype}")
                    if 'coords' in locals():
                        print(f"   coords device: {coords.device}, dtype: {coords.dtype}")
            
            return original_output
        
        # Replace forward function
        vision_tower.forward = fixed_lora_forward
    
    def _freeze_non_lora_parameters(self):
        """Freeze non-LoRA parameters"""
        for param in self.model.parameters():
            param.requires_grad = False
        
        if self.scaffold_lora is not None:
            for param in self.scaffold_lora.get_trainable_parameters():
                param.requires_grad = True
    
    def _print_parameter_stats(self):
        """Print parameter statistics"""
        if self.scaffold_lora is not None:
            total_params = sum(p.numel() for p in self.model.parameters())
            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            lora_params = sum(p.numel() for p in self.scaffold_lora.get_trainable_parameters())
            
            print("=" * 50)
            print("📊 Parameter Statistics")
            print("=" * 50)
            print(f"Total model parameters: {total_params:,}")
            print(f"Trainable parameters: {trainable_params:,}")
            print(f"LoRA parameters: {lora_params:,}")
            print(f"Training efficiency: {trainable_params/total_params:.2%}")
            print(f"Memory savings: {1-(trainable_params/total_params):.2%}")
            print("=" * 50)
    
    def interactive_mode(self):
        """Interactive mode - based on cli.py"""
        print("🏗️ ScaffoldPointLoRA Scaffold Safety Analysis Interactive Mode")
        print("Type 'quit' or 'exit' to exit.")
        print("=" * 50)
        
        # Get conversation roles
        roles = self.conv.roles
        
        # Load point cloud if specified
        pts = None
        pts_tensor = None
        if self.args.pts_file is not None:
            print(f"📂 Processing point cloud: {self.args.pts_file}")
            
            # Process with ScaffoldDataProcessor if available
            if self.data_processor is not None:
                processed_data = self.data_processor.process_scaffold_pointcloud(self.args.pts_file)
                if processed_data is not None:
                    print(f"📊 Original shape: {processed_data['original_shape']}")
                    print(f"✅ Processing complete: {processed_data['processed_shape']}")
            
            # Load point cloud - exactly like cli.py
            pts = load_pts(self.args.pts_file)
            if self.args.objaverse:
                pts[:, :3] = rotation(pts[:, :3], [0, 0, -90])
            pts_tensor = process_pts(pts, self.model.config).unsqueeze(0)
            
            # Ensure consistent dtype and device
            model_device = next(self.model.parameters()).device
            model_dtype = next(self.model.parameters()).dtype
            pts_tensor = pts_tensor.to(device=model_device, dtype=model_dtype)
            
            print(f"✅ Point cloud loaded: {self.args.pts_file}")
        
        # Interactive loop
        while True:
            try:
                inp = input(f"{roles[0]}: ")
            except EOFError:
                inp = ""
            if not inp:
                print("exit...")
                break
            
            # Handle quick analysis options
            if inp in ['1', '2', '3', '4']:
                analysis_map = {
                    '1': 'comprehensive',
                    '2': 'structural', 
                    '3': 'platform',
                    '4': 'height'
                }
                inp = self.safety_prompts[analysis_map[inp]]
            
            print(f"{roles[1]}: ", end="")
            
            if pts is not None:
                # First message - add point cloud token like cli.py
                if self.model.config.mm_use_pt_start_end:
                    inp = DEFAULT_PT_START_TOKEN + DEFAULT_POINT_TOKEN + DEFAULT_PT_END_TOKEN + '\n' + inp
                else:
                    inp = DEFAULT_POINT_TOKEN + '\n' + inp
                self.conv.append_message(self.conv.roles[0], inp)
                pts = None  # Reset pts after first use
            else:
                # Later messages
                self.conv.append_message(self.conv.roles[0], inp)
            
            self.conv.append_message(self.conv.roles[1], None)
            prompt = self.conv.get_prompt()
            
            # Tokenize - exactly like cli.py
            input_ids = tokenizer_point_token(prompt, self.tokenizer, POINT_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
            
            # Stopping criteria - exactly like cli.py
            stop_str = self.conv.sep if self.conv.sep_style != SeparatorStyle.TWO else self.conv.sep2
            keywords = [stop_str]
            stopping_criteria = KeywordsStoppingCriteria(keywords, self.tokenizer, input_ids)
            
            # Streaming output - exactly like cli.py
            streamer = TextStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)
            
            # Generate response with proper dtype consistency
            with torch.inference_mode():
                output_ids = self.model.generate(
                    input_ids,
                    points=pts_tensor,
                    do_sample=True if self.args.temperature > 0 else False,
                    temperature=self.args.temperature,
                    top_p=self.args.top_p,
                    num_beams=self.args.num_beams,
                    max_new_tokens=self.args.max_new_tokens,
                    use_cache=True,
                    stopping_criteria=[stopping_criteria],
                    streamer=streamer,
                )
            
            # Process output
            output = self.tokenizer.decode(output_ids[0, input_ids.shape[1]:], skip_special_tokens=True).strip()
            self.conv.messages[-1][-1] = output
            
            # Show quick analysis options
            print("\n💡 Quick analysis: 1(comprehensive) | 2(structural) | 3(platform) | 4(height)")
    
    def single_analysis_mode(self, analysis_type: str, output_file: Optional[str] = None):
        """Single analysis mode with fixed dtype handling"""
        if not self.args.pts_file:
            print("❌ Point cloud file required for single analysis mode")
            return
        
        print(f"🔍 Single analysis mode: {analysis_type}")
        
        # Load point cloud
        print(f"📂 Processing point cloud: {self.args.pts_file}")
        
        # Process with ScaffoldDataProcessor if available
        processed_data = {}
        if self.data_processor is not None:
            processed_data = self.data_processor.process_scaffold_pointcloud(self.args.pts_file)
            if processed_data is not None:
                print(f"📊 Original shape: {processed_data['original_shape']}")
                print(f"✅ Processing complete: {processed_data['processed_shape']}")
        
        # Load point cloud
        pts = load_pts(self.args.pts_file)
        if self.args.objaverse:
            pts[:, :3] = rotation(pts[:, :3], [0, 0, -90])
        pts_tensor = process_pts(pts, self.model.config).unsqueeze(0)
        
        # Ensure consistent dtype and device
        model_device = next(self.model.parameters()).device
        model_dtype = next(self.model.parameters()).dtype
        pts_tensor = pts_tensor.to(device=model_device, dtype=model_dtype)
        
        # Get prompt
        prompt = self.safety_prompts.get(analysis_type, self.safety_prompts['comprehensive'])
        
        # Add point cloud token
        if self.model.config.mm_use_pt_start_end:
            prompt = DEFAULT_PT_START_TOKEN + DEFAULT_POINT_TOKEN + DEFAULT_PT_END_TOKEN + '\n' + prompt
        else:
            prompt = DEFAULT_POINT_TOKEN + '\n' + prompt
        
        self.conv.append_message(self.conv.roles[0], prompt)
        self.conv.append_message(self.conv.roles[1], None)
        prompt_text = self.conv.get_prompt()
        
        # Generate response
        input_ids = tokenizer_point_token(prompt_text, self.tokenizer, POINT_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
        
        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids,
                points=pts_tensor,
                do_sample=True if self.args.temperature > 0 else False,
                temperature=self.args.temperature,
                top_p=self.args.top_p,
                num_beams=self.args.num_beams,
                max_new_tokens=self.args.max_new_tokens,
                use_cache=True
            )
        
        # Decode output
        output = self.tokenizer.decode(output_ids[0, input_ids.shape[1]:], skip_special_tokens=True).strip()
        
        # Create result
        result = {
            'analysis_type': analysis_type,
            'pts_file': self.args.pts_file,
            'processed_data': processed_data,
            'ai_response': output
        }
        
        # Save result if output file specified
        if output_file:
            with open(output_file, 'w') as f:
                json.dump(result, f, indent=2)
            print(f"✅ Result saved to: {output_file}")
        
        # Display result
        self._display_analysis_result(result)
    
    def _display_analysis_result(self, result: Dict[str, Any]):
        """Display analysis result"""
        print("\n" + "=" * 60)
        print("📋 SCAFFOLD SAFETY ANALYSIS RESULT")
        print("=" * 60)
        
        # Basic info
        print(f"📂 Point cloud file: {result['pts_file']}")
        print(f"🔍 Analysis type: {result['analysis_type']}")
        
        # Processed data info
        if result['processed_data']:
            processed_data = result['processed_data']
            print(f"📊 Original shape: {processed_data.get('original_shape', 'N/A')}")
            print(f"📊 Processed shape: {processed_data.get('processed_shape', 'N/A')}")
        
        # ScaffoldPointLoRA info
        if hasattr(self, '_last_selection_info') and self._last_selection_info:
            pointlora_info = self._last_selection_info
            print(f"🎯 ScaffoldPointLoRA selected tokens: {pointlora_info['total_selected']}")
            print(f"   - Global structure: {pointlora_info['global_count']}")
            print(f"   - Components: {pointlora_info['component_count']}")
            print(f"   - Details: {pointlora_info['detail_count']}")
        
        # AI analysis result
        print(f"\n🤖 AI Safety Analysis:")
        print("-" * 40)
        ai_response = result['ai_response']
        for line in ai_response.split('\n'):
            if line.strip():
                print(f"   {line}")
        
        print("\n" + "=" * 60)


def main():
    """Main function - based on cli.py"""
    parser = argparse.ArgumentParser(description='Fixed ScaffoldPointLoRA integrated ShapeLLM CLI')
    
    # Model arguments - same as cli.py
    parser.add_argument("--model-path", type=str, default="qizekun/ShapeLLM_13B_general_v1.0")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--conv-mode", type=str, default=None)
    parser.add_argument("--load-8bit", action="store_true")
    parser.add_argument("--load-4bit", action="store_true")
    
    # Point cloud arguments - same as cli.py
    parser.add_argument("--pts-file", type=str, help="Scaffold point cloud file (.npy)")
    parser.add_argument("--objaverse", action="store_true", help="Apply Objaverse data rotation")
    
    # ScaffoldPointLoRA arguments
    parser.add_argument("--use-scaffold-lora", action="store_true", help="Use ScaffoldPointLoRA")
    parser.add_argument("--scaffold-lora-rank", type=int, default=16, help="LoRA rank")
    parser.add_argument("--scaffold-lora-alpha", type=float, default=32.0, help="LoRA alpha")
    parser.add_argument("--training-stage", type=str, choices=['lora_only', 'full'], default='lora_only')
    
    # Generation arguments - same as cli.py
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--num-beams", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    
    # Execution mode arguments
    parser.add_argument("--analysis-type", type=str,
                       choices=['comprehensive', 'structural', 'platform', 'height'],
                       default='comprehensive', help="Analysis type")
    parser.add_argument("--mode", type=str, choices=['interactive', 'single'], default='interactive',
                       help="Execution mode")
    parser.add_argument("--output", type=str, help="Output file for results (JSON)")
    
    args = parser.parse_args()
    
    # Input validation
    if args.mode == 'single' and not args.pts_file:
        print("❌ single mode requires --pts-file argument")
        return
    
    if args.pts_file and not Path(args.pts_file).exists():
        print(f"❌ File does not exist: {args.pts_file}")
        return
    
    # Initialize and run CLI
    print("🏗️ ScaffoldPointLoRA Scaffold Safety Analysis System")
    print(f"📂 Model: {args.model_path}")
    if args.pts_file:
        print(f"📂 Input file: {args.pts_file}")
    print(f"🎯 LoRA settings: rank={args.scaffold_lora_rank}, alpha={args.scaffold_lora_alpha}")
    print("=" * 60)
    
    try:
        # Create CLI instance
        cli = ScaffoldSafetyCLI(args)
        
        # Load model
        cli.load_model()
        
        # Run based on mode
        if args.mode == 'interactive':
            cli.interactive_mode()
        else:  # single mode
            cli.single_analysis_mode(args.analysis_type, args.output)
            
    except KeyboardInterrupt:
        print("\n👋 User interrupted.")
    except Exception as e:
        print(f"❌ System error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()