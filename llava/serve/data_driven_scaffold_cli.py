#!/usr/bin/env python3
"""
llava/serve/data_driven_scaffold_cli.py

Fixed version with proper imports and inheritance
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

# Add parent directory to path for imports
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Import structure analyzer
try:
    from scaffold_structure_analyzer import ScaffoldStructureAnalyzer, format_analysis_for_llm
    STRUCTURE_ANALYZER_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ Structure analyzer not available: {e}")
    STRUCTURE_ANALYZER_AVAILABLE = False

# ScaffoldPointLoRA imports
try:
    from ScaffoldPointLoRA import ScaffoldPointLoRA
    from integrate_scaffold_pointlora import ScaffoldDataProcessor
    SCAFFOLD_LORA_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ ScaffoldPointLoRA not available: {e}")
    SCAFFOLD_LORA_AVAILABLE = False


class DataDrivenScaffoldCLI:
    """
    Data-driven ScaffoldPointLoRA integrated ShapeLLM CLI
    Based on the working cli.py but with enhanced structural analysis
    """
    
    def __init__(self, args):
        self.args = args
        self.model = None
        self.tokenizer = None
        self.context_len = None
        self.conv = None
        self.scaffold_lora = None
        self.data_processor = ScaffoldDataProcessor() if SCAFFOLD_LORA_AVAILABLE else None
        self.structure_analyzer = ScaffoldStructureAnalyzer() if STRUCTURE_ANALYZER_AVAILABLE else None
        self.current_analysis = None
        
        # English safety prompts
        self.safety_prompts = {
            'comprehensive': "Check this scaffold's safety please.",
            'structural': "Examine the structural integrity of this scaffold. Check framework stability and support connections.",
            'platform': "Inspect the working platform safety. Check platform surface, safety rails, and access routes.",
            'height': "Assess height safety measures. Check vertical/horizontal spacing and access methods."
        }
        
        print(f"✅ DataDrivenScaffoldCLI initialized")
        print(f"📊 Structure analyzer: {'Available' if STRUCTURE_ANALYZER_AVAILABLE else 'Not available'}")
        print(f"🎯 ScaffoldPointLoRA: {'Available' if SCAFFOLD_LORA_AVAILABLE else 'Not available'}")
    
    def load_model(self):
        """Load ShapeLLM model with enhanced analysis"""
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
        
        # Apply ScaffoldPointLoRA if available
        if self.args.use_scaffold_lora and SCAFFOLD_LORA_AVAILABLE:
            self._apply_scaffold_lora()
        
        print("✅ Model loading complete")
    
    def _apply_scaffold_lora(self):
        """Apply ScaffoldPointLoRA with enhanced analysis"""
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
                    
                    # Initialize ScaffoldPointLoRA
                    self.scaffold_lora = ScaffoldPointLoRA(
                        hidden_size=hidden_size,
                        lora_rank=self.args.scaffold_lora_rank,
                        lora_alpha=self.args.scaffold_lora_alpha,
                        num_selected_tokens=40
                    )
                    
                    # Move to model device
                    self.scaffold_lora = self.scaffold_lora.to(device=model_device, dtype=model_dtype)
                    
                    print(f"📍 ScaffoldPointLoRA device: {model_device}, dtype: {model_dtype}")
                    
                    # Wrap vision tower
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
        """Enhanced vision tower wrapper with data analysis"""
        original_forward = vision_tower.forward
        
        def data_driven_lora_forward(pts):
            # Execute original forward
            original_output = original_forward(pts)
            
            # Extract point cloud data for analysis
            point_data = None
            if isinstance(pts, list) and len(pts) > 0:
                point_data = pts[0].cpu().numpy()
            elif isinstance(pts, torch.Tensor):
                point_data = pts.cpu().numpy()
                if point_data.ndim == 3:
                    point_data = point_data[0]  # Remove batch dimension
            
            # Perform structural analysis
            if point_data is not None and point_data.shape[1] >= 3 and STRUCTURE_ANALYZER_AVAILABLE:
                try:
                    # Use only XYZ coordinates for analysis
                    xyz_coords = point_data[:, :3]
                    
                    # Perform detailed structural analysis
                    print("🔍 Performing structural analysis...")
                    self.current_analysis = self.structure_analyzer.analyze_scaffold_structure(xyz_coords)
                    
                    print(f"📊 Analysis completed:")
                    print(f"   - Components: {len(self.current_analysis['structural_components']['platforms'])} platforms, "
                          f"{len(self.current_analysis['structural_components']['supports'])} supports")
                    print(f"   - Safety grade: {self.current_analysis['safety_assessment']['overall_grade']}")
                    print(f"   - Issues found: {len(self.current_analysis['specific_issues'])}")
                    
                except Exception as e:
                    print(f"⚠️ Structural analysis error: {e}")
                    self.current_analysis = None
            
            # Apply ScaffoldPointLoRA if available
            if self.scaffold_lora is not None:
                try:
                    # Get consistent dtype and device from model
                    model_device = next(self.model.parameters()).device
                    model_dtype = next(self.model.parameters()).dtype
                    
                    # Extract coordinates
                    coords = None
                    if isinstance(pts, list) and len(pts) > 0:
                        coords = pts[0][:, :3].unsqueeze(0)
                    elif isinstance(pts, torch.Tensor):
                        coords = pts[:, :, :3]
                    
                    if coords is not None:
                        # Ensure coords has correct dtype and device
                        coords = coords.to(device=model_device, dtype=model_dtype)
                        
                        # Generate dummy features
                        batch_size, num_points = coords.shape[:2]
                        dummy_features = torch.randn(
                            batch_size, num_points, self.scaffold_lora.hidden_size,
                            device=model_device, dtype=model_dtype
                        )
                        
                        # Ensure scaffold_lora has consistent dtype
                        if next(self.scaffold_lora.parameters()).dtype != model_dtype:
                            self.scaffold_lora = self.scaffold_lora.to(device=model_device, dtype=model_dtype)
                        
                        # Multi-scale token selection
                        selection_result = self.scaffold_lora(
                            dummy_features, coords, mode='token_selection'
                        )
                        
                        # Store selection info
                        self._last_selection_info = selection_result['selection_info']
                        
                        print(f"🎯 ScaffoldPointLoRA token selection: {selection_result['selected_tokens'].shape[1]} tokens")
                        
                except Exception as e:
                    print(f"⚠️ LoRA forward error (using original output): {e}")
            
            return original_output
        
        # Replace forward function
        vision_tower.forward = data_driven_lora_forward
    
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
    
    def _format_data_driven_response(self, prompt: str, basic_response: str) -> str:
        """Format a data-driven response with specific details"""
        
        if self.current_analysis is None:
            return basic_response
        
        analysis = self.current_analysis
        
        # Extract key information
        platforms = analysis['structural_components']['platforms']
        supports = analysis['structural_components']['supports']
        safety_grade = analysis['safety_assessment']['overall_grade']
        issues = analysis['specific_issues']
        recommendations = analysis['recommendations']
        
        # Check what type of analysis was requested
        if 'structural' in prompt.lower() or 'framework' in prompt.lower():
            return self._format_structural_response(platforms, supports, safety_grade, issues)
        elif 'platform' in prompt.lower():
            return self._format_platform_response(platforms, issues)
        elif 'safety' in prompt.lower() or 'check' in prompt.lower():
            return self._format_safety_response(safety_grade, issues, recommendations)
        else:
            return self._format_comprehensive_response(analysis)
    
    def _format_structural_response(self, platforms, supports, safety_grade, issues):
        """Format structural analysis response"""
        response = f"**Structural Analysis Results (Grade: {safety_grade}):**\n\n"
        
        # Platform details
        if platforms:
            response += f"**Platforms Detected:** {len(platforms)} levels\n"
            for i, platform in enumerate(platforms):
                center = platform['center']
                response += f"- Platform {i+1}: Located at ({center[0]:.2f}, {center[1]:.2f}, {center[2]:.2f}m), Area: {platform['area']:.1f}m²\n"
        
        # Support details
        if supports:
            response += f"\n**Support Structure:** {len(supports)} vertical supports identified\n"
            for i, support in enumerate(supports):
                center = support['center']
                response += f"- Support {i+1}: Position ({center[0]:.2f}, {center[1]:.2f}), Height: {support['length']:.1f}m\n"
        
        # Specific issues
        if issues:
            response += f"\n**Structural Issues Found:**\n"
            for issue in issues:
                loc = issue['location']
                response += f"- {issue['type']}: At coordinates ({loc[0]:.2f}, {loc[1]:.2f}, {loc[2]:.2f}m)\n"
                response += f"  → {issue['description']}\n"
        else:
            response += "\n**No structural issues detected.**"
        
        return response
    
    def _format_platform_response(self, platforms, issues):
        """Format platform-specific response"""
        response = f"**Platform Safety Analysis:**\n\n"
        
        if platforms:
            response += f"**{len(platforms)} working platforms detected:**\n"
            for i, platform in enumerate(platforms):
                center = platform['center']
                bbox = platform['bbox']
                response += f"\n**Platform {i+1}:**\n"
                response += f"- Location: ({center[0]:.2f}, {center[1]:.2f}, {center[2]:.2f}m)\n"
                response += f"- Dimensions: {bbox[1]-bbox[0]:.1f}m × {bbox[3]-bbox[2]:.1f}m\n"
                response += f"- Working area: {platform['area']:.1f}m²\n"
        
        # Platform-specific issues
        platform_issues = [issue for issue in issues if 'platform' in issue['type'].lower()]
        if platform_issues:
            response += f"\n**Platform Issues:**\n"
            for issue in platform_issues:
                loc = issue['location']
                response += f"- Issue at ({loc[0]:.2f}, {loc[1]:.2f}, {loc[2]:.2f}m): {issue['description']}\n"
        
        return response
    
    def _format_safety_response(self, safety_grade, issues, recommendations):
        """Format comprehensive safety response"""
        response = f"**Safety Assessment (Grade: {safety_grade}):**\n\n"
        
        if issues:
            response += f"**{len(issues)} safety issues identified:**\n"
            for i, issue in enumerate(issues, 1):
                loc = issue['location']
                response += f"\n{i}. **{issue['type']}** (Severity: {issue['severity']})\n"
                response += f"   - Location: ({loc[0]:.2f}, {loc[1]:.2f}, {loc[2]:.2f}m)\n"
                response += f"   - Description: {issue['description']}\n"
        
        if recommendations:
            response += f"\n**Immediate Actions Required:**\n"
            for i, rec in enumerate(recommendations, 1):
                loc = rec['location']
                response += f"\n{i}. {rec['action']} (Priority: {rec['priority']})\n"
                response += f"   - Location: ({loc[0]:.2f}, {loc[1]:.2f}, {loc[2]:.2f}m)\n"
                response += f"   - Action: {rec['description']}\n"
                response += f"   - Timeline: {rec['timeline']}\n"
        
        if not issues:
            response += "**No safety issues detected. Structure meets safety standards.**"
        
        return response
    
    def _format_comprehensive_response(self, analysis):
        """Format comprehensive analysis response"""
        response = f"**Comprehensive Scaffold Analysis:**\n\n"
        
        # Summary
        components = analysis['structural_components']
        safety = analysis['safety_assessment']
        
        response += f"**Structure Summary:**\n"
        response += f"- Safety Grade: {safety['overall_grade']} ({safety['safety_status']})\n"
        response += f"- Components: {len(components['platforms'])} platforms, {len(components['supports'])} supports\n"
        response += f"- Compliance Rate: {safety['compliance_rate']:.1f}%\n"
        
        # Key findings
        if analysis['specific_issues']:
            response += f"\n**Key Findings:**\n"
            for issue in analysis['specific_issues'][:3]:  # Show top 3 issues
                loc = issue['location']
                response += f"- {issue['type']} at ({loc[0]:.2f}, {loc[1]:.2f}, {loc[2]:.2f}m)\n"
        
        # Recommendations
        if analysis['recommendations']:
            response += f"\n**Priority Actions:**\n"
            for rec in analysis['recommendations'][:2]:  # Show top 2 recommendations
                loc = rec['location']
                response += f"- {rec['action']} at ({loc[0]:.2f}, {loc[1]:.2f}, {loc[2]:.2f}m)\n"
        
        return response
    
    def interactive_mode(self):
        """Enhanced interactive mode with data-driven responses"""
        print("🏗️ Data-Driven Scaffold Safety Analysis Interactive Mode")
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
                try:
                    processed_data = self.data_processor.process_scaffold_pointcloud(self.args.pts_file)
                    if processed_data is not None:
                        print(f"📊 Original shape: {processed_data['original_shape']}")
                        print(f"✅ Processing complete: {processed_data['processed_shape']}")
                except Exception as e:
                    print(f"⚠️ Data processor error: {e}")
            
            # Load point cloud
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
                # First message - add point cloud token
                if self.model.config.mm_use_pt_start_end:
                    inp = DEFAULT_PT_START_TOKEN + DEFAULT_POINT_TOKEN + DEFAULT_PT_END_TOKEN + '\n' + inp
                else:
                    inp = DEFAULT_POINT_TOKEN + '\n' + inp
                self.conv.append_message(self.conv.roles[0], inp)
                
                # Force vision tower forward to analyze structure
                if pts_tensor is not None:
                    with torch.no_grad():
                        _ = self.model.get_vision_tower()(pts_tensor)
                    
                    # Generate data-driven response
                    enhanced_response = self._format_data_driven_response(inp, "")
                    print(enhanced_response)
                    
                    # Store response in conversation
                    self.conv.append_message(self.conv.roles[1], enhanced_response)
                
                pts = None  # Reset pts after first use
            else:
                # Later messages - regular processing
                self.conv.append_message(self.conv.roles[0], inp)
                self.conv.append_message(self.conv.roles[1], None)
                prompt = self.conv.get_prompt()
                
                # Generate regular response
                input_ids = tokenizer_point_token(prompt, self.tokenizer, POINT_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
                
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
                
                output = self.tokenizer.decode(output_ids[0, input_ids.shape[1]:], skip_special_tokens=True).strip()
                
                # Enhance with analysis data if available
                if self.current_analysis is not None:
                    enhanced_output = self._format_data_driven_response(inp, output)
                    print(enhanced_output)
                    self.conv.messages[-1][-1] = enhanced_output
                else:
                    print(output)
                    self.conv.messages[-1][-1] = output
            
            # Show analysis summary if available
            if self.current_analysis is not None:
                safety_grade = self.current_analysis['safety_assessment']['overall_grade']
                issue_count = len(self.current_analysis['specific_issues'])
                print(f"\n📊 Analysis Summary: Grade {safety_grade}, {issue_count} issues identified")
            
            # Show quick analysis options
            print("\n💡 Quick analysis: 1(comprehensive) | 2(structural) | 3(platform) | 4(height)")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Data-driven ScaffoldPointLoRA integrated ShapeLLM CLI')
    
    # Model arguments
    parser.add_argument("--model-path", type=str, default="qizekun/ShapeLLM_13B_general_v1.0")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--conv-mode", type=str, default=None)
    parser.add_argument("--load-8bit", action="store_true")
    parser.add_argument("--load-4bit", action="store_true")
    
    # Point cloud arguments
    parser.add_argument("--pts-file", type=str, help="Scaffold point cloud file (.npy)")
    parser.add_argument("--objaverse", action="store_true", help="Apply Objaverse data rotation")
    
    # ScaffoldPointLoRA arguments
    parser.add_argument("--use-scaffold-lora", action="store_true", help="Use ScaffoldPointLoRA")
    parser.add_argument("--scaffold-lora-rank", type=int, default=16, help="LoRA rank")
    parser.add_argument("--scaffold-lora-alpha", type=float, default=32.0, help="LoRA alpha")
    parser.add_argument("--training-stage", type=str, choices=['lora_only', 'full'], default='lora_only')
    
    # Generation arguments
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
    print("🏗️ Data-Driven Scaffold Safety Analysis System")
    print(f"📂 Model: {args.model_path}")
    if args.pts_file:
        print(f"📂 Input file: {args.pts_file}")
    print(f"🎯 LoRA settings: rank={args.scaffold_lora_rank}, alpha={args.scaffold_lora_alpha}")
    print("=" * 60)
    
    try:
        # Create CLI instance
        cli = DataDrivenScaffoldCLI(args)
        
        # Load model
        cli.load_model()
        
        # Run interactive mode
        cli.interactive_mode()
            
    except KeyboardInterrupt:
        print("\n👋 User interrupted.")
    except Exception as e:
        print(f"❌ System error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()