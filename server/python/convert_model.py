"""
Unified Model Converter
Converts PyTorch models to various formats: ONNX, TorchScript, CoreML, TFLite
Usage: python convert_model.py <model.pth> <format> <num_classes>
"""

import sys
import os
import json
import subprocess


def convert_model(model_path, format_type, num_classes):
    """Convert PyTorch model to specified format"""
    
    # Get directory paths - now we're in server/python/
    script_dir = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.join(os.path.dirname(script_dir), 'trainedModel')
    
    # Debug: Print paths
    print(f"DEBUG: Script directory: {script_dir}", file=sys.stderr)
    print(f"DEBUG: Models directory: {models_dir}", file=sys.stderr)
    print(f"DEBUG: Model path: {model_path}", file=sys.stderr)
    print(f"DEBUG: Format: {format_type}", file=sys.stderr)
    
    # Determine output file
    model_name = os.path.splitext(os.path.basename(model_path))[0]
    
    format_map = {
        'onnx': ('export_to_onnx.py', f'{model_name}.onnx'),
        'torchscript': ('export_to_torchscript.py', f'{model_name}.pt'),
        'coreml': ('export_to_coreml.py', f'{model_name}.mlmodel'),
        'tflite': ('export_to_tflite.py', f'{model_name}.tflite')
    }
    
    if format_type not in format_map:
        return {
            'success': False,
            'error': f'Unsupported format: {format_type}. Supported: {", ".join(format_map.keys())}'
        }
    
    script_name, output_filename = format_map[format_type]
    export_script = os.path.join(script_dir, script_name)
    output_path = os.path.join(models_dir, output_filename)
    
    # Debug: Print resolved paths
    print(f"DEBUG: Export script: {export_script}", file=sys.stderr)
    print(f"DEBUG: Output path: {output_path}", file=sys.stderr)
    print(f"DEBUG: Script exists: {os.path.exists(export_script)}", file=sys.stderr)
    
    # Check if export script exists
    if not os.path.exists(export_script):
        return {
            'success': False,
            'error': f'Export script not found: {script_name}',
            'searched_path': export_script
        }
    
    # Run the export script
    try:
        # TFLite conversion needs more time due to ONNX->TF->TFLite pipeline
        timeout_duration = 300 if format_type == 'tflite' else 60
        
        print(f"DEBUG: Running command: {sys.executable} {export_script} {model_path} {output_path} {num_classes}", file=sys.stderr)
        result = subprocess.run(
            [sys.executable, export_script, model_path, output_path, str(num_classes)],
            capture_output=True,
            text=True,
            timeout=timeout_duration
        )
        
        print(f"DEBUG: Process return code: {result.returncode}", file=sys.stderr)
        print(f"DEBUG: Process stdout: {result.stdout[:200] if result.stdout else 'None'}", file=sys.stderr)
        print(f"DEBUG: Process stderr: {result.stderr[:200] if result.stderr else 'None'}", file=sys.stderr)
        
        # Parse JSON output from the export script
        if result.stdout:
            try:
                lines = result.stdout.strip().split('\n')
                last_line = lines[-1]
                output_json = json.loads(last_line)
                return output_json
            except json.JSONDecodeError:
                # If JSON parsing fails, return raw output
                return {
                    'success': result.returncode == 0,
                    'output': result.stdout,
                    'error': result.stderr if result.returncode != 0 else None
                }
        
        if result.returncode != 0:
            return {
                'success': False,
                'error': f'Conversion failed',
                'stderr': result.stderr,
                'stdout': result.stdout
            }
        
        return {
            'success': True,
            'output_path': output_path
        }
        
    except subprocess.TimeoutExpired:
        return {
            'success': False,
            'error': f'Conversion timed out ({timeout_duration}s limit)'
        }
    except Exception as e:
        return {
            'success': False,
            'error': str(e)
        }


if __name__ == '__main__':
    if len(sys.argv) < 4:
        print(json.dumps({
            'success': False,
            'error': 'Usage: python convert_model.py <model.pth> <format> <num_classes>'
        }))
        sys.exit(1)
    
    model_path = sys.argv[1]
    format_type = sys.argv[2].lower()
    num_classes = int(sys.argv[3])
    
    result = convert_model(model_path, format_type, num_classes)
    print(json.dumps(result))
    
    if not result.get('success'):
        sys.exit(1)
