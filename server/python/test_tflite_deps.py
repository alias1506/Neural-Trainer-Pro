import sys
import json


def test_imports():
    results = {}
    
    try:
        import google.protobuf
        from google.protobuf import runtime_version
        results['protobuf'] = {
            'version': google.protobuf.__version__,
            'runtime_version': 'OK',
            'status': 'SUCCESS'
        }
    except ImportError as e:
        results['protobuf'] = {'status': 'FAILED', 'error': str(e)}
    
    try:
        import tensorflow as tf
        results['tensorflow'] = {
            'version': tf.__version__,
            'status': 'SUCCESS'
        }
    except ImportError as e:
        results['tensorflow'] = {'status': 'FAILED', 'error': str(e)}
    
    try:
        import onnx
        results['onnx'] = {
            'version': onnx.__version__,
            'status': 'SUCCESS'
        }
    except ImportError as e:
        results['onnx'] = {'status': 'FAILED', 'error': str(e)}
    
    try:
        import onnx2tf
        results['onnx2tf'] = {'status': 'SUCCESS'}
    except ImportError as e:
        results['onnx2tf'] = {'status': 'FAILED', 'error': str(e)}
    
    return results


if __name__ == "__main__":
    results = test_imports()
    print(json.dumps(results, indent=2))
    
    # Check if all succeeded
    all_ok = all(r.get('status') == 'SUCCESS' for r in results.values())
    sys.exit(0 if all_ok else 1)
